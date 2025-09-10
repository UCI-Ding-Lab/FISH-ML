# fishgui/gui/thumbnails.py
import time
import tkinter
import pathlib
import numpy as np
from PIL import (Image, ImageTk, ImageDraw)
import tifffile 
import cv2
import threading
from skimage import filters, morphology, measure, segmentation
from scipy import ndimage as ndi
import re
import logging
from skimage.restoration import estimate_sigma
from .canvas_view import box, segment
from ..services.session_loader import bundle
from tkinter import messagebox
from .canvas_view import segment
from ..services.apply_channel_mask import apply_channel_mask_to_frames

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def project_stack_to_2d(stack: np.ndarray, method: str = "max") -> np.ndarray:
    if method == "max":
        return np.max(stack, axis=0)
    elif method == "mean":
        return np.mean(stack, axis=0).astype(stack.dtype)
    elif method == "median":
        return np.median(stack, axis=0).astype(stack.dtype)
    elif method == "focus":
        # pick slice with highest variance of Laplacian (sharpest)
        scores = [cv2.Laplacian(slice_, cv2.CV_64F).var() for slice_ in stack]
        best = int(np.argmax(scores))
        return stack[best]
    else:
        raise ValueError(f"Unknown projection method: {method!r}")

class abstract():
    __pool: list['abstract'] = []
    __buffer: 'abstract' = None
    
    def __init__(
        self,
        sample_id: str,
        nucleus_path: pathlib.Path,
        cyto_paths: list[pathlib.Path],
        gallery_frame,
        gui
    ):
        self.gui = gui
        self.sample_id = sample_id
        self.__nucleus_path = nucleus_path
        self.__cyto_paths = cyto_paths

        nuc_arr = tifffile.imread(nucleus_path)
        if nuc_arr.ndim == 3 and nuc_arr.shape[0] > 1:   
            self.__img_np_nucleus = abstract.preprocess_nucleus_stack(nuc_arr)
        else:                                             
            self.__img_np_nucleus = abstract.normalize_to_uint8(np.squeeze(nuc_arr)) # already z-projected

        self.__img_np_647 = None
        self.__img_np_488 = None
        for p in cyto_paths:
            arr = tifffile.imread(p)
            if arr.ndim == 3 and arr.shape[0] > 1:      
                zprojected = abstract.preprocess_cytoplasm_stack(arr, top_n=8)
            else:                                        
                zprojected = np.squeeze(arr)
            
            stem = p.stem.lower()
            if "647" in stem:
                self.__img_np_647 = zprojected
            elif "488" in stem:
                self.__img_np_488 = abstract.normalize_to_uint8(abstract.remove_outliers(zprojected, k=18.0, use_median=False))
            else:
                logger.warning(f"Unrecognized cytoplasm channel in file {p.name}")

        self.__img_np_cyto1 = self.__img_np_647
        self.__img_np_cyto2 = self.__img_np_488

        self.available_channels = []
        if self.__img_np_647 is not None:
            self.available_channels.append("647")
        if self.__img_np_488 is not None:
            self.available_channels.append("488")
        self.selected_channel = self.available_channels[0] if self.available_channels else "647"

        # Set current channel
        if self.selected_channel == "647":
            self.__img_np_cyto = self.__img_np_647
        else:
            self.__img_np_cyto = self.__img_np_488

        # build thumbnail
        thumbnail_img = None
        if self.__img_np_cyto1 is not None:
            thumbnail_img = self.__img_np_cyto1
        elif self.__img_np_cyto2 is not None:
            thumbnail_img = self.__img_np_cyto2
        else:
            thumbnail_img = self.__img_np_nucleus

        rgb = abstract.grayscale_to_rgb(thumbnail_img)
        self.__img_np_rgb1 = rgb
        pil = Image.fromarray(rgb).resize((64, 64))
        tk_img = ImageTk.PhotoImage(pil)
        self.__img_pil_thumbnail = pil
        self.__img_tk_thumbnail = tk_img

        self.__label = tkinter.Label(gallery_frame,
                                     image=tk_img,
                                     width=64, height=64,
                                     relief=tkinter.FLAT, borderwidth=0)
        self.__label.pack(side=tkinter.LEFT, padx=2, pady=2)
        self.__label.bind("<Button-1>", self.on_click)
        self.__label.bind("<Control-Button-1>", self.on_multi_toggle)
        self.__label.bind("<Command-Button-1>", self.on_multi_toggle) 
        
        self.__img_pil_thumbnail_bbox = None
        self.__img_pil_thumbnail_select = None
        self.__img_pil_thumbnail_crossout = None
        self.__img_pil_thumbnail_segmented = None
        self.__img_tk_thumbnail_bbox = None
        self.__img_tk_thumbnail_select = None
        self.__img_tk_thumbnail_crossout = None
        self.__img_tk_thumbnail_segmented = None
        self.__thumbnail: str = None
        self.__bbox = []
        self.__highlighted: str = None
        self.__selected: bool = False
        self.__drawBbox: bool = False
        self.__seg = []
        self.__drawSeg: bool = False
        self.__bbox_generated: bool = False
        self.__segment_generated: bool = False

        # Initialize channel-specific segment lists
        self.__seg_647 = []
        self.__seg_488 = []

        self.__img_pil_thumbnail_select_bbox = None
        self.__img_tk_thumbnail_select_bbox = None


        abstract.addToPool(self)

    def get_cyto1(self) -> np.ndarray:
        return self.__img_np_cyto1

    def get_cyto2(self) -> np.ndarray:
        return self.__img_np_cyto2

    @property
    def segment(self) -> list:
        if not self.bbox_generated:
            self.gui.popBox("w", "Bounding Boxes Not Ready",
                            "Please generate BBOX before running segmentation.")
            return self.__seg
        
        def job():
            self.run_basic_watershed()  
            self.segment_generated = True
            self.gui.getRoot().after(0, self.gui.dismissWait)

        if not self.segment_generated:
            self.gui.indicateWait("Segmentation")
            self.gui.getRoot().update_idletasks()
            t = threading.Thread(target=job, daemon=True)
            t.start()
            while not self.segment_generated:
                time.sleep(0.1)

        return self.__seg

    @segment.setter
    def segment(self, value):
        self.__seg = value
        self.segment_generated = True if value else False

    @property
    def segmentExplict(self):
        return self.__seg

    @property
    def boundingBoxRevised(self):
        if self.noBbox():
            return []
        return [b.final for b in self.bbox]
    
    @property
    def segmentationRevised(self):
        if self.noSegment():
            return []
        return [s._segment__data.T for s in self.segment]

    @property
    def thumbnail(self) -> str:
        return self.__thumbnail
    @thumbnail.setter
    def thumbnail(self, value: str):
        self.__thumbnail = value
        if value == "default":
            self.getLabel().config(image=self.__img_tk_thumbnail)
        elif value == "bbox":
            if not self.__img_tk_thumbnail_bbox:
                self.__img_pil_thumbnail_bbox = self.__img_pil_thumbnail.copy()
                ImageDraw.Draw(self.__img_pil_thumbnail_bbox).ellipse((49, 5, 59, 15), fill=(0,0,255))
                self.__img_tk_thumbnail_bbox = ImageTk.PhotoImage(self.__img_pil_thumbnail_bbox)
            self.getLabel().config(image=self.__img_tk_thumbnail_bbox)
        elif value == "selected":
            if self.bbox_generated:
                if not self.__img_tk_thumbnail_select_bbox:
                    img = self.__img_pil_thumbnail.copy()
                    draw = ImageDraw.Draw(img)
                    draw.ellipse((49, 5, 59, 15), fill=(0,0,255))  # blue circle
                    draw.ellipse((5, 5, 15, 15), fill=(0,255,0))    # green dot
                    self.__img_tk_thumbnail_select_bbox = ImageTk.PhotoImage(img)
                self.getLabel().config(image=self.__img_tk_thumbnail_select_bbox)
            else:
                # Only green dot
                if not self.__img_tk_thumbnail_select:
                    self.__img_pil_thumbnail_select = self.__img_pil_thumbnail.copy()
                    ImageDraw.Draw(self.__img_pil_thumbnail_select).ellipse((5, 5, 15, 15), fill=(0,255,0))
                    self.__img_tk_thumbnail_select = ImageTk.PhotoImage(self.__img_pil_thumbnail_select)
                self.getLabel().config(image=self.__img_tk_thumbnail_select)
        elif value == "crossout":
            if not self.__img_tk_thumbnail_crossout:
                self.__img_pil_thumbnail_crossout = self.__img_pil_thumbnail.copy()
                ImageDraw.Draw(self.__img_pil_thumbnail_crossout).line((5, 5, 15, 15), fill=(255,0,0), width=2)
                ImageDraw.Draw(self.__img_pil_thumbnail_crossout).line((5, 15, 15, 5), fill=(255,0,0), width=2)
                self.__img_tk_thumbnail_crossout = ImageTk.PhotoImage(self.__img_pil_thumbnail_crossout)
            self.getLabel().config(image=self.__img_tk_thumbnail_crossout)
        elif value == "segmented":
            if self.__img_tk_thumbnail_segmented is None:
                # Ensure we have a bbox base image first
                if self.__img_pil_thumbnail_bbox is None:
                    base = self.__img_pil_thumbnail.copy()
                    ImageDraw.Draw(base).ellipse((49, 5, 59, 15), fill=(0,0,255))
                    self.__img_pil_thumbnail_bbox = base.copy()
                    self.__img_tk_thumbnail_bbox = ImageTk.PhotoImage(self.__img_pil_thumbnail_bbox)
                # Now build the segmented variant
                seg_img = self.__img_pil_thumbnail_bbox.copy()
                ImageDraw.Draw(seg_img).ellipse((49, 20, 59, 30), fill=(255, 165, 0))
                self.__img_pil_thumbnail_segmented = seg_img
                self.__img_tk_thumbnail_segmented = ImageTk.PhotoImage(seg_img)
            self.getLabel().config(image=self.__img_tk_thumbnail_segmented)

    @thumbnail.deleter
    def thumbnail(self):
        self.getLabel().pack_forget()
    
    @property
    def selected(self) -> bool:
        return self.__selected
    @selected.setter
    def selected(self, value: bool):
        if value:
            if not self.bbox_generated:
                self.gui.popBox("w", "BBOX Not Ready", "Bounding boxes for this image have not been generated yet.")
                return
            self.thumbnail = "selected"
            self.__selected = True
        else:
            self.thumbnail = "default"
            self.__selected = False
    @property
    def bbox(self):
        from .canvas_view import box
        if not self.bbox_generated:
            nuc_boxes = self.gui.getBackEnd().AppIntDINOwrapper(self.__img_np_nucleus)
            centers = [
                ((x0 + x1) / 2, (y0 + y1) / 2)
                for x0, y0, x1, y1 in nuc_boxes
            ]
            cyto_boxes = self.gui.getBackEnd().AppIntDINOwrapperB(self.__img_np_cyto, centers)
            self.__bbox = [box(b, self.gui) for b in cyto_boxes]
            self.bbox_generated = True
        return self.__bbox
    
    @property
    def finalized_mask(self):
        return getattr(self, "_finalized_mask", None)
    def set_finalized_mask(self, mask_list):
        self._finalized_mask = mask_list  # mask_list: list of np.ndarray

    @bbox.setter
    def bbox(self, value):
        self.__bbox = value
        self.bbox_generated = True if value else False

    def on_click(self, event):
        self.gui.getStove().bufferSetCurrent(3)       
        self.gui.getStove().dump()
        b = abstract.getBuffer()
        if b: del b.highlighted
        self.highlighted = "red"
        if self.gui.getFuncButton().selectButtonPressed():
            self.selected = not self.selected
        elif self.gui.getFuncButton().bboxButtonPressed():
            buffer = box.getBuffer()
            if buffer: buffer.selected = False
            if b: b.drawBbox = False
            self.drawBbox = True
        elif self.gui.getFuncButton().segButtonPressed():
            if b: b.drawSegmentation = False
            self.drawSegmentation = True
        abstract.setBuffer(self)
        self.gui.getStove().cook(self)
    
    def on_multi_toggle(self, event):
        self.selected = not self.selected

    def _get_seg_list_for_channel(self, ch: str):
        if ch == "647": return getattr(self, "_abstract__seg_647", [])
        if ch == "488": return getattr(self, "_abstract__seg_488", [])
        return []

    def _set_seg_list_for_channel(self, ch: str, seg_objs: list):
        if ch == "647": setattr(self, "_abstract__seg_647", seg_objs)
        elif ch == "488": setattr(self, "_abstract__seg_488", seg_objs)

    @classmethod
    def apply_channel_mask_to_frames(cls, source_channel, selected_frames, target_channels, gui=None):
        apply_channel_mask_to_frames(cls, source_channel, selected_frames, target_channels, gui)
    
    @classmethod
    def sendFirst(cls):
        for target in cls.getPool():
            if target.selected:
                target.on_click(None)
                return
    @classmethod
    def getPool(cls):
        return cls.__pool
    @classmethod
    def addToPool(cls, abs):
        cls.__pool.append(abs)
    @classmethod
    def setBuffer(cls, abs):
        cls.__buffer = abs
    @classmethod
    def getBuffer(cls):
        return cls.__buffer
    @classmethod
    def selectAll(cls):
        for abs in cls.getPool():
            abs.selected = True
    @classmethod
    def removeUnselected(cls):
        for abs in cls.getPool():
            abs.thumbnail = "default"
            if not abs.selected: del abs.thumbnail
        cls.sendFocused()
    @classmethod
    def sendFocused(cls):
        current = cls.getBuffer()
        if current: current.on_click(None)
        else: cls.sendFirst()
    @classmethod
    def saveBboxChanges(cls):
        cls.getBuffer().drawBbox = False
        cls.sendFocused()
    @classmethod
    def saveSegChanges(cls):
        cls.getBuffer().drawSegmentation = False
        cls.sendFocused()

    @classmethod
    def grabPool(cls):
        result = []
        for a in cls.getPool():
            if not getattr(a, "selected", True):
                continue
            # Use bundle to store all relevant info
            b = bundle(
                nucleus_path=a.getNucleusPath(),
                cyto_paths=list(a.getCytoplasmPaths()),
                bbox=a.boundingBoxRevised,
                segment=a.segmentationRevised
            )
            result.append(b)
        return result
    
    @staticmethod
    def grayscale_to_rgb(grayscale_img) -> np.ndarray:
        img_normalized = cv2.normalize(grayscale_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
        brightness_factor = 1
        return np.clip(img_rgb * brightness_factor, 0, 255).astype(np.uint8)
    
    @staticmethod
    def remove_outliers(img, k=20.0, use_median=False):
        """
        Clip values that are more than k std-dev (or MAD units) above center.
        Args:
            img: 16-bit numpy array
            k: threshold (e.g. 3σ)
            use_median: if True use median+MAD, else mean+std
        Returns:
            clipped float32 image in [0,1]
        """

        print("OUTLIERS REMOVING...")
        x = img.astype(np.float32)

        if use_median:
            med = np.median(x)
            mad = np.median(np.abs(x - med)) + 1e-6
            sigma = 1.4826 * mad  # robust std estimate
            thresh = med + k * sigma
        else:
            mean = np.mean(x)
            std = np.std(x)
            thresh = mean + k * std

        # clip outliers
        x_clipped = np.minimum(x, thresh)

        # normalize after clipping (to 0..1)
        x_norm = (x_clipped - x_clipped.min()) / (x_clipped.max() - x_clipped.min() + 1e-6)
        return x_norm
    
    @classmethod
    def segment_selected(cls, gui):
        selected = [a for a in cls.getPool() if a.selected]
        print(f"Segmenting {len(selected)} images: {[str(a.getNucleusPath().name) for a in selected]}")

        not_ready = [a for a in selected if not a.bbox_generated]
        if not_ready:
            names = ", ".join(getattr(a, "sample_id", "?") for a in not_ready)
            gui.popBox("w", "BBOX Not Ready",
                       f"Skipping segmentation for: {names} (BBOX still not ready).")
        ready = [a for a in selected if a.bbox_generated]
        if not ready:
            return

        def _ui_show_segmented(a):
            # remove the green dot (selection) and force orange icon
            if a.selected:
                a.selected = False
            a.thumbnail = "segmented"
            # if this frame is focused and Segment mode is on, draw masks now
            if a is cls.getBuffer() and gui.getFuncButton().segButtonPressed():
                a.drawSegmentation = True

        def segment_one(abs_obj):
            old_ch = abs_obj.selected_channel
            for channel in abs_obj.available_channels:
                # if we ALREADY have masks for this channel, just display them
                existing = abs_obj._get_seg_list_for_channel(channel)
                if existing:
                    abs_obj.selected_channel = channel
                    abs_obj._abstract__seg = existing
                    abs_obj.segment_generated = True
                    # schedule UI updates on the Tk main thread
                    gui.getRoot().after(0, lambda a=abs_obj: _ui_show_segmented(a))
                    continue

                # otherwise, run segmentation once
                abs_obj.selected_channel = channel
                _ = abs_obj.segment  # this will block until segment_generated = True
                gui.getRoot().after(0, lambda a=abs_obj: _ui_show_segmented(a))

        threads = []
        for abs_obj in selected:
            t = threading.Thread(target=segment_one, args=(abs_obj,), daemon=True)
            t.start()
            threads.append(t)

        gui.popBox("i", "Segmentation", f"Started segmentation for {len(selected)} images.")

    @staticmethod
    def normalize_to_uint8(img):
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    @staticmethod
    def clahe(img, clip_limit=4.0, tile_size=(8, 8)):
        c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return c.apply(img)

    @staticmethod
    def preprocess_nucleus_stack(stack: np.ndarray) -> np.ndarray:
        stack = stack[np.any(stack > 0, axis=(1, 2))]
        proj = np.max(stack, axis=0)
        return cv2.normalize(proj, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def preprocess_cytoplasm_stack(stack: np.ndarray, top_n: int = 8) -> np.ndarray:
        stack = stack[np.any(stack > 0, axis=(1, 2))]
        scores = [cv2.Laplacian(s, cv2.CV_64F).var() for s in stack]
        best_z = np.argsort(scores)[-top_n:]
        best_z.sort()
        proj = np.max(stack[best_z], axis=0)
        proj_u8 = cv2.normalize(proj, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(proj_u8)

    @staticmethod
    def gradient(img: np.ndarray, ksize: int = 5) -> np.ndarray:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kern)

    @staticmethod
    def mask_to_bbox(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return (xs.min(), ys.min(), xs.max(), ys.max())

    @staticmethod
    def watershed_segment_with_centers(cyt_img: np.ndarray,
                                      centers: list[tuple[float,float]]
                                     ) -> list[np.ndarray]:
        thresh = filters.threshold_otsu(cyt_img)
        binary = morphology.remove_small_holes(cyt_img > thresh, area_threshold=1000)
        binary = morphology.remove_small_objects(binary, min_size=1000)
        dist   = ndi.distance_transform_edt(binary)

        markers = np.zeros(cyt_img.shape, np.int32)
        for i,(cx,cy) in enumerate(centers, start=1):
            xi, yi = int(round(cx)), int(round(cy))
            if 0<=yi<cyt_img.shape[0] and 0<=xi<cyt_img.shape[1]:
                markers[yi,xi] = i

        elev   = -dist + 5*filters.sobel(cyt_img)
        labels = segmentation.watershed(elev, markers=markers, mask=binary)

        masks = []
        for lab in range(1, labels.max()+1):
            m = (labels==lab)
            if m.sum()>0: masks.append(abstract.postproc_mask(m))
        return masks

    def run_basic_watershed(self):
        from .canvas_view import segment
        nuc = self.__img_np_nucleus
        boxes = self.gui.getBackEnd().AppIntDINOwrapper(nuc)
        centers = [((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in boxes]

        # process 647 first (cyto1), then 488 (cyto2)
        for raw, chan in [(self.__img_np_cyto1, "647"),
                        (self.__img_np_cyto2, "488")]:
            if raw is None:
                continue

            removed = abstract.remove_outliers(raw)
            img = abstract.normalize_to_uint8(removed)

            if chan == "647":
                clahe = abstract.clahe(img, clip_limit=2.0, tile_size=(8,8))
                grad  = abstract.gradient(clahe, ksize=5)
                proc = clahe
                rgb  = np.stack([clahe, clahe, grad], axis=-1)

            else:  # chan == "488"
                rem = abstract.remove_outliers(raw)
                ch = abstract.normalize_to_uint8(rem)
                cyt_clahe = abstract.clahe(ch, clip_limit=4.0, tile_size=(8,8))

                sigma_est = estimate_sigma(cyt_clahe, channel_axis=None, average_sigmas=True)
                sigma_norm = sigma_est + 3.0
                sigma_weak = sigma_est - 10.0

                cyt_bilat = cv2.bilateralFilter(cyt_clahe, d=9, sigmaColor=sigma_norm, sigmaSpace=15, borderType=cv2.BORDER_REFLECT_101)
                cyt_edge_preserved = cv2.edgePreservingFilter(cyt_clahe, flags=1, sigma_s=sigma_norm, sigma_r=0.4)
                cyt_bilat_edge = cv2.edgePreservingFilter(cyt_bilat, flags=1, sigma_s=sigma_weak, sigma_r=0.4)

                laplacian = cv2.Laplacian(ch, cv2.CV_64F)
                laplacian = cv2.convertScaleAbs(laplacian)
                cyt_blended = cv2.addWeighted(cyt_bilat, 0.8, laplacian, 0.2, 0)

                proc = cyt_bilat_edge
                rgb  = np.stack([cyt_blended, cyt_bilat, cyt_edge_preserved], axis=-1)

            ws_masks = abstract.watershed_segment_with_centers(proc, centers)
            bboxes = [abstract.mask_to_bbox(m) for m in ws_masks]
            bboxes = [b for b in bboxes if b is not None]

            channel_masks = []
            for bb in bboxes:
                try:
                    box_input = [[[float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]]]
                    sets = self.gui.getBackEnd().finetune.AppIntPREDICTCytoplasmWrapper(rgb, box_input)
                    if sets is not None and len(sets) > 0 and sets[0] is not None and len(sets[0]) > 0:
                        best = max(sets[0], key=lambda m: m.sum())
                        channel_masks.append(abstract.postproc_mask(best))
                except Exception as e:
                    logger.error(f"SAM refine failed on {chan} box {bb}: {str(e)}")
            
            if chan == "647":
                self.__seg_647 = [segment(self.gui, m) for m in channel_masks]
            else:
                self.__seg_488 = [segment(self.gui, m) for m in channel_masks]

        # Set current seg list to selected channel
        self.__seg = self.__seg_647 if self.selected_channel == "647" else self.__seg_488
        self.__segment_generated = True
        logger.info(f"Generated {len(self.__seg)} final segments ({self.selected_channel})")

    @staticmethod
    def postproc_mask(m):
        m = morphology.remove_small_objects(m.astype(bool), min_size=200)
        m = morphology.binary_opening(m, footprint=morphology.disk(2))
        return m.astype(np.uint8)

    def getImgNumpyRGB(self) -> np.ndarray:
        if self.selected_channel == "647" and self.__img_np_647 is not None:
            base = self.__img_np_647
        elif self.selected_channel == "488" and self.__img_np_488 is not None:
            base = self.__img_np_488
        else:
            base = self.__img_np_nucleus
        return abstract.grayscale_to_rgb(base)

    def getLabel(self) -> tkinter.Label:
        return self.__label
    def getNucleusPath(self) -> pathlib.Path:
        return self.__nucleus_path
    def getCytoplasmPaths(self) -> tuple[pathlib.Path, ...]:
        return tuple(self.__cyto_paths)

    @property
    def highlighted(self) -> str:
        return self.__highlighted
    @highlighted.setter
    def highlighted(self, color: str):
        self.__highlighted = color
        self.getLabel().config(borderwidth=2, background=color)
    @highlighted.deleter
    def highlighted(self):
        self.__highlighted = None
        self.getLabel().config(borderwidth=0, background="black")

    @property
    def bbox_generated(self) -> bool:
        return self.__bbox_generated
    @bbox_generated.setter
    def bbox_generated(self, value: bool):
        self.__bbox_generated = value
        if not self.gui.getFuncButton().selectButtonPressed():
            self.thumbnail = "bbox" if value else "default"

    @property
    def segment_generated(self) -> bool:
        return self.__segment_generated
    @segment_generated.setter
    def segment_generated(self, value: bool):
        self.__segment_generated = value
        if not self.gui.getFuncButton().selectButtonPressed():
            self.thumbnail = "segmented" if value else "bbox"

    @property
    def drawBbox(self) -> bool:
        return self.__drawBbox
    @drawBbox.setter
    def drawBbox(self, value: bool):
        if not self.bbox_generated:
            self.gui.popBox("w", "No BBOX", "No BBOX is available for this image")
            self.__drawBbox = False
            return
        else:
            try:
                for b in self.bbox:
                    try:
                        b.draw = value
                        if not value:
                            from .canvas_view import box
                            box.clearBufferAndDeselect()
                    except Exception as e:
                        print(f"Error setting drawBbox: {e}")
                        continue
            except Exception as e:
                print(f"Error in drawBbox setter: {e}")
        
        try:
            self.gui.getStove().canvas.draw()
        except Exception as e:
            print(f"Error drawing canvas in drawBbox: {e}")
        
        self.__drawBbox = value

    @property
    def drawSegmentation(self) -> bool:
        return self.__drawSeg
    @drawSegmentation.setter
    def drawSegmentation(self, value: bool):
        segs = self.__seg if self.segment_generated else []
        for s in segs:
            s.draw = bool(value)
        self.__drawSeg = bool(value)

    def findBoxFromPoint(self, x: float, y: float):
        for b in self.bbox:
            if b.contains(x, y):
                return b
        return None

    def findSegFromPoint(self, x: float, y: float):
        if not self.segment_generated:
            return None
        for s in self.__seg:
            if s.contains(x, y):
                return s
        return None
    
class tifSequence():
    def __init__(self, gui):
        self.gui = gui
        container = gui.getLowerFrame().getFrameB()
        self.base = tkinter.Canvas(container, height=74)

        self.scrollbar = tkinter.Scrollbar(container, orient=tkinter.HORIZONTAL, command=self.base.xview)
        self.base.configure(xscrollcommand=self.scrollbar.set)

        self.gallery_frame = tkinter.Frame(self.base)
        self.base.create_window((0, 0), window=self.gallery_frame, anchor="nw")

        self.base.bind("<Configure>", lambda e: self.update_scrollregion())
        self.base.bind_all("<MouseWheel>", self.on_mouse_wheel)
        self.base.bind_all("<Button-4>", self.on_mouse_wheel)
        self.base.bind_all("<Button-5>", self.on_mouse_wheel)

    def update_scrollregion(self):
        self.base.update_idletasks()
        self.base.config(scrollregion=self.base.bbox("all"))

    def on_mouse_wheel(self, event):
        if event.num == 4:
            self.base.xview_scroll(-1, "units")
        elif event.num == 5:
            self.base.xview_scroll(1, "units")
        elif event.delta:
            if event.delta > 0:
                self.base.xview_scroll(-1, "units")
            else:
                self.base.xview_scroll(1, "units")
        
    def pack(self):
        self.base.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        self.scrollbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    
    def addToGallery(self, tif_files: list):
        logger.debug(f"addToGallery → starting with {len(tif_files)} files")
        grouped: dict[str, dict[str, pathlib.Path]] = {}

        for path in tif_files:
            path = pathlib.Path(path)
            stem = path.stem
            pos = re.search(r"s(\d{1,4})", stem, re.IGNORECASE)
            chan = re.search(r"w[-_]?(?:.*?)?(DAPI|488|647)", stem, re.IGNORECASE)
            
            if not (pos and chan):
                logger.warning(f"addToGallery → skipping {stem!r}, couldn't parse s### or w###")
                continue

            sample_id = pos.group(1)
            wavelength = chan.group(1).upper()
            grouped.setdefault(sample_id, {})[wavelength] = path

        logger.debug(f"addToGallery → grouped into samples: {list(grouped.keys())}")

        for sample_id, channels in grouped.items():
            nucleus_path = channels.get("DAPI")
            if nucleus_path is None:
                logger.warning(f"addToGallery → sample {sample_id} has no DAPI, skipping")
                continue

            cyto_paths: list[pathlib.Path] = []
            if "647" in channels:
                cyto_paths.append(channels["647"])
            if "488" in channels:
                cyto_paths.append(channels["488"])

            logger.info(f"addToGallery → instantiating abstract for sample {sample_id}")
            abs_obj = abstract(
                sample_id,
                nucleus_path,
                cyto_paths,
                self.gallery_frame,
                self.gui
            )
            self.gui.getSeasoning().update_channel_menu(abs_obj.available_channels)

        abstract.sendFirst()
        self.update_scrollregion()
