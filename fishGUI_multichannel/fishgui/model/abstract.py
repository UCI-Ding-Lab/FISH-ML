# fishgui/model/abstract.py
import time, threading, pathlib, tkinter
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import cv2, tifffile
from .shapes import box
from .segment import segment
from skimage import filters, morphology, segmentation
from scipy import ndimage as ndi

class abstract(object):
    __pool = []
    __buffer = None

    def __init__(self, path, gallery_frame, gui):
        self._path = pathlib.Path(path)
        self.gui = gui

        stack = tifffile.imread(self._path)

        # channel routing
        if stack.ndim == 2 or stack.shape[0] == 2:
            cyto2_index = -1
            nuc_index = 1
        elif stack.ndim == 3 or stack.shape[0] == 3:
            cyto2_index = 1
            nuc_index = 2
        else:
            raise ValueError("Unexpected TIF shape: %r" % (stack.shape,))

        nucleus = stack[nuc_index]
        cyto1 = stack[0]
        cyto2 = stack[cyto2_index] if cyto2_index != -1 else None

        cyto1 = self.clahe(self.normalize_to_uint8(cyto1))
        cyto2 = self.clahe(self.normalize_to_uint8(cyto2)) if cyto2_index != -1 else None
        nucleus = self.clahe(nucleus)

        self._nucleus = nucleus
        self._cyto1 = cyto1
        self._cyto2 = cyto2
        self._cyt_clahe = cyto1

        self._rgb1 = self.grayscale_to_rgb(self._cyto1)
        thumb = Image.fromarray(self._rgb1).resize((64, 64))
        self._thumb_pil = thumb
        self._thumb_tk = ImageTk.PhotoImage(thumb)

        self._label = tkinter.Label(gallery_frame, image=self._thumb_tk, width=64, height=64,
                                    relief=tkinter.FLAT, borderwidth=0)
        self._label.pack(side=tkinter.LEFT, padx=2, pady=2)
        self._label.bind("<Button-1>", self.on_click)

        # thumbs for states
        self._thumb_bbox_pil = None
        self._thumb_sel_pil = None
        self._thumb_cross_pil = None
        self._thumb_seg_pil = None
        self._thumb_bbox_tk = None
        self._thumb_sel_tk = None
        self._thumb_cross_tk = None
        self._thumb_seg_tk = None

        self._thumbnail = None
        self._bbox = []
        self._seg = []
        self._selected = True
        self._drawBbox = False
        self._drawSeg = False
        self._bbox_generated = False
        self._segment_generated = False
        self._highlighted = None

        abstract.addToPool(self)

    # ---------- segment generation (watershed basic) ----------
    def run_basic_watershed(self):
        self._seg = []
        self._segment_generated = False

        # 1) nuclei via backend detector
        nuc_boxes = self.gui.getBackEnd().AppIntDINOwrapper(self._nucleus)
        centers = [((x1+x2)/2.0, (y1+y2)/2.0) for x1,y1,x2,y2 in nuc_boxes]

        # 2) watershed masks on cytoplasm channel
        masks = self.watershed_segment(self._cyt_clahe, centers)
        for m in masks:
            if m.sum() > 0:
                self._seg.append(segment(self.gui, m))

    @property
    def segment(self):
        def job():
            self.run_basic_watershed()
            self.segment_generated = True
            self.gui.getRoot().after(0, self.gui.dismissWait)

        if not self._segment_generated:
            self.gui.indicateWait("Segmentation")
            self.gui.getRoot().update_idletasks()
            t = threading.Thread(target=job, daemon=True)
            t.start()
            while not self._segment_generated:
                time.sleep(0.1)
        return self._seg

    @segment.setter
    def segment(self, value):
        self._seg = value
        self.segment_generated = True if value else False
    @property
    def segmentExplict(self):
        return self._seg

    # ---------- bbox generation via backend ----------
    @property
    def bbox(self):
        if not self._bbox_generated:
            nuc_boxes = self.gui.getBackEnd().AppIntDINOwrapper(self._nucleus)
            centers = [((x0+x1)/2.0, (y0+y1)/2.0) for x0,y0,x1,y1 in nuc_boxes]
            cyto_boxes = self.gui.getBackEnd().AppIntDINOwrapperB(self._cyto1, centers)
            self._bbox = [box(b, self.gui) for b in cyto_boxes]
            self.bbox_generated = True
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        self._bbox = value
        self.bbox_generated = True if value else False

    # ---------- properties used by GUI ----------
    @property
    def bbox_generated(self):
        return self._bbox_generated
    @bbox_generated.setter
    def bbox_generated(self, v):
        self._bbox_generated = v
        if not self.gui.getFuncButton().selectButtonPressed():
            self.thumbnail = "bbox" if v else "default"

    @property
    def segment_generated(self):
        return self._segment_generated
    @segment_generated.setter
    def segment_generated(self, v):
        self._segment_generated = v
        if not self.gui.getFuncButton().selectButtonPressed():
            self.thumbnail = "segmented" if v else "bbox"

    @property
    def drawBbox(self):
        return self._drawBbox
    @drawBbox.setter
    def drawBbox(self, value):
        if not self.bbox_generated:
            self.gui.popBox("w", "No BBOX", "No BBOX is available for this image")
            self._drawBbox = False
            return
        for b in self.bbox:
            b.draw = value
            if not value:
                from .shapes import box as Box  # avoid cycle
                Box.clearBufferAndDeselect()
        self.gui.getStove().canvas.draw()
        self._drawBbox = value

    @property
    def drawSegmentation(self):
        return self._drawSeg
    @drawSegmentation.setter
    def drawSegmentation(self, value):
        for s in self.segment:
            s.draw = True if value else False
        self._drawSeg = value

    @property
    def thumbnail(self):
        return self._thumbnail
    @thumbnail.setter
    def thumbnail(self, v):
        self._thumbnail = v
        if v == "default":
            self.getLabel().config(image=self._thumb_tk)
        elif v == "bbox":
            if not self._thumb_bbox_tk:
                self._thumb_bbox_pil = self._thumb_pil.copy()
                ImageDraw.Draw(self._thumb_bbox_pil).ellipse((49,5,59,15), fill=(0,0,255))
                self._thumb_bbox_tk = ImageTk.PhotoImage(self._thumb_bbox_pil)
            self.getLabel().config(image=self._thumb_bbox_tk)
        elif v == "selected":
            if not self._thumb_sel_tk:
                self._thumb_sel_pil = self._thumb_pil.copy()
                ImageDraw.Draw(self._thumb_sel_pil).ellipse((5,5,15,15), fill=(0,255,0))
                self._thumb_sel_tk = ImageTk.PhotoImage(self._thumb_sel_pil)
            self.getLabel().config(image=self._thumb_sel_tk)
        elif v == "crossout":
            if not self._thumb_cross_tk:
                self._thumb_cross_pil = self._thumb_pil.copy()
                d = ImageDraw.Draw(self._thumb_cross_pil)
                d.line((5,5,15,15), fill=(255,0,0), width=2)
                d.line((5,15,15,5), fill=(255,0,0), width=2)
                self._thumb_cross_tk = ImageTk.PhotoImage(self._thumb_cross_pil)
            self.getLabel().config(image=self._thumb_cross_tk)
        elif v == "segmented":
            if not self._thumb_seg_tk:
                # needs bbox thumb base first
                if self._thumb_bbox_pil is None:
                    self._thumb_bbox_pil = self._thumb_pil.copy()
                self._thumb_seg_pil = self._thumb_bbox_pil.copy()
                ImageDraw.Draw(self._thumb_seg_pil).ellipse((49,20,59,30), fill=(255,165,0))
                self._thumb_seg_tk = ImageTk.PhotoImage(self._thumb_seg_pil)
            self.getLabel().config(image=self._thumb_seg_tk)

    @property
    def selected(self):
        return self._selected
    @selected.setter
    def selected(self, v):
        self._selected = bool(v)
        self.thumbnail = "selected" if v else "crossout"

    @property
    def boundingBoxRevised(self):
        if self.noBbox(): return []
        return [b.final for b in self.bbox]

    @property
    def segmentationRevised(self):
        if self.noSegment():
            return []
        # segment stores mask as self._data (transposed internally); return original orientation
        return [s._data.T for s in self.segment]

    # ---------- gallery interactions ----------
    def on_click(self, _):
        self.gui.getStove().bufferSetCurrent(3)
        self.gui.getStove().dump()
        b = abstract.getBuffer()
        if b: b.highlighted = None
        self.highlighted = "red"
        if self.gui.getFuncButton().selectButtonPressed():
            self.selected = not self.selected
        elif self.gui.getFuncButton().bboxButtonPressed():
            from .shapes import box as Box
            buf = Box.getBuffer()
            if buf: buf.selected = False
            if b: b.drawBbox = False
            self.drawBbox = True
        elif self.gui.getFuncButton().segButtonPressed():
            if b: b.drawSegmentation = False
            self.drawSegmentation = True
        abstract.setBuffer(self)
        self.gui.getStove().cook(self)

    @property
    def highlighted(self):
        return self._highlighted
    @highlighted.setter
    def highlighted(self, color):
        self._highlighted = color
        self.getLabel().config(borderwidth=2, background=color)

    # ---------- class pool ----------
    @classmethod
    def getPool(cls):
        return cls.__pool
    @classmethod
    def addToPool(cls, a):
        cls.__pool.append(a)
    @classmethod
    def getBuffer(cls):
        return cls.__buffer
    @classmethod
    def setBuffer(cls, a):
        cls.__buffer = a
    @classmethod
    def sendFirst(cls):
        for a in cls.getPool():
            if a.selected:
                a.on_click(None); return
    @classmethod
    def sendFocused(cls):
        cur = cls.getBuffer()
        if cur: cur.on_click(None)
        else: cls.sendFirst()
    @classmethod
    def selectAll(cls):
        for a in cls.getPool(): a.selected = True
    @classmethod
    def removeUnselected(cls):
        for a in cls.getPool():
            a.thumbnail = "default"
            if not a.selected:
                a.getLabel().pack_forget()
        cls.sendFocused()
    @classmethod
    def saveBboxChanges(cls):
        cls.getBuffer().drawBbox = False
        cls.sendFocused()
    @classmethod
    def saveSegChanges(cls):
        cls.getBuffer().drawSegmentation = False
        cls.sendFocused()

    # used by Progress.load()
    def restore_bbox_and_segments(self, bbox_list, seg_list):
        self._bbox = [box(b, self.gui) for b in (bbox_list or [])]
        self._seg  = [segment(self.gui, s) for s in (seg_list or [])]
        self._bbox_generated = bool(self._bbox)
        self._segment_generated = bool(self._seg)

    # ---------- simple getters ----------
    def getImgNumpyGreyscale(self):
        return self._nucleus
    def getImgNumpyRGB(self):
        return self._rgb1
    def getLabel(self):
        return self._label
    def getAbsPath(self):
        return self._path
    def noBbox(self):
        return not len(self._bbox)
    def noSegment(self):
        return not len(self._seg)

    # ---------- imaging helpers & watershed ----------
    @staticmethod
    def clahe(img, clip_limit=4.0, tile_size=(8, 8)):
        if img is None: return None
        c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return c.apply(img)

    @staticmethod
    def normalize_to_uint8(img):
        if img is None: return None
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def grayscale_to_rgb(grayscale_img):
        img = cv2.normalize(grayscale_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _postproc_mask(m):
        m = morphology.remove_small_objects(m.astype(bool), min_size=200)
        m = morphology.binary_opening(m, footprint=morphology.disk(2))
        return m.astype(np.uint8)

    @staticmethod
    def watershed_segment(cyt_img, centers, thresh_method="otsu"):
        if thresh_method == "otsu":
            thresh = filters.threshold_otsu(cyt_img)
        else:
            thresh = np.percentile(cyt_img, 30)

        binary = cyt_img > thresh
        binary = morphology.remove_small_holes(binary, area_threshold=1000)
        binary = morphology.remove_small_objects(binary, min_size=1000)
        dist = ndi.distance_transform_edt(binary)

        markers = np.zeros(cyt_img.shape, dtype=np.int32)
        for idx, (cx, cy) in enumerate(centers, 1):
            xi, yi = int(round(cx)), int(round(cy))
            if 0 <= yi < cyt_img.shape[0] and 0 <= xi < cyt_img.shape[1]:
                markers[yi, xi] = idx

        labels = segmentation.watershed(-dist, markers, mask=binary)
        masks = []
        for i in range(1, len(centers)+1):
            cell = (labels == i)
            masks.append(abstract._postproc_mask(cell))
        return masks


def findBoxFromPoint(self, x, y):
    """Return the top-most bbox under (x,y) or None."""
    # self.bbox triggers generation if needed
    for b in reversed(self.bbox):
        if b.contains(x, y):
            return b
    return None

def findSegFromPoint(self, x, y):
    """Return the top-most segment under (x,y) or None."""
    # self.segment triggers generation if needed
    for s in reversed(self.segment):
        if s.contains(x, y):
            return s
    return None