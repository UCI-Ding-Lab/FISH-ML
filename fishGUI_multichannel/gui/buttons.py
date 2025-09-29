import tkinter
import pathlib
from tkinter import filedialog
import tkinter as tk
import threading
import concurrent.futures
import time
import logging

logger = logging.getLogger(__name__)

class progress():
    @staticmethod
    def generateBbox(gui, abstracts):
        """Generate bounding boxes in the background using threading and concurrency."""
        def generate_bboxes():
            def generate_bbox(abs):
                start_time = time.time()
                _ = abs.bbox
                end_time = time.time()
                print(f"Generated bbox for {abs.getAbsPath()} in {end_time - start_time:.4f} seconds")
            import os
            max_workers = min(os.cpu_count(), len(abstracts))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(generate_bbox, abstracts)
        threading.Thread(target=generate_bboxes, daemon=True).start()

class funcButton():
    def __init__(self, gui):
        self.gui = gui
        container = gui.getLowerFrame().getFrameC()
        self.toggle = {"SELECT": tkinter.IntVar(value=0),
                       "BBOX": tkinter.IntVar(value=0),
                       "SEGMENT": tkinter.IntVar(value=0),
                       "EXPORT": tkinter.IntVar(value=0)}
        self.IMPORT = tkinter.Button(container,
                                     text="Import",
                                     height=2,
                                     relief=tkinter.RAISED,
                                     command=self.IMPORT_call)
        self.SELECT = tkinter.Checkbutton(container,
                                          text="Select",
                                          height=2,
                                          variable=self.toggle["SELECT"],
                                          onvalue=1,
                                          offvalue=0,
                                          indicatoron=False,
                                          command=self.SELECT_call)
        self.BBOX = tkinter.Checkbutton(container,
                                        text="BBOX",
                                        height=2,
                                        variable=self.toggle["BBOX"],
                                        onvalue=1,
                                        offvalue=0,
                                        indicatoron=False,
                                        command=self.BBOX_call)
        self.SEGMENT = tkinter.Checkbutton(container,
                                           text="Segment",
                                           height=2,
                                           variable=self.toggle["SEGMENT"],
                                           onvalue=1,
                                           offvalue=0,
                                           indicatoron=False,
                                           command=self.SEGMENT_call)
        self.EXPORT = tkinter.Checkbutton(container,
                                    text="Export(MATLAB)",
                                    height=2,
                                    variable=self.toggle["EXPORT"],
                                    onvalue=1,
                                    offvalue=0,
                                    indicatoron=False,
                                    command=self.EXPORT_call)
        self.APPLY_MASK = tkinter.Button(container,
                                 text="Apply Channel Mask",
                                 height=2,
                                 relief=tkinter.RAISED,
                                 command=self.APPLY_MASK_call)
    
    def pack(self):
        self.IMPORT.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
        self.SELECT.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
        self.BBOX.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
        self.SEGMENT.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
        self.APPLY_MASK.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
        self.EXPORT.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
    
    def selectButtonPressed(self) -> bool:
        return self.toggle["SELECT"].get()
    def bboxButtonPressed(self) -> bool:
        return self.toggle["BBOX"].get()
    def segButtonPressed(self) -> bool:
        return self.toggle["SEGMENT"].get()
    
    def IMPORT_call(self):
        from .thumbnails import abstract
        folder_path = filedialog.askdirectory()
        logger.debug(f"IMPORT_call → user picked folder: {folder_path!r}")
        if not folder_path:
            logger.debug("IMPORT_call → no folder selected, exiting.")
            return

        folder = pathlib.Path(folder_path)
        tif_files = [file.resolve() for file in folder.glob("*.tif")]
        logger.debug(f"IMPORT_call → found {len(tif_files)} .tif files")

        try:
            self.gui.getTifSequence().addToGallery(tif_files)
        except Exception as e:
            logger.exception("IMPORT_call → addToGallery raised exception")
            self.gui.popBox("e", "Import Error", str(e))
            return

        pool = abstract.getPool()
        logger.debug(f"IMPORT_call → abstract pool size after addToGallery: {len(pool)}")
        if len(pool) == 0:
            self.gui.popBox("w", "No Image", "No image is available")
            return

        progress.generateBbox(self.gui, abstracts=pool)

    def SELECT_call(self):
        from .thumbnails import abstract
        if self.selectButtonPressed():
            abstract.selectAll()
        elif not self.selectButtonPressed():
            abstract.removeUnselected()
            self.gui.getTifSequence().resetPosition()
            for abs in abstract.getPool():
                abs.thumbnail = "bbox" if abs.bbox_generated else "default"
    
    def BBOX_call(self):
        from .thumbnails import abstract
        if not self.gui.getStove().isLoaded():
            self.gui.popBox("w", "Image Not Loaded", "Please select an image first")
            self.toggle["BBOX"].set(0)
            return
        
        abs = abstract.getBuffer()
        if not abs:
            self.gui.popBox("w", "No Image Selected", "Please select an image first")
            self.toggle["BBOX"].set(0)
            return
            
        if self.bboxButtonPressed():
            # Entering BBOX mode
            if not abs.bbox_generated:
                self.gui.popBox("w", "Bounding Boxes Not Ready", "Bounding boxes for this image have not been generated yet.")
                self.toggle["BBOX"].set(0)
                return
            abs.drawBbox = True
        else:
            # Exiting BBOX mode
            abs.drawBbox = False
    
    def SEGMENT_call(self):
        from .thumbnails import abstract
        selected = [a for a in abstract.getPool() if a.selected]
        if not selected:
            self.gui.popBox("w", "No Image Selected", "Please select an image first")
            self.toggle["SEGMENT"].set(0)
            return

        if self.segButtonPressed():
            abstract.segment_selected(self.gui)
            for abs in selected:
                abs.drawSegmentation = True
        else:
            for abs in selected:
                abs.drawSegmentation = False
    

    def EXPORT_call(self):
        from .thumbnails import abstract
        import threading, time
        logger.debug("EXPORT_call: invoked")

        # Guardrails + log state
        is_loaded = self.gui.getStove().isLoaded()
        bbox_mode = self.bboxButtonPressed()
        seg_mode  = self.segButtonPressed()
        logger.debug("EXPORT_call: isLoaded=%s, bboxMode=%s, segMode=%s", is_loaded, bbox_mode, seg_mode)

        if not is_loaded:
            logger.warning("EXPORT_call: blocked (no image loaded)")
            self.gui.popBox("w", "Image Not Loaded", "Please select an image first")
            self.toggle["EXPORT"].set(0)
            return
        if bbox_mode:
            logger.warning("EXPORT_call: blocked (BBOX mode active)")
            self.gui.popBox("w", "BBOX Mode", "Please exit BBOX mode first")
            self.toggle["EXPORT"].set(0)
            return
        if seg_mode:
            logger.warning("EXPORT_call: blocked (Segmentation mode active)")
            self.gui.popBox("w", "Segmentation Mode", "Please exit Segmentation mode first")
            self.toggle["EXPORT"].set(0)
            return

        logger.debug("EXPORT_call: indicating wait…")
        self.gui.indicateWait("Dataset conversion")

        def job():
            t0 = time.perf_counter()
            logger.debug("Export job: thread started")

            try:
                from tkinter import filedialog
                logger.debug("Export job: opening save dialog (running from worker thread)")
                f = filedialog.asksaveasfilename(
                    defaultextension=".mat",
                    filetypes=[("Matlab files", "*.mat")],
                    title="Export Results As"
                )
                logger.debug("Export job: save path selected=%r", f)

                if not f:
                    logger.info("Export job: user cancelled save dialog")
                    return

                pool = abstract.getPool()
                selected = [i for i in pool if i.selected]
                logger.debug("Export job: selected frames=%d -> %s",
                            len(selected), [getattr(i, 'sample_id', '?') for i in selected])

                toSave = [i for i in selected if len(i.segmentExplict)]
                logger.debug("Export job: frames with segments=%d", len(toSave))

                d = {"name": [], "image": [], "xy": [], "masks": []}

                for abs in toSave:
                    try:
                        name = str(abs.getAbsPath())
                    except Exception:
                        name = str(getattr(abs, "sample_id", "unknown"))
                    try:
                        img = abs.getImgNumpyRGB()
                        img_shape = getattr(img, "shape", None)
                    except Exception as e:
                        logger.exception("Export job: getImgNumpyRGB failed for %s", name)
                        img = None
                        img_shape = None

                    segs = getattr(abs, "segment", [])
                    logger.debug("Export job: packing %s, img_shape=%s, seg_count=%d",
                                name, img_shape, len(segs))

                    d["name"].append(name)
                    d["image"].append(img)
                    try:
                        d["xy"].append([seg.xy for seg in segs])
                    except Exception:
                        logger.exception("Export job: reading seg.xy failed for %s", name)
                        d["xy"].append([])

                    try:
                        # NOTE: your code stores seg.box; logging their shapes can help
                        boxes = [getattr(seg, "box", None) for seg in segs]
                        d["masks"].append(boxes)
                        logger.debug("Export job: appended %d boxes for %s", len(boxes), name)
                    except Exception:
                        logger.exception("Export job: reading seg.box failed for %s", name)
                        d["masks"].append([])

                # TODO: actually write MAT here (matPacker.create or scipy.io.savemat)
                logger.debug("Export job: data prepared (counts) names=%d, images=%d, xy=%d, masks=%d",
                            len(d["name"]), len(d["image"]), len(d["xy"]), len(d["masks"]))

                self.gui.getRoot().after(0, lambda: self.gui.popBox("i", "Export", f"Export completed to {f}"))

            except Exception:
                logger.exception("Export job: failed with exception")
                self.gui.getRoot().after(0, lambda: self.gui.popBox("e", "Export Error", "See console for details"))

            finally:
                elapsed = time.perf_counter() - t0
                logger.debug("Export job: finished in %.2fs", elapsed)
                self.gui.getRoot().after(0, self.gui.dismissWait)

        threading.Thread(target=job, daemon=True, name="ExportThread").start()

    
    # def EXPORT_call(self):
    #     from .thumbnails import abstract
    #     if not self.gui.getStove().isLoaded():
    #         self.gui.popBox("w", "Image Not Loaded", "Please select an image first")
    #         self.toggle["EXPORT"].set(0)
    #         return
    #     if self.bboxButtonPressed():
    #         self.gui.popBox("w", "BBOX Mode", "Please exit BBOX mode first")
    #         self.toggle["EXPORT"].set(0)
    #         return
    #     if self.segButtonPressed():
    #         self.gui.popBox("w", "Segmentation Mode", "Please exit Segmentation mode first")
    #         self.toggle["EXPORT"].set(0)
    #         return
        
    
        # # Export functionality
        # self.gui.indicateWait("Dataset conversion")
        # def job():
        #     logger.debug("Export job: thread started")
        #     try:
        #         from tkinter import filedialog
        #         logger.debug("Export job: opening save dialog (running from worker thread)")
        #         f = filedialog.asksaveasfilename(defaultextension=".mat", 
        #                                        filetypes=[("Matlab files", "*.mat")],
        #                                        title="Export Results As")
                
        #         logger.debug("Export job: save path selected=%r", f)
        #         if f:
        #             toSave = [i for i in abstract.getPool() if i.selected and len(i.segmentExplict)]
        #             d = {"name":[],"image":[],"xy":[],"masks":[]}
        #             for abs in toSave:
        #                 d["name"].append(str(abs.getAbsPath()))
        #                 d["image"].append(abs.getImgNumpyRGB())
        #                 d["xy"].append([seg.xy for seg in abs.segment])
        #                 d["masks"].append([seg.box for seg in abs.segment])
        #             # Note: You'll need to implement matPacker.create or use scipy.io.savemat
        #             logger.debug("Export job: data prepared (counts) names=%d, images=%d, xy=%d, masks=%d",
        #                  len(d["name"]), len(d["image"]), len(d["xy"]), len(d["masks"]))
        #             self.gui.popBox("i", "Export", f"Export completed to {f}")
        #     except Exception as e:
        #         self.gui.popBox("e", "Export Error", f"Failed to export: {e}")
        #     finally:
        #         self.gui.getRoot().after(0, self.gui.dismissWait)

    def APPLY_MASK_call(self):
        from .thumbnails import abstract

        available_channels = ["488", "647"]

        def channel_callback(selected_channel):
            # Quick current-frame sanity check (e.g., Vadym’s frame with no 647)
            buf = abstract.getBuffer()
            if buf and selected_channel not in getattr(buf, "available_channels", []):
                self.gui.popBox("w", "Channel Not Available",
                                f"Current frame {getattr(buf, 'sample_id', '?')} has no channel {selected_channel}.")
                return

            frame_names = [a.sample_id for a in abstract.getPool()]

            def frame_callback(selection):
                pool = abstract.getPool()

                # Resolve which indices we’ll act on
                if selection == "all":
                    idxs = range(len(pool))
                    frames_sel = "all"
                elif selection == "next5":
                    try:
                        start = pool.index(abstract.getBuffer())
                    except ValueError:
                        start = 0
                    idxs = range(start, min(start + 5, len(pool)))
                    frames_sel = idxs
                else:
                    # default to all
                    idxs = range(len(pool))
                    frames_sel = "all"

                # Validate channel availability across chosen frames
                missing = [pool[i].sample_id for i in idxs
                        if selected_channel not in getattr(pool[i], "available_channels", [])]
                if missing:
                    # Warn and abort (don’t flip Segment mode on)
                    preview = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
                    self.gui.popBox("w", "Channel Not Available",
                                    f"Channel {selected_channel} is missing for: {preview}")
                    self.toggle["SEGMENT"].set(0)
                    return

                # All good → apply masks
                abstract.apply_channel_mask_to_frames(
                    gui=self.gui, 
                    source_ch=selected_channel,
                    frames_sel=frames_sel,
                    targets_sel="all_channels"
                )

                # turn segment mode ON only if current buffer exists, has BBOX, and has masks
                buf = abstract.getBuffer()
                if buf and buf.bbox_generated and (buf.segment_generated or buf._get_seg_list_for_channel(buf.selected_channel)):
                    self.toggle["SEGMENT"].set(1)
                    buf.drawSegmentation = True
                else:
                    self.toggle["SEGMENT"].set(0)

            FrameSelectPopup(self.gui.getRoot(), frame_names, frame_callback)

        ChannelSelectPopup(self.gui.getRoot(), available_channels, channel_callback)


class ChannelSelectPopup(tk.Toplevel):
    def __init__(self, parent, available_channels, callback):
        super().__init__(parent)
        self.title("Select Channel Mask")
        self.callback = callback
        self.selected_channel = tk.StringVar(value=available_channels[0])

        tk.Label(self, text="Which channel mask do you want to apply for current frame?\n(Chosen channel mask will apply to all channels)").pack(pady=10)

        frame = tk.Frame(self)
        frame.pack(pady=10)
        for ch in available_channels:
            tk.Radiobutton(frame, text=f"Channel {ch}", variable=self.selected_channel, value=ch).pack(side=tk.LEFT, padx=20)

        tk.Button(self, text="Next", command=self.on_next).pack(pady=10)

    def on_next(self):
        self.callback(self.selected_channel.get())
        self.destroy()

class FrameSelectPopup(tk.Toplevel):
    def __init__(self, parent, frame_names, callback):
        super().__init__(parent)
        self.title("Apply Mask To Frames")
        self.callback = callback
        self.selection = tk.StringVar(value="all")

        tk.Label(self, text="How many more frames would you like to add channel masks?").pack(pady=10)

        canvas = tk.Canvas(self, height=120)
        scrollbar = tk.Scrollbar(self, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=scrollbar.set)
        frame = tk.Frame(canvas)
        canvas.create_window((0,0), window=frame, anchor="nw")
        canvas.pack(fill="x")
        scrollbar.pack(fill="x")

        for name in frame_names:
            tk.Label(frame, text=name, relief=tk.RIDGE, width=18).pack(side=tk.LEFT, padx=2, pady=2)

        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        tk.Radiobutton(self, text="Select Next 5", variable=self.selection, value="next5").pack(anchor="w", padx=20)
        tk.Radiobutton(self, text="Select All Frames", variable=self.selection, value="all").pack(anchor="w", padx=20)

        tk.Button(self, text="Finish & Apply", command=self.on_apply).pack(pady=10)

    def on_apply(self):
        self.callback(self.selection.get())
        self.destroy()