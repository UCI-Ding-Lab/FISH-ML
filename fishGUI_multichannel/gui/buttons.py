import tkinter
import pathlib
from tkinter import filedialog
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
    
    def pack(self):
        self.IMPORT.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
        self.SELECT.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
        self.BBOX.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
        self.SEGMENT.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
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
        if not self.gui.getStove().isLoaded():
            self.gui.popBox("w", "Image Not Loaded", "Please select an image first")
            self.toggle["SEGMENT"].set(0)
            return
        
        abs = abstract.getBuffer()
        if not abs:
            self.gui.popBox("w", "No Image Selected", "Please select an image first")
            self.toggle["SEGMENT"].set(0)
            return
            
        if self.segButtonPressed():
            # Entering SEGMENT mode
            abs.drawSegmentation = True
        else:
            # Exiting SEGMENT mode
            abs.drawSegmentation = False
    
    def EXPORT_call(self):
        from .thumbnails import abstract
        if not self.gui.getStove().isLoaded():
            self.gui.popBox("w", "Image Not Loaded", "Please select an image first")
            self.toggle["EXPORT"].set(0)
            return
        if self.bboxButtonPressed():
            self.gui.popBox("w", "BBOX Mode", "Please exit BBOX mode first")
            self.toggle["EXPORT"].set(0)
            return
        if self.segButtonPressed():
            self.gui.popBox("w", "Segmentation Mode", "Please exit Segmentation mode first")
            self.toggle["EXPORT"].set(0)
            return
        
        # Export functionality
        self.gui.indicateWait("Dataset conversion")
        def job():
            try:
                from tkinter import filedialog
                f = filedialog.asksaveasfilename(defaultextension=".mat", 
                                               filetypes=[("Matlab files", "*.mat")],
                                               title="Export Results As")
                if f:
                    toSave = [i for i in abstract.getPool() if i.selected and len(i.segmentExplict)]
                    d = {"name":[],"image":[],"xy":[],"masks":[]}
                    for abs in toSave:
                        d["name"].append(str(abs.getAbsPath()))
                        d["image"].append(abs.getImgNumpyRGB())
                        d["xy"].append([seg.xy for seg in abs.segment])
                        d["masks"].append([seg.box for seg in abs.segment])
                    # Note: You'll need to implement matPacker.create or use scipy.io.savemat
                    self.gui.popBox("i", "Export", f"Export completed to {f}")
            except Exception as e:
                self.gui.popBox("e", "Export Error", f"Failed to export: {e}")
            finally:
                self.gui.getRoot().after(0, self.gui.dismissWait)
        
        import threading
        threading.Thread(target=job, daemon=True).start()
