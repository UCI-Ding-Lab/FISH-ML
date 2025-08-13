# fishgui/gui/buttons.py
import tkinter as tk
import pathlib
from tkinter import filedialog
from ..model.abstract import abstract
from ..services.progress import Progress

class funcButton:
    def __init__(self, gui):
        self.gui = gui
        container = gui.getLowerFrame().getFrameC()
        self.toggle = {"SELECT": tk.IntVar(value=0),
                       "BBOX": tk.IntVar(value=0),
                       "SEGMENT": tk.IntVar(value=0),
                       "EXPORT": tk.IntVar(value=0)}
        self.IMPORT = tk.Button(container, text="Import", height=2, relief=tk.RAISED, command=self.IMPORT_call)
        self.SELECT = tk.Checkbutton(container, text="Select", height=2,
                                     variable=self.toggle["SELECT"], onvalue=1, offvalue=0,
                                     indicatoron=False, command=self.SELECT_call)
        self.BBOX = tk.Checkbutton(container, text="BBOX", height=2,
                                   variable=self.toggle["BBOX"], onvalue=1, offvalue=0,
                                   indicatoron=False, command=self.BBOX_call)
        self.SEGMENT = tk.Checkbutton(container, text="Segment", height=2,
                                      variable=self.toggle["SEGMENT"], onvalue=1, offvalue=0,
                                      indicatoron=False, command=self.SEGMENT_call)
        self.EXPORT = tk.Checkbutton(container, text="Export(MATLAB)", height=2,
                                     variable=self.toggle["EXPORT"], onvalue=1, offvalue=0,
                                     indicatoron=False, command=self.EXPORT_call)

    def pack(self):
        self.IMPORT.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.SELECT.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.BBOX.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.SEGMENT.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.EXPORT.pack(side=tk.LEFT, expand=True, fill=tk.X)

    def unpack(self):
        for w in (self.IMPORT, self.SELECT, self.BBOX, self.SEGMENT, self.EXPORT):
            w.pack_forget()

    def getButtonWidget(self, name):
        return {"IMPORT": self.IMPORT, "SELECT": self.SELECT, "BBOX": self.BBOX,
                "SEGMENT": self.SEGMENT, "EXPORT": self.EXPORT}[name]

    def selectButtonPressed(self): return self.toggle["SELECT"].get()
    def bboxButtonPressed(self):   return self.toggle["BBOX"].get()
    def segButtonPressed(self):    return self.toggle["SEGMENT"].get()

    def IMPORT_call(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return
        tif_files = [str(p.resolve()) for p in pathlib.Path(folder_path).glob("*.tif")]
        self.gui.getTifSequence().addToGallery(tif_files)
        pool = abstract.getPool()
        if not pool:
            self.gui.popBox("w", "No Image", "No image is available")
            return
        Progress.generateBbox(self.gui, abstracts=pool)

    def SELECT_call(self):
        if self.selectButtonPressed():
            abstract.selectAll()
        else:
            abstract.removeUnselected()
            self.gui.getTifSequence().resetPosition()
            for a in abstract.getPool():
                a.thumbnail = "bbox" if a.bbox_generated else "default"

    def BBOX_call(self):
        if not self.gui.getStove().isLoaded():
            self.gui.popBox("w", "Image Not Loaded", "Please select an image first")
            self.toggle["BBOX"].set(0); return
        if self.bboxButtonPressed():
            a = abstract.getBuffer()
            if a and not a.bbox_generated:
                self.gui.popBox("w", "Bounding Boxes Not Ready",
                                "Bounding boxes for this image have not been generated yet.")
                self.toggle["BBOX"].set(0); return
            abstract.sendFocused()
        else:
            abstract.saveBboxChanges()

    def SEGMENT_call(self):
        if not self.gui.getStove().isLoaded():
            self.gui.popBox("w", "Image Not Loaded", "Please select an image first")
            self.toggle["SEGMENT"].set(0); return
        if self.bboxButtonPressed():
            self.gui.popBox("w", "BBOX Mode", "Please exit BBOX mode first")
            self.toggle["SEGMENT"].set(0); return
        if self.segButtonPressed():
            abstract.sendFocused()
        else:
            abstract.saveSegChanges()

    def EXPORT_call(self):
        if not self.gui.getStove().isLoaded():
            self.gui.popBox("w", "Image Not Loaded", "Please select an image first")
            self.toggle["EXPORT"].set(0); return
        if self.bboxButtonPressed():
            self.gui.popBox("w", "BBOX Mode", "Please exit BBOX mode first")
            self.toggle["EXPORT"].set(0); return
        if self.segButtonPressed():
            self.gui.popBox("w", "Segmentation Mode", "Please exit Segmentation mode first")
            self.toggle["EXPORT"].set(0); return

        self.gui.indicateWait("Dataset conversion")
        import threading
        def job():
            Progress.export(self.gui)
            self.gui.getRoot().after(0, self.gui.dismissWait)
        threading.Thread(target=job, daemon=True).start()
