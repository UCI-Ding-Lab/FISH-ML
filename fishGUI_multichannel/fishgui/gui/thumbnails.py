# fishgui/gui/thumbnails.py
import tkinter as tk
import pathlib
from PIL import Image, ImageTk, ImageDraw
from ..model.abstract import abstract

class tifSequence:
    def __init__(self, gui):
        self.gui = gui
        container = gui.getLowerFrame().getFrameB()
        self.base = tk.Canvas(container, height=74)
        self.scrollbar = tk.Scrollbar(container, orient=tk.HORIZONTAL, command=self.base.xview)
        self.base.configure(xscrollcommand=self.scrollbar.set)
        self.gallery_frame = tk.Frame(self.base)
        self.base.create_window((0, 0), window=self.gallery_frame, anchor="nw")
        self.base.bind("<Configure>", lambda e: self.update_scrollregion())
        self.base.bind_all("<MouseWheel>", self.on_mouse_wheel)
        self.base.bind_all("<Button-4>", self.on_mouse_wheel)
        self.base.bind_all("<Button-5>", self.on_mouse_wheel)

    def update_scrollregion(self):
        self.base.update_idletasks()
        self.base.config(scrollregion=self.base.bbox("all"))

    def on_mouse_wheel(self, event):
        if getattr(event, "num", None) == 4:
            self.base.xview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.base.xview_scroll(1, "units")
        elif getattr(event, "delta", 0):
            self.base.xview_scroll(-1 if event.delta > 0 else 1, "units")

    def pack(self):
        self.base.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

    def unpack(self):
        self.base.pack_forget(); self.scrollbar.pack_forget()

    def addToGallery(self, tif_files):
        for p in tif_files:
            abstract(pathlib.Path(p), self.gallery_frame, self.gui)
        abstract.sendFirst()
        self.update_scrollregion()

    def resetPosition(self):
        self.base.xview_moveto(0); self.base.yview_moveto(0)
