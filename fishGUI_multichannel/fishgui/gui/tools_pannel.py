# fishgui/gui/tools_pannel.py
import tkinter as tk
from PIL import Image, ImageTk
import pathlib
from ..model.shapes import box
from ..services.progress import Progress

class seasoning:
    def __init__(self, gui):
        self.gui = gui
        self.toolbank = tk.Frame(self.gui.getLowerFrame().getFrameA(), width=150, background="grey")
        self.button1 = tk.Button(self.toolbank, height=2, text="Save Progress", command=self.SAVEPROG_CALL)
        self.button2 = tk.Button(self.toolbank, height=2, text="Load Progress", command=self.LOADPROG_CALL)
        self.sep = tk.Frame(self.toolbank, height=1, bd=0, relief=tk.SUNKEN, bg="black")
        self.seg_editor = tk.LabelFrame(self.toolbank, text="Segmentation Editor")

        icon_path = pathlib.Path(self.gui.getBackEnd().config["gui"]["icon_folder"])
        self.tools_icon = {
            "brush": ImageTk.PhotoImage(Image.open(icon_path/"brush.png")),
            "eraser": ImageTk.PhotoImage(Image.open(icon_path/"eraser.png")),
            "add_bbox": ImageTk.PhotoImage(Image.open(icon_path/"bbox.png")),
        }

        self.tools_var = {"brush": tk.IntVar(value=0),
                          "eraser": tk.IntVar(value=0),
                          "add_bbox": tk.IntVar(value=0)}

        self.tools = {
            "brush": tk.Checkbutton(self.seg_editor, image=self.tools_icon["brush"],
                                    variable=self.tools_var["brush"], onvalue=1, offvalue=0,
                                    indicatoron=False, command=lambda: self.press_act("brush")),
            "eraser": tk.Checkbutton(self.seg_editor, image=self.tools_icon["eraser"],
                                     variable=self.tools_var["eraser"], onvalue=1, offvalue=0,
                                     indicatoron=False, command=lambda: self.press_act("eraser")),
            "add_bbox": tk.Button(self.seg_editor, image=self.tools_icon["add_bbox"], command=self.ADDBBOX_CALL),
        }

        self.marker_size_var = tk.IntVar(value=15)
        self.marker_size_scale = tk.Scale(self.toolbank, from_=10, to=30, orient=tk.HORIZONTAL,
                                          label="Marker Size", variable=self.marker_size_var)

        self.sep2 = tk.Frame(self.toolbank, height=1, bd=0, relief=tk.SUNKEN, bg="black")
        self.contrast_var = tk.IntVar(value=250)
        self.contrast_bar = tk.Scale(self.toolbank, from_=0, to=500, orient=tk.HORIZONTAL,
                                     label="Contrast", variable=self.contrast_var)
        self.contrast_bar.bind("<ButtonRelease-1>", self.on_contrast_bar_change)
        self.contrast_reset = tk.Button(self.toolbank, text="Reset",
                                        command=lambda: self.on_contrast_bar_change(None))

        self.sep3 = tk.Frame(self.toolbank, height=1, bd=0, relief=tk.SUNKEN, bg="black")
        self.brightness_var = tk.IntVar(value=250)
        self.brightness_bar = tk.Scale(self.toolbank, from_=0, to=500, orient=tk.HORIZONTAL,
                                       label="Brightness", variable=self.brightness_var)
        self.brightness_bar.bind("<ButtonRelease-1>", self.on_brightness_bar_change)
        self.brightness_reset = tk.Button(self.toolbank, text="Reset",
                                          command=lambda: self.on_brightness_bar_change(None))

    def pack(self):
        self.toolbank.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.button1.pack(side=tk.TOP, fill=tk.X)
        self.button2.pack(side=tk.TOP, fill=tk.X)
        self.sep.pack(fill=tk.X)
        self.seg_editor.pack(side=tk.TOP, fill=tk.X)
        self.tools["brush"].grid(row=0, column=0)
        self.tools["eraser"].grid(row=0, column=1)
        self.tools["add_bbox"].grid(row=1, column=0, columnspan=2)
        self.marker_size_scale.pack(side=tk.TOP, fill=tk.X)
        self.sep2.pack(fill=tk.X)
        self.contrast_bar.pack(side=tk.TOP, fill=tk.X)
        self.contrast_reset.pack(side=tk.TOP, fill=tk.X)
        self.sep3.pack(fill=tk.X)
        self.brightness_bar.pack(side=tk.TOP, fill=tk.X)
        self.brightness_reset.pack(side=tk.TOP, fill=tk.X)

    def on_contrast_bar_change(self, event):
        if event is None: self.contrast_var.set(250)
        factor = (0.01 * self.contrast_var.get()) - 1.5
        self.gui.getStove().adjust_contrast(factor)

    def on_brightness_bar_change(self, event):
        if event is None: self.brightness_var.set(250)
        factor = (250 - self.brightness_var.get()) * 0.005
        self.gui.getStove().adjust_brightness(factor)

    def press_act(self, which):
        if not self.gui.getFuncButton().segButtonPressed():
            self.gui.popBox("w", "Segmentation Mode", "Please enter Segmentation mode first")
            for v in self.tools_var.values(): v.set(0)
            return
        for k, v in self.tools_var.items():
            if k != which: v.set(0)

    def get_marker_size(self): return self.marker_size_var.get()
    def burshButtonPressed(self): return self.tools_var["brush"].get()     # keep spelling for now
    def eraserButtonPressed(self): return self.tools_var["eraser"].get()

    def ADDBBOX_CALL(self):
        if not self.gui.getFuncButton().bboxButtonPressed():
            self.gui.popBox("w", "BBOX Mode", "Please enter BBOX mode first"); return
        loaded = self.gui.getStove().getLoaded()
        if not loaded:
            self.gui.popBox("w", "No Image", "No image is loaded"); return
        h, w = loaded.getImgNumpyRGB().shape[0], loaded.getImgNumpyRGB().shape[1]
        new = box([w//2-150, h//2-150, w//2+150, h//2+150], self.gui)
        loaded.bbox.append(new); new.selected = True; new.draw = True
        self.gui.getStove().canvas.draw()

    def SAVEPROG_CALL(self):
        self.gui.indicateWait("Pkl save")
        import threading
        def job():
            from ..model.abstract import abstract
            Progress.save(abstract)
            self.gui.getRoot().after(0, self.gui.dismissWait)
        threading.Thread(target=job, daemon=True).start()


    def LOADPROG_CALL(self):
        self.gui.indicateWait("Pkl load")
        import threading
        def job():
            from ..model.abstract import abstract
            Progress.load(self.gui, abstract, abstract)  # ctor is abstract
            self.gui.getRoot().after(0, self.gui.dismissWait)
        threading.Thread(target=job, daemon=True).start()
