# fishgui/gui/tools_pannel.py
import tkinter
import pathlib
import threading
from PIL import Image, ImageTk
from ..model.shapes import box
from ..services.progress import Progress

class seasoning():
    def __init__(self, gui):
        self.gui = gui
        self.toolbank = tkinter.Frame(self.gui.getLowerFrame().getFrameA(), width=150, background="grey")
        self.button1 = tkinter.Button(self.toolbank, height=2, text="Save Progress", command=self.SAVEPROG_CALL)
        self.button2 = tkinter.Button(self.toolbank, height=2, text="Load Progress", command=self.LOADPROG_CALL)
        self.sep = tkinter.Frame(self.toolbank, height=1, bd=0, relief=tkinter.SUNKEN, bg="black")
        self.seg_editor = tkinter.LabelFrame(self.toolbank, text="Segmentation Editor")
        
        icon_path = pathlib.Path(self.gui.getBackEnd().config["gui"]["icon_folder"])
        
        self.tools_icon = {"brush": ImageTk.PhotoImage(Image.open(icon_path/"brush.png")),
                           "eraser": ImageTk.PhotoImage(Image.open(icon_path/"eraser.png")),
                           "add_bbox": ImageTk.PhotoImage(Image.open(icon_path/"bbox.png"))}
        
        self.tools_var = {"brush": tkinter.IntVar(value=0),
                          "eraser": tkinter.IntVar(value=0),
                          "add_bbox": tkinter.IntVar(value=0)}
        
        self.tools = {"brush": tkinter.Checkbutton(self.seg_editor
                                                   ,image=self.tools_icon["brush"]
                                                   ,variable=self.tools_var["brush"]
                                                   ,onvalue=1,offvalue=0,indicatoron=False
                                                   ,command=lambda: self.press_act("brush")),
                    "eraser": tkinter.Checkbutton(self.seg_editor
                                                    ,image=self.tools_icon["eraser"]
                                                    ,variable=self.tools_var["eraser"]
                                                    ,onvalue=1,offvalue=0,indicatoron=False
                                                    ,command=lambda: self.press_act("eraser")),
                    "add_bbox": tkinter.Button(self.seg_editor
                                               ,image=self.tools_icon["add_bbox"]
                                               ,command=self.ADDBBOX_CALL)}
        
        self.marker_size_var = tkinter.IntVar(value=15)
        self.marker_size_scale = tkinter.Scale(self.toolbank,
                                                from_=10,
                                                to=30,
                                                orient=tkinter.HORIZONTAL,
                                                label="Marker Size",
                                                variable=self.marker_size_var
                                            )
        self.sep2 = tkinter.Frame(self.toolbank, height=1, bd=0, relief=tkinter.SUNKEN, bg="black")
        self.contrast_var = tkinter.IntVar(value=250)
        self.contrast_bar = tkinter.Scale(self.toolbank,
                                          from_=0,
                                          to=500,
                                          orient=tkinter.HORIZONTAL,
                                          label="Contrast",
                                          variable=self.contrast_var
                                          )
        self.contrast_bar.bind("<ButtonRelease-1>", self.on_contrast_bar_change)
        self.contrast_reset = tkinter.Button(self.toolbank, text="Reset", command=lambda: self.on_contrast_bar_change(None))
        self.sep3 = tkinter.Frame(self.toolbank, height=1, bd=0, relief=tkinter.SUNKEN, bg="black")
        self.brightness_var = tkinter.IntVar(value=250)
        self.brightness_bar = tkinter.Scale(self.toolbank,
                                             from_=0,
                                             to=500,
                                             orient=tkinter.HORIZONTAL,
                                             label="Brightness",
                                             variable=self.brightness_var
                                            )
        self.brightness_bar.bind("<ButtonRelease-1>", self.on_brightness_bar_change)
        self.brightness_reset = tkinter.Button(self.toolbank, text="Reset", command=lambda: self.on_brightness_bar_change(None))
        self.channel_var = tkinter.StringVar(value="647")
        self.channel_selector = tkinter.OptionMenu(self.toolbank, self.channel_var, "")

    def update_channel_menu(self, channels: list[str]):
        menu = self.channel_selector["menu"]
        menu.delete(0, "end")
        for ch in channels:
            menu.add_command(label=ch,
                            command=lambda v=ch: self.on_channel_change(v))
        # set default
        if channels:
            self.channel_var.set(channels[0])
        # disable the widget if there's only one choice
        state = "normal" if len(channels) > 1 else "disabled"
        self.channel_selector.configure(state=state)

    def pack(self):
        self.toolbank.pack(side=tkinter.RIGHT, fill=tkinter.BOTH)
        self.button1.pack(side=tkinter.TOP, fill=tkinter.X)
        self.button2.pack(side=tkinter.TOP, fill=tkinter.X)
        self.sep.pack(fill=tkinter.X)
        self.seg_editor.pack(side=tkinter.TOP, fill=tkinter.X)
        self.tools["brush"].grid(row=0, column=0)
        self.tools["eraser"].grid(row=0, column=1)
        self.tools["add_bbox"].grid(row=1, column=0, columnspan=2)
        self.marker_size_scale.pack(side=tkinter.TOP, fill=tkinter.X)
        self.sep2.pack(fill=tkinter.X)
        self.contrast_bar.pack(side=tkinter.TOP, fill=tkinter.X)
        self.contrast_reset.pack(side=tkinter.TOP, fill=tkinter.X)
        self.sep3.pack(fill=tkinter.X)
        self.brightness_bar.pack(side=tkinter.TOP, fill=tkinter.X)
        self.brightness_reset.pack(side=tkinter.TOP, fill=tkinter.X)
        self.channel_selector.pack(side=tkinter.TOP, fill=tkinter.X)

    def on_contrast_bar_change(self, event):
        if event is None:
            self.contrast_var.set(250)
        contrast_value = self.contrast_var.get()
        factor = (0.01 * contrast_value) - 1.5
        self.gui.getStove().adjust_contrast(factor)
    
    def on_brightness_bar_change(self, event):
        if event is None:
            self.brightness_var.set(250)
        brightness_value = self.brightness_var.get()
        factor = (250 - brightness_value) * 0.005
        self.gui.getStove().adjust_brightness(factor)
    
    def press_act(self, widget: str):
        if not self.gui.getFuncButton().segButtonPressed():
            self.gui.popBox("w", "Segmentation Mode", "Please enter Segmentation mode first")
            for k, v in self.tools_var.items():
                v.set(0)
            return
        for k, v in self.tools_var.items():
            if k != widget:
                v.set(0)
    
    def get_marker_size(self) -> int:
        return self.marker_size_var.get()
    def burshButtonPressed(self) -> bool:
        return self.tools_var["brush"].get()
    def eraserButtonPressed(self) -> bool:
        return self.tools_var["eraser"].get()
    
    def ADDBBOX_CALL(self):
        if not self.gui.getFuncButton().bboxButtonPressed():
            self.gui.popBox("w", "BBOX Mode", "Please enter BBOX mode first")
            return
        loaded_image = self.gui.getStove().getLoaded()
        if not loaded_image:
            self.gui.popBox("w", "No Image", "No image is loaded")
            return
        from .canvas_view import box
        width, height = loaded_image.getImgNumpyRGB().shape[1], loaded_image.getImgNumpyRGB().shape[0]
        bbox = [width // 2 - 150, height // 2 - 150, width // 2 + 150, height // 2 + 150]
        new_box = box(bbox, self.gui)
        loaded_image.bbox.append(new_box)
        new_box.selected = True
        new_box.draw = True
        box.setBuffer(new_box)
        self.gui.getStove().canvas.draw()

    def SAVEPROG_CALL(self):
        from .thumbnails import abstract
        self.gui.indicateWait("Pkl save")
        def job():
            try:
                from tkinter import filedialog
                import pickle
                f = filedialog.asksaveasfilename(defaultextension=".pkl", 
                                               filetypes=[("Pickle files", "*.pkl")],
                                               title="Save Session As")
                if f:
                    b = abstract.grabPool()
                    with open(f, "wb") as file:
                        pickle.dump(b, file)
                    self.gui.popBox("i", "Done", "Session saved as " + f)
            except Exception as e:
                self.gui.popBox("e", "Save Error", f"Failed to save: {e}")
            finally:
                self.gui.getRoot().after(0, self.gui.dismissWait)
        
        import threading
        threading.Thread(target=job, daemon=True).start()
    
    def LOADPROG_CALL(self):
        from .thumbnails import abstract
        self.gui.indicateWait("Pkl load")
        def job():
            try:
                from tkinter import filedialog
                import pickle
                import pathlib
                f = filedialog.askopenfilename(filetypes=[("Progress files", "*.pkl")])
                if f:
                    with open(f, "rb") as file:
                        data = pickle.load(file)
                    abstract.getPool().clear()
                    for bundle in data:
                        path, bbox_data, seg_data = bundle
                        if not pathlib.Path(path).exists():
                            self.gui.popBox("w", "Warning", f"Image {path} not found!")
                            continue
                        from .canvas_view import box, segment
                        abs_obj = abstract(pathlib.Path(path), self.gui.getTifSequence().gallery_frame, self.gui)
                        abs_obj.bbox = [box(each, self.gui) for each in bbox_data]
                        abs_obj.segment = [segment(self.gui, segm) for segm in seg_data]
                    abstract.sendFirst()
            except Exception as e:
                self.gui.popBox("e", "Load Error", f"Failed to load session: {e}")
            finally:
                self.gui.getRoot().after(0, self.gui.dismissWait)
        
        import threading
        threading.Thread(target=job, daemon=True).start()

    def on_channel_change(self, new_chan: str):
        from .thumbnails import abstract
        from PIL import Image, ImageTk
        
        # Update the channel variable to reflect the change in the UI
        self.channel_var.set(new_chan)
        
        abs_obj = self.gui.getStove().getLoaded()
        if not abs_obj:
            return

        # turn off old overlays
        abs_obj.drawSegmentation = False

        # update the pointer
        abs_obj.selected_channel = new_chan
        abs_obj._abstract__seg = (
            abs_obj._abstract__seg_647
            if new_chan == "647"
            else abs_obj._abstract__seg_488
        )
        abs_obj._abstract__img_np_cyto = (
            abs_obj._abstract__img_np_cyto1
            if new_chan == "647"
            else abs_obj._abstract__img_np_cyto2
        )

        # rebuild the thumbnail
        rgb = abstract.grayscale_to_rgb(abs_obj._abstract__img_np_cyto)
        abs_obj._abstract__img_pil_thumbnail = Image.fromarray(rgb).resize((64, 64))
        abs_obj._abstract__img_tk_thumbnail = ImageTk.PhotoImage(abs_obj._abstract__img_pil_thumbnail)
        abs_obj.getLabel().config(image=abs_obj._abstract__img_tk_thumbnail)

        # redraw everything
        self.gui.getStove().cook(abs_obj)
        abs_obj.drawSegmentation = True
