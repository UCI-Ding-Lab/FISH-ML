from __future__ import annotations
import time
import tkinter
import pathlib
import pickle
import os
import numpy as np
from PIL import (Image, ImageTk, ImageDraw)
from tkinter import filedialog, messagebox, simpledialog
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.patches import Rectangle, Circle
from matplotlib.backend_bases import MouseEvent
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from skimage import measure
from datasets import Dataset
import fishCore
import threading
import concurrent.futures
import logging

class FishToolBar(NavigationToolbar2Tk):
    def __init__(self, canvas, window, gui: FishGUI):
        super().__init__(canvas, window)
        self.fishGUI = gui
    
    def resetToolBank(self):
        self.fishGUI.getSeasoning().tools_var["brush"].set(0)
        self.fishGUI.getSeasoning().tools_var["eraser"].set(0)
    
    # overwriting
    def home(self):
        self.resetToolBank()
        super().home()
    def zoom(self, *args):
        self.resetToolBank()
        super().zoom(*args)

# debug
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

class bundle():
    def __init__(self) -> None:
        self.path: pathlib.Path = None
        self.bbox: list[list] = None
        self.segment: list[np.ndarray] = None

class progress():
    @staticmethod
    def save():
        f = filedialog.asksaveasfilename(defaultextension=".pkl", 
                                         filetypes=[("Pickle files", "*.pkl")],
                                         title="Save Session As")
        if not f: 
            return
        with open(f, "wb") as file:
            pickle.dump(abstract.zip(), file)
        messagebox.showinfo("Done", "Session saved as " + f)
    
    @staticmethod
    def load(gui: FishGUI):
        f = filedialog.askopenfilename(filetypes=[("Progress files", "*.pkl")])
        if not f: return
        try:
            with open(f, "rb") as file:
                data = pickle.load(file)
            abstract.getPool().clear()
            for b in data:
                bund: bundle = b
                if not pathlib.Path(bund.path).exists():
                    messagebox.showwarning("Warning", f"Image {bund.path} not found!")
                    continue
                abs = abstract(bund.path, gui.getTifSequence().gallery_frame, gui)
                abs.bbox = [box(each, gui) for each in bund.bbox]
                abs.segment = [segment(gui, seg) for seg in bund.segment]
                abs.bbox_generated = True if abs.bbox else False
            abstract.sendFirst()
            messagebox.showinfo("Done", "Session loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load session: {e}")
    
    @staticmethod
    def export(gui: FishGUI):
        selected_path = filedialog.askdirectory(title="Save Under...")
        if selected_path:
            folder_name = simpledialog.askstring("Dataset", "Enter the name of the Dataset folder:")
            if folder_name:
                new_folder_path = pathlib.Path(selected_path) / folder_name
                try:
                    os.makedirs(new_folder_path)
                except FileExistsError:
                    messagebox.showwarning("Warning", "Folder already exists!")
                    return
        toSave = [i for i in abstract.getPool() if i.selected and len(i.segmentExplict)]
        d = {"name":[],"image":[],"xy":[],"masks":[]}
        for abs in toSave:
            d["name"].append(abs.getAbsPath())
            d["image"].append(abs.getImgNumpyRGB())
            d["xy"].append([seg.xy for seg in abs.segment])
            d["masks"].append([seg.box for seg in abs.segment])
        dataset = Dataset.from_dict(d)
        dataset.save_to_disk(new_folder_path)
        messagebox.showinfo("Done", "Dataset exported successfully!")

    @staticmethod
    def generateBbox(gui: FishGUI, abstracts: list[abstract]):
        """Generate bounding boxes in the background using threading and concurrency."""
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        def generate_bboxes():
            def generate_bbox(abs):
                start_time = time.time()
                _ = abs.bbox
                abs.bbox_generated = True
                if not gui.getFuncButton().selectButtonPressed():
                    abs.thumbnail = "bbox"
                end_time = time.time()
                logging.info(f"Generated bbox for {abs.getAbsPath()} in {end_time - start_time:.4f} seconds")
            max_workers = min(1, len(abstracts))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(generate_bbox, abstracts)
        threading.Thread(target=generate_bboxes, daemon=True).start() # Start the thread of generating bboxes
                
class segment():
    __buffer: segment = None
    def __init__(self, gui: FishGUI, data: np.ndarray):
        # transpose to match matplotlib axis
        self.__data: np.ndarray = data.T
        self.gui = gui
        self.__patch: PathPatch = None
        self.__draw: bool = False
        self.__exist: bool = True
        self.__selected: bool = False
    
    @property
    def xy(self):
        ys, xs = np.where(self.__data.T == 1)
        return xs.min(), ys.min()
    
    @property
    def box(self):
        ys, xs = np.where(self.__data.T == 1)
        return self.__data.T[xs.min():xs.max(), ys.min():ys.max()]
        
    
    @property
    def patch(self) -> PathPatch:
        if not self.__patch:
            c = measure.find_contours(self.__data, level=0.5)[0]
            vertices = np.array(c)
            codes = np.full(len(vertices), Path.LINETO)
            codes[0] = Path.MOVETO
            path = Path(vertices, codes)
            self.__patch = PathPatch(path, facecolor='none', edgecolor='orange', linewidth=0.5)
        return self.__patch
    
    @timer
    def contour_wrapper(self, data):
        return measure.find_contours(data, level=0.5)[0]
    
    @timer
    def recal_patch(self):
        c = self.contour_wrapper(self.__data)
        vertices = np.array(c)
        codes = np.full(len(vertices), Path.LINETO)
        codes[0] = Path.MOVETO
        self.patch.get_path().vertices = vertices
        self.patch.get_path().codes = codes
    
    @property
    def selected(self) -> bool:
        return self.__selected
    @selected.setter
    def selected(self, value: bool):
        if self.__selected == value:
            return
        self.__selected = value
        self.draw = False
        self.patch.set_edgecolor('cyan' if value else 'orange')
        canvas = self.gui.getStove().canvas
        subplot = self.gui.getStove().subplot
        background = canvas.copy_from_bbox(subplot.bbox)
        canvas.restore_region(background)
        subplot.draw_artist(self.patch)
        canvas.blit(subplot.bbox) # use blitting to optimize the drawing speed
        self.draw = True
    
    @property
    def draw(self) -> bool:
        return self.__draw
    @draw.setter
    def draw(self, value: bool):
        if self.__draw == value: return
        if value:
            self.gui.getStove().subplot.add_patch(self.patch)
        else:
            self.patch.remove()
        self.gui.getStove().canvas.draw()
        self.__draw = value
    
    @property
    def exist(self) -> bool:
        return self.__exist
    
    @classmethod
    def setBuffer(cls, segment: segment):
        cls.__buffer = segment
    @classmethod
    def getBuffer(cls) -> segment:
        return cls.__buffer
    @classmethod
    def clearBuffer(cls):
        cls.__buffer = None
    @classmethod
    def clearBufferAndDeselect(cls):
        current = cls.getBuffer()
        if current: current.selected = False
        cls.__buffer = None
    
    def contains(self, x: float, y: float) -> bool:
        p = self.gui.getStove().subplot.transData.transform((x, y))
        return self.patch.contains_point(p)

    def update_mask(self, x, y, radius, erase=False):
        x_int, y_int = int(x), int(y)
        for i in range(x_int - radius, x_int + radius + 1):
            for j in range(y_int - radius, y_int + radius + 1):
                if (i - x_int)**2 + (j - y_int)**2 <= radius**2:
                    if 0 <= i < self.__data.shape[0] and 0 <= j < self.__data.shape[1]:
                        if erase:
                            self.__data[i, j] = 0  # Erase the mask (set to 0)
                        else:
                            self.__data[i, j] = 1  # Draw the mask (set to 1)

    def delete(self):
        abs = self.gui.getStove().getLoaded()
        if abs and self in abs.segment:
            abs.segment.remove(self)
            self.draw = False
            segment.clearBuffer()
            self.gui.getStove().canvas.flush_events()
    
class anchor():
    __buffer: anchor = None
    def __init__(self, x: float, y: float, gui: FishGUI, master: box, location: str):
        self.gui = gui
        self.__color: str = 'cyan'
        self.__draw: bool = False
        self.__selected: bool = False
        self.__loc: str = location
        self.__patch = Circle((x, y), radius=10, linewidth=0.5, edgecolor=self.color, facecolor='none')
    
    @property
    def patch(self) -> Circle:
        return self.__patch
    @property
    def location(self) -> str:
        return self.__loc
    
    @property
    def color(self) -> str:
        return self.__color
    @color.setter
    def color(self, value: str):
        self.__color = value
        self.draw = False
        self.patch.set_edgecolor(value)
        self.draw = True
    
    @property
    def draw(self) -> bool:
        return self.__draw
    @draw.setter
    def draw(self, value: bool):
        if self.__draw == value: return
        if value:
            self.gui.getStove().subplot.add_patch(self.patch)
        else:
            self.patch.remove()
        self.gui.getStove().canvas.draw()
        self.__draw = value
    
    @property
    def selected(self) -> bool:
        return self.__selected
    @selected.setter
    def selected(self, value: bool):
        if self.__selected == value: return
        self.color = 'b' if value else 'cyan'
        self.__selected = value
    
    def contains(self, x: float, y: float) -> bool:
        p = self.gui.getStove().subplot.transData.transform((x, y))
        return self.patch.contains_point(p)
    
    @classmethod
    def setBuffer(cls, anchor: anchor):
        cls.__buffer = anchor
    @classmethod
    def getBuffer(cls) -> anchor:
        return cls.__buffer
    @classmethod
    def clearBuffer(cls):
        if cls.getBuffer(): cls.getBuffer().selected = False
        cls.setBuffer(None)

class box():
    __buffer: box = None
    def __init__(self, bbox: list, gui: FishGUI):
        self.gui = gui
        self.__bbox = bbox
        min_x, min_y, max_x, max_y = self.__bbox
        self.__rect = Rectangle((min_x, min_y),
                                max_x - min_x,
                                max_y - min_y,
                                linewidth=float(self.gui.getBackEnd().config["info"]["bbox_preview_line_width"]),
                                edgecolor='r',
                                facecolor='none')
        self.__anchors = {"bottom-left": anchor(min_x, min_y, gui, self, "bottom-left"),
                          "bottom-right": anchor(max_x, min_y, gui, self, "bottom-right"),
                          "top-left": anchor(min_x, max_y, gui, self, "top-left"),
                          "top-right": anchor(max_x, max_y, gui, self, "top-right"),
                          "pos-anchor": anchor((max_x - min_x) // 2 + min_x, max_y, gui, self, "pos-anchor")}
        self.__draw: bool = False
        self.__selected: bool = False
    
    @property
    def rect(self) -> Rectangle:
        return self.__rect
    @property
    def anchors(self) -> dict[str,anchor]:
        return self.__anchors
    @property
    def final(self) -> list:
        return [self.rect.get_x(),
                self.rect.get_y(),
                self.rect.get_x() + self.rect.get_width(),
                self.rect.get_y() + self.rect.get_height()]
    
    @property
    def selected(self) -> bool:
        return self.__selected
    @selected.setter
    def selected(self, value: bool):
        if self.__selected == value:
            return
        self.__selected = value
        self.draw = False
        self.rect.set_edgecolor('cyan' if value else 'r')
        for _, anchor_obj in self.__anchors.items():
            anchor_obj.draw = value
        canvas = self.gui.getStove().canvas
        subplot = self.gui.getStove().subplot
        background = canvas.copy_from_bbox(subplot.bbox)
        canvas.restore_region(background)
        subplot.draw_artist(self.rect)
        if self.__selected:
            for anchor_obj in self.__anchors.values():
                subplot.draw_artist(anchor_obj.patch)
        canvas.blit(subplot.bbox) # use blitting to optimize the drawing speed
        self.draw = True
    
    @property
    def draw(self):
        return self.__draw
    @draw.setter
    def draw(self, value: bool):
        if self.__draw == value: return
        if value:
            self.gui.getStove().subplot.add_patch(self.rect)
        else:
            self.rect.remove()
        self.gui.getStove().canvas.draw()
        self.__draw = value
    
    def contains(self, x: float, y: float) -> bool:
        p = self.gui.getStove().subplot.transData.transform((x, y))
        return self.rect.contains_point(p)
    def anchorContains(self, x: float, y: float) -> str:
        for k, v in self.anchors.items():
            if v.contains(x, y):
                return k
        return None
    def anchorUpdate(self):
        self.anchors["bottom-left"].patch.set_center((self.rect.get_x(), self.rect.get_y()))
        self.anchors["bottom-right"].patch.set_center((self.rect.get_x() + self.rect.get_width(), self.rect.get_y()))
        self.anchors["top-left"].patch.set_center((self.rect.get_x(), self.rect.get_y() + self.rect.get_height()))
        self.anchors["top-right"].patch.set_center((self.rect.get_x() + self.rect.get_width(), self.rect.get_y() + self.rect.get_height()))
        self.anchors["pos-anchor"].patch.set_center((self.rect.get_x() + self.rect.get_width() / 2, self.rect.get_y() + self.rect.get_height()))
    @classmethod
    def setBuffer(cls, box: box):
        cls.__buffer = box
    @classmethod
    def getBuffer(cls) -> box:
        return cls.__buffer
    @classmethod
    def clearBuffer(cls):
        cls.__buffer = None
    @classmethod
    def clearBufferAndDeselect(cls):
        current = cls.getBuffer()
        if current: current.selected = False
        cls.__buffer = None

    def delete(self):
        abs = self.gui.getStove().getLoaded()
        if abs and self in abs.bbox:
            abs.bbox.remove(self)
            self.draw = False
            for anchor in self.anchors.values():
                anchor.draw = False
            box.clearBuffer()
            self.gui.getStove().canvas.flush_events()
        
class seasoning():
    def __init__(self, gui: FishGUI):
        self.gui = gui
        self.toolbank = tkinter.Frame(self.gui.getLowerFrame().getFrameA(), width=150, background="grey")
        self.button1 = tkinter.Button(self.toolbank, height=2, text="Save Progress", command=progress.save)
        self.button2 = tkinter.Button(self.toolbank, height=2, text="Load Progress", command=lambda: progress.load(gui))
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
                                               ,command=self.add_bbox_call)}
        
        self.marker_size_var = tkinter.IntVar(value=15)
        self.marker_size_scale = tkinter.Scale(self.toolbank,
                                                from_=10,
                                                to=30,
                                                orient=tkinter.HORIZONTAL,
                                                label="Marker Size",
                                                variable=self.marker_size_var
                                            )

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
    
    def press_act(self, widget: str):
        if not self.gui.getFuncButton().segButtonPressed():
            FishGUI.popBox("w", "Segmentation Mode", "Please enter Segmentation mode first")
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
    
    def add_bbox_call(self):
        if not self.gui.getFuncButton().bboxButtonPressed():
            FishGUI.popBox("w", "BBOX Mode", "Please enter BBOX mode first")
            return
        loaded_image = self.gui.getStove().getLoaded()
        if not loaded_image:
            FishGUI.popBox("w", "No Image", "No image is loaded")
            return
        width, height = loaded_image.getImgNumpyRGB().shape[1], loaded_image.getImgNumpyRGB().shape[0]
        bbox = [width // 2 - 150, height // 2 - 150, width // 2 + 150, height // 2 + 150]
        new_box = box(bbox, self.gui)
        loaded_image.bbox.append(new_box)
        new_box.selected = True
        new_box.draw = True
        self.gui.getStove().canvas.draw()
        
class stove():
    BILT_BUFFER1 = None
    BILT_BUFFER2 = None
    BILT_BUFFER3 = None
    def __init__(self, gui: FishGUI):
        self.gui = gui
        self.pit = tkinter.Frame(self.gui.getLowerFrame().getFrameA(), background="black")
        self.sep = tkinter.Frame(self.gui.getLowerFrame().getFrameA(), width=1, bd=0, relief=tkinter.SUNKEN, bg="black")
        
    
        self.figure = Figure(figsize=(3,3), dpi=200)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.subplot = self.figure.add_subplot(111)
        self.subplot.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.figure, self.pit)
        self.canvas.mpl_connect("button_press_event", self.onCanvasClick)
        self.canvas.mpl_connect("button_release_event", self.onCanvasRelease)
        self.canvas.mpl_connect("motion_notify_event", self.onCanvasDrag)
        self.toolbar = FishToolBar(self.canvas, self.pit, self.gui)
        self.toolbar.update()
        self.tb_pointer = Circle((0, 0), 15, linewidth=0.5, edgecolor='cyan', facecolor='none')
        self.xs = []
        self.ys = []
        self.markers: list[Circle] = []
        self.press = False
        
        self.__onLoad: abstract = None
    
    def get_tb_pointer(self) -> Circle:
        self.tb_pointer.set_radius(self.gui.getSeasoning().get_marker_size())
        return self.tb_pointer
        
    @property
    def biltbg(self):
        return self.canvas.copy_from_bbox(self.subplot.bbox)
    def bufferSetCurrent(self, buffer):
        if buffer == 1:
            self.BILT_BUFFER1 = self.biltbg
        elif buffer == 2:
            self.BILT_BUFFER2 = self.biltbg
        elif buffer == 3:
            self.BILT_BUFFER3 = self.biltbg

    def pack(self):
        self.pit.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)
        self.sep.pack(side=tkinter.LEFT, fill=tkinter.Y)
        
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        self.toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH)
        
    def cook(self, abs: abstract):
        self.setLoaded(abs)
        self.subplot.imshow(self.getLoaded().getImgNumpyRGB())
        self.subplot.set_axis_off()
        self.canvas.draw()
    def dump(self):
        self.clearLoaded()
        self.subplot.clear()
        self.subplot.set_axis_off()
        self.canvas.draw()
        
    def onCanvasClick(self, event: MouseEvent):
        self.press = True
        if stove.isLeftClick(event):
            if event.inaxes != self.subplot:
                return
            if self.gui.getFuncButton().bboxButtonPressed():
                # check if click on anchor
                if box.getBuffer() and box.getBuffer().selected:
                    anchorName = box.getBuffer().anchorContains(event.xdata, event.ydata)
                    if anchorName:
                        target = box.getBuffer().anchors[anchorName]
                        target.selected = True
                        anchor.setBuffer(target)
                        return
                # check if click on box
                target = self.getLoaded().findBoxFromPoint(event.xdata, event.ydata)
                box.clearBufferAndDeselect()
                if target:
                    target.selected = True
                    box.setBuffer(target)
            elif self.gui.getFuncButton().segButtonPressed():
                if self.gui.getSeasoning().burshButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
                    self.xs = [event.xdata]
                    self.ys = [event.ydata]
                    
                    self.bufferSetCurrent(1)
                    self.bufferSetCurrent(2)
                    self.canvas.restore_region(self.BILT_BUFFER1)
                    self.marker_draw(event.xdata, event.ydata)
                    self.canvas.blit(self.subplot.bbox)
                    
                elif self.gui.getSeasoning().eraserButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
                    self.xs = [event.xdata]
                    self.ys = [event.ydata]
                    self.marker_draw(event.xdata, event.ydata)
                    self.get_tb_pointer().set_center((event.xdata, event.ydata))
                    self.subplot.add_patch(self.get_tb_pointer())
                    self.canvas.draw()
                else:
                    target = self.getLoaded().findSegFromPoint(event.xdata, event.ydata)
                    segment.clearBufferAndDeselect()
                    if target:
                        target.selected = True
                        segment.setBuffer(target)
    def onCanvasRelease(self, event: MouseEvent):
        self.press = False
        if self.gui.getFuncButton().bboxButtonPressed():
            anchor.clearBuffer()
        elif self.gui.getFuncButton().segButtonPressed():
            if self.gui.getSeasoning().burshButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
                final = list(zip(self.xs, self.ys))
                
                for marker in self.markers:
                    marker.remove()
                self.markers.clear()
                for x, y in final:
                    segment.getBuffer().update_mask(x, y, self.gui.getSeasoning().get_marker_size())
                segment.getBuffer().recal_patch()
                self.canvas.draw_idle()
                
                self.xs.clear()
                self.ys.clear()

            elif self.gui.getSeasoning().eraserButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
                final = list(zip(self.xs, self.ys))
                for marker in self.markers:
                    marker.remove()
                self.markers.clear()
                self.get_tb_pointer().remove()
                self.canvas.draw_idle()
                for x, y in final:
                    segment.getBuffer().update_mask(x, y, self.gui.getSeasoning().get_marker_size(), erase=True)
                segment.getBuffer().recal_patch()
                self.xs.clear()
                self.ys.clear()

    def onCanvasDrag(self, event: MouseEvent):
        if self.gui.getFuncButton().bboxButtonPressed() and self.press:
            a = anchor.getBuffer()
            b = box.getBuffer()
            if a and a.selected and b and b.selected:
                if event.inaxes != b.rect.axes: return
                x0, y0 = b.rect.get_xy()
                width, height = b.rect.get_width(), b.rect.get_height()
                if a.location == "bottom-left":
                    new_x0 = event.xdata
                    new_y0 = event.ydata
                    width += x0 - new_x0
                    height += y0 - new_y0
                    b.rect.set_xy((new_x0, new_y0))
                elif a.location == "bottom-right":
                    width = event.xdata - x0
                    height += y0 - event.ydata
                    b.rect.set_xy((x0, event.ydata))
                elif a.location == "top-right":
                    width = event.xdata - x0
                    height = event.ydata - y0
                elif a.location == "top-left":
                    new_x0 = event.xdata
                    height = event.ydata - y0
                    width += x0 - new_x0
                    b.rect.set_xy((new_x0, y0))
                elif a.location == "pos-anchor":
                    dx = event.xdata - (b.rect.get_x() + width / 2)
                    dy = event.ydata - (b.rect.get_y() + height)
                    new_x0 = x0 + dx
                    new_y0 = y0 + dy
                    b.rect.set_xy((new_x0, new_y0))
                b.rect.set_width(width)
                b.rect.set_height(height)
                b.anchorUpdate()
                self.canvas.draw()
        elif self.gui.getFuncButton().segButtonPressed() and self.press:
            if self.gui.getSeasoning().burshButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
                current_x, current_y = event.xdata, event.ydata
                # connect the current point with the last point
                if len(self.xs) > 0 and len(self.ys) > 0:
                    last_x, last_y = self.xs[-1], self.ys[-1]
                    distance = np.sqrt((current_x - last_x) ** 2 + (current_y - last_y) ** 2)
                    if distance > 5: # distance threshold
                        num_intermediate_points = int(distance // 1)
                        x_values = np.linspace(last_x, current_x, num_intermediate_points + 1)
                        y_values = np.linspace(last_y, current_y, num_intermediate_points + 1)
                        for x, y in zip(x_values, y_values):
                            self.xs.append(x)
                            self.ys.append(y)
                            
                            self.bufferSetCurrent(1)
                            self.canvas.restore_region(self.BILT_BUFFER1)
                            self.marker_draw(x, y)
                            self.canvas.blit(self.subplot.bbox)       
                else:
                    self.xs.append(current_x)
                    self.ys.append(current_y)
                    
                    self.canvas.bufferSetCurrent(1)
                    self.canvas.restore_region(self.BILT_BUFFER1)
                    self.marker_draw(x, y)
                    self.canvas.blit(self.subplot.bbox)
                segment.getBuffer().update_mask(current_x, current_y, self.gui.getSeasoning().get_marker_size())

            elif self.gui.getSeasoning().eraserButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
                current_x, current_y = event.xdata, event.ydata
                if len(self.xs) > 0 and len(self.ys) > 0:
                    last_x, last_y = self.xs[-1], self.ys[-1]
                    distance = np.sqrt((current_x - last_x) ** 2 + (current_y - last_y) ** 2)
                    if distance > 5: # distance threshold
                        num_intermediate_points = int(distance // 1)
                        x_values = np.linspace(last_x, current_x, num_intermediate_points + 1)
                        y_values = np.linspace(last_y, current_y, num_intermediate_points + 1)
                        for x, y in zip(x_values, y_values):
                            self.xs.append(x)
                            self.ys.append(y)
                            self.marker_draw(x, y)
                            self.get_tb_pointer().set_center((x, y))
                else:
                    self.xs.append(current_x)
                    self.ys.append(current_y)
                    self.marker_draw(current_x, current_y)
                    self.get_tb_pointer().set_center((current_x, current_y))
                self.canvas.draw()
                segment.getBuffer().update_mask(current_x, current_y, self.gui.getSeasoning().get_marker_size(), erase=True)

    def marker_draw(self, x, y):
        circle = Circle((x, y), self.gui.getSeasoning().get_marker_size(), color='red', alpha=0.01)
        self.markers.append(circle)
        self.subplot.add_patch(circle)
        self.subplot.draw_artist(circle)
    
    def isLoaded(self) -> bool:
        return self.__onLoad is not None
    def getLoaded(self) -> abstract:
        return self.__onLoad
    def setLoaded(self, abs: abstract):
        self.__onLoad = abs
    def clearLoaded(self):
        self.__onLoad = None
    
    @staticmethod
    def isLeftClick(event: MouseEvent) -> bool:
        return event.button == 1

class abstract():
    __pool: list[abstract] = []
    __buffer: abstract = None
    
    def __init__(self, path: pathlib.Path, gallery_frame, gui: FishGUI):
        self.__abs_path = path
        self.gui = gui
        
        self.__img_np_gs = np.array(Image.open(self.__abs_path)) # 2048 2048
        self.__img_np_rgb = self.grayscale_to_rgb(self.__img_np_gs) # 2048 2048 3
        self.__img_pil_thumbnail = Image.fromarray(self.__img_np_rgb).resize((64, 64))
        self.__img_tk_thumbnail = ImageTk.PhotoImage(self.__img_pil_thumbnail)
        
        self.__label = tkinter.Label(gallery_frame,
                                     image=self.__img_tk_thumbnail,
                                     width=64,
                                     height=64,
                                     relief=tkinter.FLAT,
                                     borderwidth=0)
        self.__label.pack(side=tkinter.LEFT, padx=2, pady=2)
        self.__label.bind("<Button-1>", self.on_click)
        
        self.__img_pil_thumbnail_bbox = None
        self.__img_pil_thumbnail_select = None
        self.__img_pil_thumbnail_crossout = None
        self.__thumbnail: str = None
        self.__bbox: list[box] = []
        self.__highlighted: str = None
        self.__selected: bool = True
        self.__drawBbox: bool = False
        self.__seg: list[segment] = []
        self.__drawSeg: bool = False

        self.__bbox_generated: bool = False

        abstract.addToPool(self)
        
    @property
    def segment(self) -> list[segment]:
        if not self.__seg:
            masks: np.ndarray = self.gui.getBackEnd().finetune.AppIntPREDICTwrapper(self.getImgNumpyGreyscale(), self.boundingBoxRevised)
            for m in masks:
                self.__seg.append(segment(self.gui, m))
        return self.__seg
    @segment.setter
    def segment(self, value: list[segment]):
        self.__seg = value
    @segment.deleter
    def segment(self):
        self.__seg = []
    
    @property
    def segmentExplict(self) -> list[segment]:
        return self.__seg
    
    @property
    def thumbnail(self) -> str:
        return self.__thumbnail
    @thumbnail.setter
    def thumbnail(self, value: str):
        self.__thumbnail = value
        if value == "default":
            self.getLabel().config(image=self.__img_tk_thumbnail)
        elif value == "bbox":
            if not self.__img_pil_thumbnail_bbox:
                self.__img_pil_thumbnail_bbox = self.__img_pil_thumbnail.copy()
                ImageDraw.Draw(self.__img_pil_thumbnail_bbox).ellipse((49, 5, 59, 15), fill=(0,0,255))
                self.__img_pil_thumbnail_bbox = ImageTk.PhotoImage(self.__img_pil_thumbnail_bbox)
            self.getLabel().config(image=self.__img_pil_thumbnail_bbox)
        elif value == "selected":
            if not self.__img_pil_thumbnail_select:
                self.__img_pil_thumbnail_select = self.__img_pil_thumbnail.copy()
                ImageDraw.Draw(self.__img_pil_thumbnail_select).ellipse((5, 5, 15, 15), fill=(0,255,0))
                self.__img_pil_thumbnail_select = ImageTk.PhotoImage(self.__img_pil_thumbnail_select)
            self.getLabel().config(image=self.__img_pil_thumbnail_select)
        elif value == "crossout":
            if not self.__img_pil_thumbnail_crossout:
                self.__img_pil_thumbnail_crossout = self.__img_pil_thumbnail.copy()
                ImageDraw.Draw(self.__img_pil_thumbnail_crossout).line((5, 5, 15, 15), fill=(255,0,0), width=2)
                ImageDraw.Draw(self.__img_pil_thumbnail_crossout).line((5, 15, 15, 5), fill=(255,0,0), width=2)
                self.__img_pil_thumbnail_crossout = ImageTk.PhotoImage(self.__img_pil_thumbnail_crossout)
            self.getLabel().config(image=self.__img_pil_thumbnail_crossout)
    @thumbnail.deleter
    def thumbnail(self):
        self.getLabel().pack_forget()
    
    @property
    def selected(self) -> bool:
        return self.__selected
    @selected.setter
    def selected(self, value: bool):
        if value:
            self.thumbnail = "selected"
            self.__selected = True
        else:
            self.thumbnail = "crossout"
            self.__selected = False
    
    @property
    def bbox(self) -> list[box]:
        if not self.bbox_generated:
            raw = self.gui.getBackEnd().AppIntDINOwrapper(self.__img_np_gs) # list[list]
            self.__bbox = [box(each, self.gui) for each in raw]
            self.bbox_generated = True
        return self.__bbox
    @bbox.setter
    def bbox(self, value: list[box]):
        self.__bbox = value

    @property
    def bbox_generated(self) -> bool:
        return self.__bbox_generated
    @bbox_generated.setter
    def bbox_generated(self, value: bool):
        self.__bbox_generated = value

    @property
    def boundingBoxRevised(self) -> list[list]:
        return [b.final for b in self.bbox]
    
    @property
    def segmentationRevised(self) -> list[np.ndarray]:
        return [s._segment__data.T for s in self.segment]
    
    @property
    def drawBbox(self) -> bool:
        return self.__drawBbox
    @drawBbox.setter
    def drawBbox(self, value: bool):
        if not self.bbox_generated:
            FishGUI.popBox("w", "No BBOX", "No BBOX is available for this image")
            self.__drawBbox = False
            return
        else:
            for b in self.bbox:
                b.draw = value
                if not value:
                    box.clearBufferAndDeselect()
        # self.thumbnail = "bbox" if value else "default"
        self.gui.getStove().canvas.draw()
        self.__drawBbox = value
    
    @property
    def drawSegmentation(self) -> bool:
        return self.__drawSeg
    @drawSegmentation.setter
    def drawSegmentation(self, value: bool):
        for s in self.segment:
            s.draw = True if value else False
        self.__drawSeg = value
    
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
    
    @classmethod
    def sendFirst(cls):
        for target in cls.getPool():
            if target.selected:
                target.on_click(None)
                return
    @classmethod
    def sendFocused(cls):
        current = cls.getBuffer()
        if current: current.on_click(None)
        else: cls.sendFirst()
    @classmethod
    def getPool(cls) -> list[abstract]:
        return cls.__pool
    @classmethod
    def addToPool(cls, abs: abstract):
        cls.__pool.append(abs)
    @classmethod
    def setBuffer(cls, abs: abstract):
        cls.__buffer = abs
    @classmethod
    def getBuffer(cls) -> abstract:
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
    def saveBboxChanges(cls):
        cls.getBuffer().drawBbox = False
        cls.sendFocused()
    @classmethod
    def saveSegChanges(cls):
        cls.getBuffer().drawSegmentation = False
        cls.sendFocused()
    @classmethod
    def zip(cls) -> list[bundle]:
        result = []
        for abs in cls.getPool():
            if abs.selected:
                b = bundle()
                b.path = abs.getAbsPath()
                if abs.noBbox():
                    b.bbox = []
                    b.segment = []
                else:
                    b.bbox = abs.boundingBoxRevised
                    b.segment = abs.segmentationRevised
                result.append(b)
        return result
        
    @staticmethod
    def grayscale_to_rgb(grayscale_img) -> np.ndarray:
        rgb_img = np.zeros((2048, 2048, 3), dtype=np.uint8)
        img_uint8 = np.clip(grayscale_img//256, 0, 255).astype(np.uint8)
        rgb_img[:, :, 0] = img_uint8
        rgb_img[:, :, 1] = img_uint8
        rgb_img[:, :, 2] = img_uint8
        dynamic_exp_factor = 255.0 / np.max(rgb_img[:, :, 0])
        brightened_image = np.clip(rgb_img * dynamic_exp_factor, 0, 255).astype(np.uint8)
        return brightened_image
    
    def findBoxFromPoint(self, x: float, y: float) -> box:
        for b in self.bbox:
            if b.contains(x, y):
                return b
        return None
    def findSegFromPoint(self, x: float, y: float) -> segment:
        for s in self.segment:
            if s.contains(x, y):
                return s
        return None
    
    def getImgNumpyGreyscale(self) -> np.ndarray:
        return self.__img_np_gs
    def getImgNumpyRGB(self) -> np.ndarray:
        return self.__img_np_rgb
    def getLabel(self) -> tkinter.Label:
        return self.__label
    def getAbsPath(self) -> pathlib.Path:
        return self.__abs_path
    def noBbox(self) -> bool:
        return not len(self.__bbox)

class tifSequence():
    def __init__(self, gui: FishGUI):
        self.gui = gui
        container = gui.getLowerFrame().getFrameB()
        self.base = tkinter.Canvas(container, height=74)

        self.scrollbar = tkinter.Scrollbar(container, orient=tkinter.HORIZONTAL, command=self.base.xview)
        self.base.configure(yscrollcommand=self.scrollbar.set)

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
        if event.num == 4:  # Linux scrolling up
            self.base.xview_scroll(-1, "units")
        elif event.num == 5:  # Linux scrolling down
            self.base.xview_scroll(1, "units")
        elif event.delta:  # Windows/macOS
            if event.delta > 0:
                self.base.xview_scroll(-1, "units")
            else:
                self.base.xview_scroll(1, "units")
        
    def pack(self):
        self.base.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        self.scrollbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    
    def unpack(self):
        self.base.pack_forget()
        self.scrollbar.pack_forget()
    
    def addToGallery(self, tif_files: list[pathlib.Path]):
        for path in tif_files:
            abs = abstract(path, self.gallery_frame, self.gui)
        abstract.sendFirst()
    
    def resetPosition(self):
        self.base.xview_moveto(0)
        self.base.yview_moveto(0)
        
class funcButton():
    def __init__(self, gui: FishGUI):
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
                                    text="Export",
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
    
    def unpack(self):
        self.IMPORT.pack_forget()
        self.SELECT.pack_forget()
        self.BBOX.pack_forget()
        self.SEGMENT.pack_forget()
        self.EXPORT.pack_forget()
    
    def getButtonWidget(self, name: str):
        return {"IMPORT": self.IMPORT,
                "SELECT": self.SELECT,
                "BBOX": self.BBOX,
                "SEGMENT": self.SEGMENT,
                "EXPORT": self.EXPORT}[name]
    
    def selectButtonPressed(self) -> bool:
        return self.toggle["SELECT"].get()
    def bboxButtonPressed(self) -> bool:
        return self.toggle["BBOX"].get()
    def segButtonPressed(self) -> bool:
        return self.toggle["SEGMENT"].get()
    
    def IMPORT_call(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            folder = pathlib.Path(folder_path)
            tif_files = [str(file.resolve()) for file in folder.glob("*.tif")]
            self.gui.getTifSequence().addToGallery(tif_files)
        pool = abstract.getPool()
        progress.generateBbox(self.gui, abstracts=pool)

    def SELECT_call(self):
        if self.selectButtonPressed():
            abstract.selectAll()
        elif not self.selectButtonPressed():
            abstract.removeUnselected()
            self.gui.getTifSequence().resetPosition()
            for abs in abstract.getPool():
                abs.thumbnail = "bbox" if abs.bbox_generated else "default"
    
    def BBOX_call(self):
        if not self.gui.getStove().isLoaded():
            FishGUI.popBox("w", "Image Not Loaded", "Please select an image first")
            self.toggle["BBOX"].set(0)
            return
        if self.bboxButtonPressed():
            abs = abstract.getBuffer()
            if abs:
                if not abs.bbox_generated:
                    FishGUI.popBox("w", "Bounding Boxes Not Ready", "Bounding boxes for this image have not been generated yet.")
                    self.toggle["BBOX"].set(0)
                    return
                abstract.sendFocused()
        elif not self.bboxButtonPressed():
            abstract.saveBboxChanges()
    
    def SEGMENT_call(self):
        if not self.gui.getStove().isLoaded():
            FishGUI.popBox("w", "Image Not Loaded", "Please select an image first")
            self.toggle["SEGMENT"].set(0)
            return
        if self.bboxButtonPressed():
            FishGUI.popBox("w", "BBOX Mode", "Please exit BBOX mode first")
            self.toggle["SEGMENT"].set(0)
            return
        if self.segButtonPressed():
            abstract.sendFocused()
        if not self.segButtonPressed():
            abstract.saveSegChanges()
    
    def EXPORT_call(self):
        if not self.gui.getStove().isLoaded():
            FishGUI.popBox("w", "Image Not Loaded", "Please select an image first")
            self.toggle["EXPORT"].set(0)
            return
        if self.bboxButtonPressed():
            FishGUI.popBox("w", "BBOX Mode", "Please exit BBOX mode first")
            self.toggle["EXPORT"].set(0)
            return
        if self.segButtonPressed():
            FishGUI.popBox("w", "Segmentation Mode", "Please exit Segmentation mode first")
            self.toggle["EXPORT"].set(0)
            return
        progress.export(self.gui)

class lf():
    def __init__(self, gui: FishGUI):
        self.__a = tkinter.Frame(gui.getRoot())
        self.__s = tkinter.Frame(gui.getRoot(), height=1, bd=0, relief=tkinter.SUNKEN, bg="black")
        self.__b = tkinter.Frame(gui.getRoot(), padx=5, pady=5)
        self.__c = tkinter.Frame(gui.getRoot(), padx=5, pady=5)
    
    def pack(self):
        self.__a.pack(expand=True, fill=tkinter.BOTH)
        self.__s.pack(fill=tkinter.X)
        self.__b.pack(fill=tkinter.X)
        self.__c.pack(fill=tkinter.X)
    
    def unpack(self):
        self.__a.pack_forget()
        self.__b.pack_forget()
        self.__c.pack_forget()
    
    def getFrameA(self) -> tkinter.Frame:
        return self.__a
    def getFrameB(self) -> tkinter.Frame:
        return self.__b
    def getFrameC(self) -> tkinter.Frame:
        return self.__c

class FishGUI(object):
    def __init__(self, root):
        self.__root: tkinter.Tk = root
        self.__root.title("FISH UI Prototype")
        self.__root.geometry("870x1000")
        
        self.__be: fishCore.Fish = fishCore.Fish(pathlib.Path("./config.ini"))
        self.__be.set_model_version("3.50")
        
        self.__lf = lf(self)
        self.__tifSequence = tifSequence(self)
        self.__funcButton = funcButton(self)
        self.__seasoning = seasoning(self)
        self.__stove = stove(self)
        
        self.__lf.pack()
        self.__stove.pack()
        self.__tifSequence.pack()
        self.__funcButton.pack()
        self.__seasoning.pack()

        self.__root.bind('<BackSpace>', self.onDelete)
        self.__root.focus_set()
    
    @staticmethod    
    def popBox(type: str, title: str, message: str):
        if type == "e":
            messagebox.showerror(title, message)
        elif type == "w":
            messagebox.showwarning(title, message)
        elif type == "i":
            messagebox.showinfo(title, message)
        else:
            raise AttributeError("Invalid type")

    # Delete operation (Backspace)
    def onDelete(self, event):
        selected_box = box.getBuffer()
        selected_seg = segment.getBuffer()

        if selected_box:
            selected_box.delete()
        elif selected_seg:
            selected_seg.delete()
        else:
            self.popBox('w', 'No Selection', 'No bounding box or segmentation mask is selected.')

    def getLowerFrame(self) -> lf:
        return self.__lf
    def getStove(self) -> stove:
        return self.__stove
    def getTifSequence(self) -> tifSequence:
        return self.__tifSequence
    def getFuncButton(self) -> funcButton:
        return self.__funcButton
    def getSeasoning(self) -> seasoning:
        return self.__seasoning
    def getBackEnd(self) -> fishCore.Fish:
        return self.__be
    def getRoot(self) -> tkinter.Tk:
        return self.__root
        
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    root = tkinter.Tk()
    app = FishGUI(root)
    root.mainloop()
