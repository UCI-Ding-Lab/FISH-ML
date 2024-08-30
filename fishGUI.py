from __future__ import annotations
import tkinter
import pathlib
import pickle
import numpy as np
from PIL import (Image, ImageTk, ImageDraw)
from tkinter import filedialog, messagebox, simpledialog
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.patches import Rectangle, Circle
from matplotlib.backend_bases import MouseEvent
import fishCore

class bundle():
    def __init__(self) -> None:
        self.path: pathlib.Path = None
        self.bbox: list[list] = None

class progress():
    @staticmethod
    def save():
        f = simpledialog.askstring("Save", "Session Name: ")
        if not f: f = "fish"
        f = f + ".pkl"
        with open(f, "wb") as file:
            pickle.dump(abstract.zip(), file)
        messagebox.showinfo("Done", "Session saved as " + f)
    
    @staticmethod
    def load(gui: FishGUI):
        f = filedialog.askopenfilename(filetypes=[("Progress files", "*.pkl")])
        if not f: return
        with open(f, "rb") as file:
            data = pickle.load(file)
        for b in data:
            bund: bundle = b
            abs = abstract(bund.path, gui.getTifSequence().gallery_frame, gui)
            abs.bbox = [box(each, gui) for each in bund.bbox]
        abstract.sendFirst()

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
                          "top-right": anchor(max_x, max_y, gui, self, "top-right")}
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
        self.__selected = value
        self.draw = False
        if value:
            self.rect.set_edgecolor('g')
            for anc in self.__anchors.values():
                anc.draw = True
                anc.selected = False
        else:
            self.rect.set_edgecolor('r')
            for anc in self.__anchors.values():
                anc.selected = False
                anc.draw = False
        self.gui.getStove().canvas.draw()
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
        

class seasoning():
    def __init__(self, gui: FishGUI):
        self.gui = gui
        self.toolbank = tkinter.Frame(self.gui.getLowerFrame().getFrameA(), width=250, background="grey")
        self.button1 = tkinter.Button(self.toolbank, height=2, text="Save Progress", command=progress.save)
        self.button2 = tkinter.Button(self.toolbank, height=2, text="Load Progress", command=lambda: progress.load(gui))
        
    
    def pack(self):
        self.toolbank.pack(side=tkinter.RIGHT, expand=True, fill=tkinter.BOTH)
        self.button1.pack(side=tkinter.TOP, fill=tkinter.X)
        self.button2.pack(side=tkinter.TOP, fill=tkinter.X)

class stove():
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
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.pit)
        self.toolbar.update()
        
        self.__onLoad: abstract = None

    def pack(self):
        self.pit.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)
        self.sep.pack(side=tkinter.LEFT, fill=tkinter.Y)
        
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        self.toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=True)
        
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
        if stove.isLeftClick(event):
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
    def onCanvasRelease(self, event: MouseEvent):
        if self.gui.getFuncButton().bboxButtonPressed():
            anchor.clearBuffer()
    def onCanvasDrag(self, event: MouseEvent):
        if self.gui.getFuncButton().bboxButtonPressed():
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
                b.rect.set_width(width)
                b.rect.set_height(height)
                b.anchorUpdate()
                self.canvas.draw()
    
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
        
        abstract.addToPool(self)
    
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
        if not len(self.__bbox):
            raw = self.gui.getBackEnd().AppIntDINOwrapper(self.__img_np_gs) # list[list]
            for each in raw:
                o = box(each, self.gui)
                self.__bbox.append(o)
        return self.__bbox
    @bbox.setter
    def bbox(self, value: list[box]):
        self.__bbox = value
    
    @property
    def boundingBoxRevised(self) -> list[list]:
        return [b.final for b in self.bbox]
    
    @property
    def drawBbox(self) -> bool:
        return self.__drawBbox
    @drawBbox.setter
    def drawBbox(self, value: bool):
        for b in self.bbox:
            b.draw = value
            if not value:
                box.clearBufferAndDeselect()
        self.thumbnail = "bbox" if value else "default"
        self.gui.getStove().canvas.draw()
        self.__drawBbox = value
    
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
        abstract.setBuffer(self)
        self.gui.getStove().cook(self)
    
    @classmethod
    def sendFirst(cls):
        for target in cls.getPool():
            if target.selected:
                target.on_click(None)
                return
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
        cls.sendFirst()
    @classmethod
    def saveBboxChanges(cls):
        cls.getBuffer().drawBbox = False
        cls.sendFirst()
    @classmethod
    def zip(cls) -> list[bundle]:
        result = []
        for abs in cls.getPool():
            if abs.selected:
                b = bundle()
                b.path = abs.getAbsPath()
                b.bbox = abs.boundingBoxRevised
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
    
    def getImgNumpyGreyscale(self) -> np.ndarray:
        return self.__img_np_gs
    def getImgNumpyRGB(self) -> np.ndarray:
        return self.__img_np_rgb
    def getLabel(self) -> tkinter.Label:
        return self.__label
    def getAbsPath(self) -> pathlib.Path:
        return self.__abs_path

class tifSequence():
    def __init__(self, gui: FishGUI):
        self.gui = gui
        container = gui.getLowerFrame().getFrameB()
        self.base = tkinter.Canvas(container, height=74)

        self.scrollbar = tkinter.Scrollbar(container, orient=tkinter.HORIZONTAL, command=self.base.xview)
        self.base.configure(yscrollcommand=self.scrollbar.set)

        self.gallery_frame = tkinter.Frame(self.base)
        self.base.create_window((0, 0), window=self.gallery_frame, anchor="nw")
        
    def pack(self):
        self.base.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        self.scrollbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    
    def unpack(self):
        self.base.pack_forget()
        self.scrollbar.pack_forget()
    
    def addToGallery(self, tif_files: list[pathlib.Path]):
        for path in tif_files:
            _ = abstract(path, self.gallery_frame, self.gui)
        abstract.sendFirst()
    
    def resetPosition(self):
        self.base.xview_moveto(0)
        self.base.yview_moveto(0)
        
class funcButton():
    def __init__(self, gui: FishGUI):
        self.gui = gui
        container = gui.getLowerFrame().getFrameC()
        self.IMPORT = tkinter.Button(container,
                                     text="Import",
                                     height=2,
                                     relief=tkinter.RAISED,
                                     command=self.IMPORT_call)
        self.__select_toggle = tkinter.IntVar(value=0)
        self.SELECT = tkinter.Checkbutton(container,
                                          text="Select",
                                          height=2,
                                          variable=self.__select_toggle,
                                          onvalue=1,
                                          offvalue=0,
                                          indicatoron=False,
                                          command=self.SELECT_call)
        self.__bbox_toggle = tkinter.IntVar(value=0)
        self.BBOX = tkinter.Checkbutton(container,
                                        text="BBOX",
                                        height=2,
                                        variable=self.__bbox_toggle,
                                        onvalue=1,
                                        offvalue=0,
                                        indicatoron=False,
                                        command=self.BBOX_call)
        self.SEGMENT = tkinter.Button(container,
                                      text="Segment",
                                      height=2,
                                      relief=tkinter.RAISED)
        self.SAVE = tkinter.Button(container,
                                   text="Save",
                                   height=2,
                                   relief=tkinter.RAISED)
    
    def pack(self):
        self.IMPORT.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
        self.SELECT.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
        self.BBOX.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
        self.SEGMENT.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
        self.SAVE.pack(side=tkinter.LEFT, expand=True, fill=tkinter.X)
    
    def unpack(self):
        self.IMPORT.pack_forget()
        self.SELECT.pack_forget()
        self.BBOX.pack_forget()
        self.SEGMENT.pack_forget()
        self.SAVE.pack_forget()
    
    def getButtonWidget(self, name: str):
        return {"IMPORT": self.IMPORT,
                "SELECT": self.SELECT,
                "BBOX": self.BBOX,
                "SEGMENT": self.SEGMENT,
                "SAVE": self.SAVE}[name]
    
    def selectButtonPressed(self) -> bool:
        return self.__select_toggle.get()
    def bboxButtonPressed(self) -> bool:
        return self.__bbox_toggle.get()
    
    def IMPORT_call(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            folder = pathlib.Path(folder_path)
            tif_files = [str(file.resolve()) for file in folder.glob("*.tif")]
            self.gui.getTifSequence().addToGallery(tif_files)
            
    def BBOX_call(self):
        if not self.gui.getStove().isLoaded():
            FishGUI.popBox("w", "Image Not Loaded", "Please select an image first")
        if self.bboxButtonPressed():
            abstract.sendFirst()
        elif not self.bboxButtonPressed():
            abstract.saveBboxChanges()

    def SELECT_call(self):
        if self.selectButtonPressed():
            abstract.selectAll()
        elif not self.selectButtonPressed():
            abstract.removeUnselected()
            self.gui.getTifSequence().resetPosition()
    
    def SEGMENT_call(self):
        if not self.gui.getStove().isLoaded():
            FishGUI.popBox("w", "Image Not Loaded", "Please select an image first")

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
        self.__root.title("Quick Seg")
        self.__root.geometry("1000x800")
        
        self.__be: fishCore.Fish = fishCore.Fish(pathlib.Path("./config.ini"))
        self.__be.set_model_version("3.50")
        
        self.__lf = lf(self)
        self.__stove = stove(self)
        self.__tifSequence = tifSequence(self)
        self.__funcButton = funcButton(self)
        self.__seasoning = seasoning(self)
        
        self.__lf.pack()
        self.__stove.pack()
        self.__tifSequence.pack()
        self.__funcButton.pack()
        self.__seasoning.pack()
    
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
    root = tkinter.Tk()
    app = FishGUI(root)
    root.mainloop()