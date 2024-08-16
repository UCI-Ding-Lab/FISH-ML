from __future__ import annotations
import tkinter
import pathlib
import numpy as np
from PIL import (Image, ImageTk, ImageDraw)
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.patches import Rectangle
import fishCore

class seasoning():
    def __init__(self, gui: FishGUI):
        self.gui = gui
        self.toolbank = tkinter.Frame(self.gui.getLowerFrame().getFrameA(), width=250, background="grey")
        self.button1 = tkinter.Button(self.toolbank, height=2, text="FuncA")
        self.button2 = tkinter.Button(self.toolbank, height=2, text="FuncB")
        self.button3 = tkinter.Button(self.toolbank, height=2, text="FuncC")
        
    
    def pack(self):
        self.toolbank.pack(side=tkinter.RIGHT, expand=True, fill=tkinter.BOTH)
        self.button1.pack(side=tkinter.TOP, fill=tkinter.X)
        self.button2.pack(side=tkinter.TOP, fill=tkinter.X)
        self.button3.pack(side=tkinter.TOP, fill=tkinter.X)

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
    
    def cookBbox(self, bboxes: list[list]):
        for bbox in bboxes:
            min_x, min_y, max_x, max_y = bbox
            rect = Rectangle((min_x, min_y),
                             max_x - min_x,
                             max_y - min_y,
                             linewidth=float(self.gui.getBackEnd().config["info"]["bbox_preview_line_width"]),
                             edgecolor='r',facecolor='none')
            self.subplot.add_patch(rect)
        self.canvas.draw()
    
    def dump(self):
        self.clearLoaded()
        self.subplot.clear()
        self.subplot.set_axis_off()
        self.canvas.draw()
    
    def isLoaded(self) -> bool:
        return self.__onLoad is not None
    def getLoaded(self) -> abstract:
        return self.__onLoad
    def setLoaded(self, abs: abstract):
        self.__onLoad = abs
    def clearLoaded(self):
        self.__onLoad = None

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
        self.__bbox: list[list] = None
        self.__highlighted: str = None
        
        abstract.addToPool(self)
    
    @property
    def bboxThumbnail(self) -> ImageTk.PhotoImage:
        if not self.__img_pil_thumbnail_bbox:
            self.__img_pil_thumbnail_bbox = self.__img_pil_thumbnail.copy()
            ImageDraw.Draw(self.__img_pil_thumbnail_bbox).ellipse((49, 5, 59, 15), fill=(0,255,0))
        self.__img_pil_thumbnail_bbox = ImageTk.PhotoImage(self.__img_pil_thumbnail_bbox)
        return self.__img_pil_thumbnail_bbox
    
    @property
    def bbox(self) -> list[list]:
        if self.__bbox is None:
            self.bbox = self.gui.getBackEnd().AppIntDINOwrapper(self.__img_np_gs)
        return self.__bbox
    @bbox.setter
    def bbox(self, bboxes: list[list]):
        self.__bbox = bboxes
        self.getLabel().config(image=self.bboxThumbnail)
    
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
        self.highlighted = "red"
        b = abstract.getBuffer()
        if b: del b.highlighted
        abstract.setBuffer(self)
        self.gui.getStove().cook(self)
    
    @classmethod
    def sendFirst(cls):
        target: abstract = cls.getPool()[0]
        target.on_click(None)
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
    
    def getImgNumpyGreyscale(self) -> np.ndarray:
        return self.__img_np_gs
    def getImgNumpyRGB(self) -> np.ndarray:
        return self.__img_np_rgb
    def getLabel(self) -> tkinter.Label:
        return self.__label

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
        
class funcButton():
    def __init__(self, gui: FishGUI):
        self.gui = gui
        container = gui.getLowerFrame().getFrameC()
        self.IMPORT = tkinter.Button(container, text="Import", height=2, command=self.IMPORT_call)
        self.SELECT = tkinter.Button(container, text="Select", height=2)
        self.BBOX = tkinter.Button(container, text="BBox", height=2, command=self.BBOX_call)
        self.SEGMENT = tkinter.Button(container, text="Segment", height=2)
        self.SAVE = tkinter.Button(container, text="Save", height=2)
    
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
    
    def IMPORT_call(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            folder = pathlib.Path(folder_path)
            tif_files = [str(file.resolve()) for file in folder.glob("*.tif")]
            self.gui.getTifSequence().addToGallery(tif_files)
            
    def BBOX_call(self):
        if not self.gui.getStove().isLoaded():
            FishGUI.popBox("w", "Image Not Loaded", "Please select an image first")
        l = self.gui.getStove().getLoaded()
        self.gui.getStove().cookBbox(l.bbox)
        

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