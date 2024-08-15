import tkinter
import pathlib
import numpy as np
from PIL import (Image, ImageTk, ImageEnhance)
from tkinter import filedialog
import matplotlib
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

class seasoning():
    def __init__(self, root):
        self.toolbank = tkinter.Frame(root, width=250, background="grey")
        self.button1 = tkinter.Button(self.toolbank, height=2, text="FuncA")
        self.button2 = tkinter.Button(self.toolbank, height=2, text="FuncB")
        self.button3 = tkinter.Button(self.toolbank, height=2, text="FuncC")
        
    
    def pack(self):
        self.toolbank.pack(side=tkinter.RIGHT, expand=True, fill=tkinter.BOTH)
        self.button1.pack(side=tkinter.TOP, fill=tkinter.X)
        self.button2.pack(side=tkinter.TOP, fill=tkinter.X)
        self.button3.pack(side=tkinter.TOP, fill=tkinter.X)

class stove():
    def __init__(self, root):
        self.pit = tkinter.Frame(root, background="black")
    
        self.figure = Figure(figsize=(3,3), dpi=200)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.subplot = self.figure.add_subplot(111)
        self.subplot.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.figure, self.pit)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.pit)
        self.toolbar.update()
    
    def pack(self):
        self.pit.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)
        
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        self.toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=True)
        
    def cook(self, img: np.ndarray):
        self.subplot.imshow(img)
        self.subplot.set_axis_off()
        self.canvas.draw()
    
    def dump(self):
        self.subplot.clear()
        self.subplot.set_axis_off()
        self.canvas.draw()

class abstract():
    def __init__(self, path: pathlib.Path, gallery_frame, stove: stove):
        self.abs_path = path
        self.stove = stove
        self.img_np_gs = np.array(Image.open(path))
        self.img_np_rgb = self.grayscale_to_rgb(self.img_np_gs)
        self.img_tk_rgb = ImageTk.PhotoImage(Image.fromarray(self.img_np_rgb).resize((64, 64)))
        self.label = tkinter.Label(gallery_frame, image=self.img_tk_rgb, width=64, height=64)
        self.label.pack(side=tkinter.LEFT, padx=2, pady=2)
        self.label.bind("<Button-1>", self.on_click)
        
        self.state = None
    
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
    
    def on_click(self, event):
        self.stove.dump()
        self.stove.cook(self.img_np_rgb)

class tifSequence():
    def __init__(self, root, stove: stove):
        self.stove = stove
        self.base = tkinter.Canvas(root, height=74)

        self.scrollbar = tkinter.Scrollbar(root, orient=tkinter.HORIZONTAL, command=self.base.xview)
        self.base.configure(yscrollcommand=self.scrollbar.set)

        self.gallery_frame = tkinter.Frame(self.base)
        self.base.create_window((0, 0), window=self.gallery_frame, anchor="nw")
        
        self.save = []
    
    def pack(self):
        self.base.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        self.scrollbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    
    def unpack(self):
        self.base.pack_forget()
        self.scrollbar.pack_forget()
    
    def addToGallery(self, tif_files: list[pathlib.Path]):
        for path in tif_files:
            abs = abstract(path, self.gallery_frame, self.stove)
            self.save.append(abs)
            
        
        
class funcButton():
    def __init__(self, root, height: int, tifSequence: tifSequence, stove: stove):
        self.tifSequence = tifSequence
        self.stove = stove
        self.IMPORT = tkinter.Button(root, text="Import", height=height, command=self.IMPORT_call)
        self.SELECT = tkinter.Button(root, text="Select", height=height)
        self.BBOX = tkinter.Button(root, text="BBox", height=height)
        self.SEGMENT = tkinter.Button(root, text="Segment", height=height)
        self.SAVE = tkinter.Button(root, text="Save", height=height)
    
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
            self.tifSequence.addToGallery(tif_files)
            
        

class lf():
    def __init__(self, root):
        self.a = tkinter.Frame(root, background="red")
        self.b = tkinter.Frame(root, background="blue",padx=5, pady=5)
        self.c = tkinter.Frame(root, background="green", padx=5, pady=5)
    
    def pack(self):
        self.a.pack(expand=True, fill=tkinter.BOTH)
        self.b.pack(fill=tkinter.X)
        self.c.pack(fill=tkinter.X)
    
    def unpack(self):
        self.a.pack_forget()
        self.b.pack_forget()
        self.c.pack_forget()

class FishGUI(object):
    def __init__(self, root):
        self.root: tkinter.Tk = root
        self.root.title("Quick Seg")
        self.root.geometry("1000x800")
        
        self.lf = lf(self.root)
        self.lf.pack()
        self.stove = stove(self.lf.a)
        self.stove.pack()
        self.tifSequence = tifSequence(self.lf.b, self.stove)
        self.tifSequence.pack()
        self.funcButton = funcButton(self.lf.c, 2, self.tifSequence, self.stove)
        self.funcButton.pack()
        self.seasoning = seasoning(self.lf.a)
        self.seasoning.pack()
        
        
    
        
if __name__ == "__main__":
    root = tkinter.Tk()
    app = FishGUI(root)
    root.mainloop()