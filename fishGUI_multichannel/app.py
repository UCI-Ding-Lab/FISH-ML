from tkinter import messagebox
import tkinter as tk
import pathlib
import fishCore
from .gui.thumbnails import tifSequence
from .gui.buttons import funcButton
from .gui.frames import lf
from .gui.tools_pannel import seasoning
from .gui.canvas.box import box
from .gui.canvas.segment import segment
from .gui.canvas.stove import stove

"""
To run the app: python -m fishGUI_multichannel.app
"""

class FishGUI(object):
    def __init__(self, root):
        self.__root: tk.Tk = root
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
        
        self.__waitWindow = None

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
    
    def indicateWait(self, content: str):
        self.__waitWindow = tk.Toplevel(self.__root)
        self.__waitWindow.title("FISH-ML")
        self.__waitWindow.geometry("300x100")
        self.__waitWindow.resizable(False, False)
        tk.Label(self.__waitWindow, text=content+" in progress...").pack()
    def dismissWait(self):
        if self.__waitWindow:
            self.__waitWindow.destroy()
            self.__waitWindow = None

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
    def getRoot(self) -> tk.Tk:
        return self.__root

    # ---- popups / status ----
    def popBox(self, level: str, title: str, msg: str):
        level = (level or "").lower()
        if level.startswith("i"):
            messagebox.showinfo(title, msg, parent=self.__root)
        elif level.startswith("w"):
            messagebox.showwarning(title, msg, parent=self.__root)
        else:
            messagebox.showerror(title, msg, parent=self.__root)

    def onDelete(self, event):
        selected_box = box.getBuffer()
        selected_seg = segment.getBuffer()

        if selected_box:
            selected_box.delete()
        elif selected_seg:
            selected_seg.delete()
        else:
            self.popBox('w', 'No Selection', 'No bounding box or segmentation mask is selected.')

def main():
    root = tk.Tk()
    app = FishGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
