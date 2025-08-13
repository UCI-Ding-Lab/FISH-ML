# fishgui/gui/frames.py
import tkinter as tk

class lf:
    def __init__(self, gui):
        self.__a = tk.Frame(gui.getRoot())
        self.__s = tk.Frame(gui.getRoot(), height=1, bd=0, relief=tk.SUNKEN, bg="black")
        self.__b = tk.Frame(gui.getRoot(), padx=5, pady=5)
        self.__c = tk.Frame(gui.getRoot(), padx=5, pady=5)

    def pack(self):
        self.__a.pack(expand=True, fill=tk.ALL)
        self.__s.pack(fill=tk.X)
        self.__b.pack(fill=tk.X)
        self.__c.pack(fill=tk.X)

    def unpack(self):
        self.__a.pack_forget()
        self.__b.pack_forget()
        self.__c.pack_forget()

    def getFrameA(self): return self.__a
    def getFrameB(self): return self.__b
    def getFrameC(self): return self.__c
