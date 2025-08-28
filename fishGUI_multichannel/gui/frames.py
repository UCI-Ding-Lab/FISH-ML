import tkinter

class lf():
    def __init__(self, gui):
        self.__a = tkinter.Frame(gui.getRoot(), height=100)
        self.__a.pack_propagate(False)
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
