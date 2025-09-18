from matplotlib.patches import Circle

class anchor():
    __buffer: 'anchor' = None
    def __init__(self, x: float, y: float, gui, master, location: str):
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
        if self.__draw == value: 
            return
        try:
            if value:
                if hasattr(self.gui.getStove().subplot, 'patches') and self.patch not in self.gui.getStove().subplot.patches:
                    self.gui.getStove().subplot.add_patch(self.patch)
            else:
                if hasattr(self.gui.getStove().subplot, 'patches'):
                    try:
                        self.patch.remove()
                    except (NotImplementedError, ValueError):
                        patches = self.gui.getStove().subplot.patches
                        if self.patch in patches:
                            patches.remove(self.patch)
        except Exception as e:
            print(f"Error in anchor draw setter: {e}")
        
        self.__draw = value
        try:
            self.gui.getStove().canvas.draw()
        except Exception as e:
            print(f"Error drawing canvas: {e}")

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
    def setBuffer(cls, anchor: 'anchor'):
        cls.__buffer = anchor
    @classmethod
    def getBuffer(cls) -> 'anchor':
        return cls.__buffer
    @classmethod
    def clearBuffer(cls):
        if cls.getBuffer(): cls.getBuffer().selected = False
        cls.setBuffer(None)