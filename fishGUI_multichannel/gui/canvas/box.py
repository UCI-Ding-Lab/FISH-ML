from matplotlib.patches import Rectangle, Circle
from .anchor import anchor

class box():
    __buffer: 'box' = None
    def __init__(self, bbox: list, gui):
        self.gui = gui
        self.__bbox = bbox
        min_x, min_y, max_x, max_y = self.__bbox
        self.__rect = Rectangle((min_x, min_y),
                                max_x - min_x,
                                max_y - min_y,
                                linewidth=float(self.gui.getBackEnd().config["info"]["bbox_preview_line_width"]),
                                edgecolor='r',
                                facecolor='none')
        self.__center = Circle(self.__rect.get_center(), radius=5, color='lime', fill=True)
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
    def center(self) -> Circle:
        return self.__center
    @center.setter
    def center(self, c) -> Circle:
        self.__center = Circle(c, radius=5, color='lime', fill=True)
    @property
    def anchors(self) -> dict[str, anchor]:
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
        canvas.blit(subplot.bbox)
        self.draw = True
    
    def delete(self):
        abs = self.gui.getStove().getLoaded()
        if abs and self in abs.bbox:
            abs.bbox.remove(self)
            self.draw = False
            for anchor in self.anchors.values():
                anchor.draw = False
            box.clearBuffer()
            self.gui.getStove().canvas.flush_events()
            
    @property
    def draw(self):
        return self.__draw
    @draw.setter
    def draw(self, value: bool):
        if self.__draw == value: return
        try:
            if value:
                if hasattr(self.gui.getStove().subplot, 'patches') and self.rect not in self.gui.getStove().subplot.patches:
                    self.gui.getStove().subplot.add_patch(self.rect)
                    nuc_center = Circle(self.rect.get_center(), radius=5, color='lime', fill=True)
                    self.gui.getStove().subplot.add_patch(self.center)
            else:
                if hasattr(self.gui.getStove().subplot, 'patches'):
                    try:
                        self.rect.remove()
                        self.center.remove() # Removes circles when BBOX mode is exited
                    except (NotImplementedError, ValueError):
                        patches = self.gui.getStove().subplot.patches
                        if self.rect in patches:
                            # print(self.rect, type(self.rect))
                            # print(self.center)
                            patches.remove(self.rect)
        except Exception as e:
            print(f"Error in box draw setter: {e}")
        
        self.__draw = value
        try:
            self.gui.getStove().canvas.draw()
        except Exception as e:
            print(f"Error drawing canvas: {e}")

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
    def removeCenter(cls, gui, center: Circle):
        patches = list(gui.getStove().subplot.patches)
        if center in patches:
            patches.remove(center)

        try:
            center.remove()
            print("Previous nucleus center is removed")
        except NotImplementedError as e:
            print("Previous circle has already been removed", e)
   
    @classmethod
    def setBuffer(cls, box: 'box'):
        cls.__buffer = box
    @classmethod
    def getBuffer(cls) -> 'box':
        return cls.__buffer
    @classmethod
    def clearBuffer(cls):
        cls.__buffer = None
    @classmethod
    def clearBufferAndDeselect(cls):
        current = cls.getBuffer()
        if current: current.selected = False
        cls.__buffer = None