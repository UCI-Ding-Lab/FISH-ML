# fishgui/model/shapes.py
from matplotlib.patches import Rectangle, Circle

class anchor(object):
    __buffer = None

    def __init__(self, x, y, gui, master, location):
        self.gui = gui
        self._loc = location
        self._selected = False
        self._draw = False
        self._patch = Circle((x, y), radius=10, linewidth=0.5, edgecolor='cyan', facecolor='none')

    @property
    def patch(self):
        return self._patch

    @property
    def location(self):
        return self._loc

    @property
    def draw(self):
        return self._draw
    @draw.setter
    def draw(self, value):
        if self._draw == value: return
        ax = self.gui.getStove().subplot
        (ax.add_patch(self.patch) if value else self.patch.remove())
        self.gui.getStove().canvas.draw()
        self._draw = value

    @property
    def selected(self):
        return self._selected
    @selected.setter
    def selected(self, v):
        if self._selected == v: return
        self._selected = v
        self.patch.set_edgecolor('b' if v else 'cyan')
        # redraw handled by box when it toggles anchors

    def contains(self, x, y):
        p = self.gui.getStove().subplot.transData.transform((x, y))
        return self.patch.contains_point(p)

    # class buffer
    @classmethod
    def setBuffer(cls, a): cls.__buffer = a
    @classmethod
    def getBuffer(cls): return cls.__buffer
    @classmethod
    def clearBuffer(cls):
        if cls.__buffer: cls.__buffer.selected = False
        cls.__buffer = None


class box(object):
    __buffer = None

    def __init__(self, bbox, gui):
        self.gui = gui
        x0, y0, x1, y1 = bbox
        self._rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                               linewidth=float(self.gui.getBackEnd().config["info"]["bbox_preview_line_width"]),
                               edgecolor='r', facecolor='none')
        self._anchors = {
            "bottom-left": anchor(x0, y0, gui, self, "bottom-left"),
            "bottom-right": anchor(x1, y0, gui, self, "bottom-right"),
            "top-left": anchor(x0, y1, gui, self, "top-left"),
            "top-right": anchor(x1, y1, gui, self, "top-right"),
            "pos-anchor": anchor((x1 - x0)//2 + x0, y1, gui, self, "pos-anchor"),
        }
        self._draw = False
        self._selected = False

    @property
    def rect(self):
        return self._rect

    @property
    def anchors(self):
        return self._anchors

    @property
    def final(self):
        return [self.rect.get_x(),
                self.rect.get_y(),
                self.rect.get_x() + self.rect.get_width(),
                self.rect.get_y() + self.rect.get_height()]

    @property
    def selected(self):
        return self._selected
    @selected.setter
    def selected(self, value):
        if self._selected == value: return
        self._selected = value
        self.draw = False
        self.rect.set_edgecolor('cyan' if value else 'r')
        for a in self._anchors.values():
            a.draw = value
        canvas = self.gui.getStove().canvas
        ax = self.gui.getStove().subplot
        bg = canvas.copy_from_bbox(ax.bbox)
        canvas.restore_region(bg)
        ax.draw_artist(self.rect)
        if value:
            for a in self._anchors.values():
                ax.draw_artist(a.patch)
        canvas.blit(ax.bbox)
        self.draw = True

    @property
    def draw(self):
        return self._draw
    @draw.setter
    def draw(self, value):
        if self._draw == value: return
        ax = self.gui.getStove().subplot
        (ax.add_patch(self.rect) if value else self.rect.remove())
        self.gui.getStove().canvas.draw()
        self._draw = value

    def contains(self, x, y):
        p = self.gui.getStove().subplot.transData.transform((x, y))
        return self.rect.contains_point(p)

    def anchorContains(self, x, y):
        for k, v in self.anchors.items():
            if v.contains(x, y):
                return k
        return None

    def anchorUpdate(self):
        r = self.rect
        self.anchors["bottom-left"].patch.set_center((r.get_x(), r.get_y()))
        self.anchors["bottom-right"].patch.set_center((r.get_x()+r.get_width(), r.get_y()))
        self.anchors["top-left"].patch.set_center((r.get_x(), r.get_y()+r.get_height()))
        self.anchors["top-right"].patch.set_center((r.get_x()+r.get_width(), r.get_y()+r.get_height()))
        self.anchors["pos-anchor"].patch.set_center((r.get_x()+r.get_width()/2, r.get_y()+r.get_height()))

    # class buffer
    @classmethod
    def setBuffer(cls, b): cls.__buffer = b
    @classmethod
    def getBuffer(cls): return cls.__buffer
    @classmethod
    def clearBuffer(cls): cls.__buffer = None
    @classmethod
    def clearBufferAndDeselect(cls):
        cur = cls.getBuffer()
        if cur: cur.selected = False
        cls.__buffer = None

    def delete(self):
        abs_obj = self.gui.getStove().getLoaded()
        if abs_obj and self in abs_obj.bbox:
            abs_obj.bbox.remove(self)
            self.draw = False
            for a in self.anchors.values():
                a.draw = False
            box.clearBuffer()
            self.gui.getStove().canvas.flush_events()
