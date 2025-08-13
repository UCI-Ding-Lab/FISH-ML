# fishgui/model/segment.py
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from skimage import measure

class segment:
    __buffer = None

    def __init__(self, gui, data):
        self._data = np.asarray(data).T      # match axes
        self.gui = gui
        self._patch = None
        self._draw = False
        self._selected = False

    # ----- geometry helpers -----
    @property
    def xy(self):
        xs, ys = np.where(self._data.T == 1)
        return (int(xs.min()), int(ys.min())) if xs.size and ys.size else (0, 0)

    @property
    def box(self):
        ys, xs = np.where(self._data.T == 1)
        if xs.size == 0 or ys.size == 0:
            return np.zeros((1, 1), dtype=self._data.dtype)
        return self._data.T[ys.min():ys.max(), xs.min():xs.max()]

    # ----- path / patch -----
    @property
    def patch(self):
        if self._patch is None:
            c = measure.find_contours(self._data, level=0.5)[0]
            v = np.array(c)
            codes = np.full(len(v), Path.LINETO); codes[0] = Path.MOVETO
            self._patch = PathPatch(Path(v, codes), facecolor='none', edgecolor='orange', linewidth=0.5)
        return self._patch

    def recal_patch(self):
        c = measure.find_contours(self._data, level=0.5)[0]
        v = np.array(c)
        codes = np.full(len(v), Path.LINETO); codes[0] = Path.MOVETO
        self.patch.get_path().vertices = v
        self.patch.get_path().codes = codes

    # ----- selection / draw -----
    @property
    def selected(self): return self._selected
    @selected.setter
    def selected(self, value):
        if self._selected == value: return
        self._selected = bool(value)
        self.draw = False
        self.patch.set_edgecolor('cyan' if self._selected else 'orange')
        canvas = self.gui.getStove().canvas
        ax = self.gui.getStove().subplot
        bg = canvas.copy_from_bbox(ax.bbox)
        canvas.restore_region(bg)
        ax.draw_artist(self.patch)
        canvas.blit(ax.bbox)
        self.draw = True

    @property
    def draw(self): return self._draw
    @draw.setter
    def draw(self, value):
        if self._draw == value: return
        ax = self.gui.getStove().subplot
        (ax.add_patch(self.patch) if value else self.patch.remove())
        self.gui.getStove().canvas.draw()
        self._draw = bool(value)

    # ----- hit test / edit -----
    def contains(self, x, y):
        p = self.gui.getStove().subplot.transData.transform((x, y))
        return self.patch.contains_point(p)

    def update_mask(self, x, y, radius, erase=False):
        xi, yi = int(x), int(y)
        for i in range(xi - radius, xi + radius + 1):
            for j in range(yi - radius, yi + radius + 1):
                if (i - xi)**2 + (j - yi)**2 <= radius**2:
                    if 0 <= i < self._data.shape[0] and 0 <= j < self._data.shape[1]:
                        self._data[i, j] = 0 if erase else 1

    def delete(self):
        abs_obj = self.gui.getStove().getLoaded()
        if abs_obj and self in abs_obj.segment:
            abs_obj.segment.remove(self)
            self.draw = False
            segment.clearBuffer()
            self.gui.getStove().canvas.flush_events()

    # ----- class buffer helpers -----
    @classmethod
    def setBuffer(cls, seg): cls.__buffer = seg
    @classmethod
    def getBuffer(cls): return cls.__buffer
    @classmethod
    def clearBuffer(cls): cls.__buffer = None
    @classmethod
    def clearBufferAndDeselect(cls):
        cur = cls.getBuffer()
        if cur: cur.selected = False
        cls.__buffer = None
