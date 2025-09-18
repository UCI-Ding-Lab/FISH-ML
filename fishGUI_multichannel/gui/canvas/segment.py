import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from skimage import measure

class segment():
    __buffer: 'segment' = None

    def __init__(self, gui, data: np.ndarray):
        self.__data: np.ndarray = data.T
        self.gui = gui
        self.__patch: PathPatch = None
        self.__draw: bool = False
        self.__exist: bool = True
        self.__selected: bool = False

    @property
    def xy(self):
        xs, ys = np.where(self.__data.T == 1)
        return xs.min(), ys.min()
    
    @property
    def box(self):
        ys, xs = np.where(self.__data.T == 1)
        return self.__data.T[ys.min():ys.max(), xs.min():xs.max()]

    @property
    def patch(self) -> PathPatch:
        if not self.__patch:
            try:
                c = measure.find_contours(self.__data, level=0.5)[0]
                vertices = np.array(c)
                codes = np.full(len(vertices), Path.LINETO)
                codes[0] = Path.MOVETO
                path = Path(vertices, codes)
                self.__patch = PathPatch(path, facecolor='none', edgecolor='orange', linewidth=0.5)
            except IndexError:
                # If no contours found, create empty patch
                vertices = np.array([[0, 0]])
                codes = np.array([Path.MOVETO])
                path = Path(vertices, codes)
                self.__patch = PathPatch(path, facecolor='none', edgecolor='orange', linewidth=0.5)
        return self.__patch

    @property
    def exist(self) -> bool:
        return self.__exist

    @property
    def selected(self) -> bool:
        return self.__selected
    @selected.setter
    def selected(self, value: bool):
        if self.__selected == value:
            return
        self.__selected = value
        self.draw = False
        self.patch.set_edgecolor('cyan' if value else 'orange')
        canvas = self.gui.getStove().canvas
        subplot = self.gui.getStove().subplot
        background = canvas.copy_from_bbox(subplot.bbox)
        canvas.restore_region(background)
        subplot.draw_artist(self.patch)
        canvas.blit(subplot.bbox)
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
                self.gui.getStove().subplot.add_patch(self.patch) # Segment Mode: Adding patch
            else:
                try:
                    self.__patch.remove()
                except (NotImplementedError, ValueError, AttributeError):
                    # If normal removal fails, try manual removal from patches list
                    patches = self.gui.getStove().subplot.patches
                    if self.__patch and self.__patch in patches:
                        patches.remove(self.__patch)
        except Exception as e:
            print(f"Error in segment draw setter: {e}")
        
        self.__draw = value
        self.gui.getStove().canvas.draw()

    def contains(self, x: float, y: float) -> bool:
        p = self.gui.getStove().subplot.transData.transform((x, y))
        return self.patch.contains_point(p)

    def update_mask(self, x, y, radius, erase=False):
        print(f"[DEBUG] update_mask called at ({x}, {y}) with radius {radius}, erase={erase}")
        x_int, y_int = int(x), int(y)
        for i in range(x_int - radius, x_int + radius + 1):
            for j in range(y_int - radius, y_int + radius + 1):
                if (i - x_int)**2 + (j - y_int)**2 <= radius**2:
                    if 0 <= i < self.__data.shape[0] and 0 <= j < self.__data.shape[1]:
                        if erase:
                            self.__data[i, j] = 0
                        else:
                            self.__data[i, j] = 1

    def recal_patch(self):
        print("[DEBUG] recal_patch called")
        try:
            c = measure.find_contours(self.__data, level=0.5)[0]
            vertices = np.array(c)
            codes = np.full(len(vertices), Path.LINETO)
            codes[0] = Path.MOVETO
            self.patch.get_path().vertices = vertices
            self.patch.get_path().codes = codes
        except IndexError:
            # If no contours found, set to empty
            pass

    def delete(self):
        abs = self.gui.getStove().getLoaded()
        if abs and self in abs.segment:
            abs.segment.remove(self)
            self.draw = False
            segment.clearBuffer()
            self.gui.getStove().canvas.flush_events()

    @classmethod
    def setBuffer(cls, segment: 'segment'):
        cls.__buffer = segment
    @classmethod
    def getBuffer(cls) -> 'segment':
        return cls.__buffer
    @classmethod
    def clearBuffer(cls):
        cls.__buffer = None
    @classmethod
    def clearBufferAndDeselect(cls):
        current = cls.getBuffer()
        if current: current.selected = False
        cls.__buffer = None