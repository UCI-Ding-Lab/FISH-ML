import tkinter
import numpy as np
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.patches import Circle, Rectangle, PathPatch
from matplotlib.backend_bases import MouseEvent
from matplotlib.path import Path
from skimage import measure

class FishToolBar(NavigationToolbar2Tk):
    def __init__(self, canvas, window, gui):
        super().__init__(canvas, window)
        self.fishGUI = gui
    
    def resetToolBank(self):
        self.fishGUI.getSeasoning().tools_var["brush"].set(0)
        self.fishGUI.getSeasoning().tools_var["eraser"].set(0)
    
    def home(self):
        self.resetToolBank()
        super().home()
    def zoom(self, *args):
        self.resetToolBank()
        super().zoom(*args)
    
    def deactivate_all_tools(self):
        if self.mode:
            self.mode = ""
            self.set_message("")
            self._update_buttons_checked()

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
                self.gui.getStove().subplot.add_patch(self.patch)
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
            else:
                if hasattr(self.gui.getStove().subplot, 'patches'):
                    try:
                        self.rect.remove()
                    except (NotImplementedError, ValueError):
                        patches = self.gui.getStove().subplot.patches
                        if self.rect in patches:
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

class stove():
    BILT_BUFFER1 = None
    BILT_BUFFER2 = None
    BILT_BUFFER3 = None
    
    def __init__(self, gui):
        self.gui = gui
        self.pit = tkinter.Frame(self.gui.getLowerFrame().getFrameA(), background="black")
        self.sep = tkinter.Frame(self.gui.getLowerFrame().getFrameA(), width=1, bd=0, relief=tkinter.SUNKEN, bg="black")
        
        self.ax_img = None
        self.figure = Figure(figsize=(3,3), dpi=200)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.subplot = self.figure.add_subplot(111)
        self.subplot.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.figure, self.pit)
        self.canvas.mpl_connect("button_press_event", self.onCanvasClick)
        self.canvas.mpl_connect("button_release_event", self.onCanvasRelease)
        self.canvas.mpl_connect("motion_notify_event", self.onCanvasDrag)
        self.toolbar = FishToolBar(self.canvas, self.pit, self.gui)
        self.toolbar.update()
        self.tb_pointer = Circle((0, 0), 15, linewidth=0.5, edgecolor='cyan', facecolor='none')
        self.xs = []
        self.ys = []
        self.markers: list[Circle] = []
        self.press = False
        
        self.__onLoad = None

    @property
    def biltbg(self):
        return self.canvas.copy_from_bbox(self.subplot.bbox)
    
    def bufferSetCurrent(self, buffer):
        if buffer == 1:
            self.BILT_BUFFER1 = self.biltbg
        elif buffer == 2:
            self.BILT_BUFFER2 = self.biltbg
        elif buffer == 3:
            self.BILT_BUFFER3 = self.biltbg

    def pack(self):
        self.pit.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)
        self.sep.pack(side=tkinter.LEFT, fill=tkinter.Y)
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        self.toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH)
        
    def cook(self, abs):
        self.setLoaded(abs)
        self.ax_img = self.subplot.imshow(self.getLoaded().getImgNumpyRGB())
        self.subplot.set_axis_off()
        self.canvas.draw()

    def dump(self):
        self.clearLoaded()
        # Clear all patches safely
        try:
            if hasattr(self.subplot, 'patches'):
                # Create a copy of the patches list to avoid modification during iteration
                patches_to_remove = list(self.subplot.patches)
                for patch in patches_to_remove:
                    try:
                        patch.remove()
                    except (NotImplementedError, ValueError, AttributeError):
                        # If removal fails, just continue
                        pass
                # Clear the patches list completely
                self.subplot.patches.clear()
        except AttributeError:
            pass
        
        self.subplot.clear()
        self.subplot.set_axis_off()
        self.canvas.draw()
    
    def adjust_contrast(self, factor):
        img = self.getLoaded().getImgNumpyRGB()
        img = img.astype(np.float32) / 255.0
        img = np.clip(0.5 + factor * (img - 0.5), 0, 1)
        img = (img * 255).astype(np.uint8)
        self.ax_img.set_data(img)
        self.canvas.draw()
    
    def adjust_brightness(self, factor):
        img = self.getLoaded().getImgNumpyRGB()
        img = img.astype(np.float32) / 255.0
        img = np.clip(factor + img, 0, 1)
        img = (img * 255).astype(np.uint8)
        self.ax_img.set_data(img)
        self.canvas.draw()
        
    def onCanvasClick(self, event: MouseEvent):
        self.press = True
        if stove.isLeftClick(event):
            if event.inaxes != self.subplot:
                return
                
            # Handle BBOX mode interactions
            if self.gui.getFuncButton().bboxButtonPressed():
                if box.getBuffer() and box.getBuffer().selected:
                    anchorName = box.getBuffer().anchorContains(event.xdata, event.ydata)
                    if anchorName:
                        target = box.getBuffer().anchors[anchorName]
                        target.selected = True
                        anchor.setBuffer(target)
                        return
                target = self.getLoaded().findBoxFromPoint(event.xdata, event.ydata)
                box.clearBufferAndDeselect()
                if target:
                    self.toolbar.deactivate_all_tools()
                    target.selected = True
                    box.setBuffer(target)
                    return  # Exit early if we found a box
                    
            # Handle SEGMENT mode interactions (only if not handled by bbox above)
            if self.gui.getFuncButton().segButtonPressed():
                # Handle brush/eraser tools
                if self.gui.getSeasoning().burshButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
                    self.xs = [event.xdata]
                    self.ys = [event.ydata]
                    self.bufferSetCurrent(1)
                    self.bufferSetCurrent(2)
                    self.canvas.restore_region(self.BILT_BUFFER1)
                    self.marker_draw(event.xdata, event.ydata)
                    self.canvas.blit(self.subplot.bbox)
                elif self.gui.getSeasoning().eraserButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
                    self.xs = [event.xdata]
                    self.ys = [event.ydata]
                    self.bufferSetCurrent(1)
                    self.bufferSetCurrent(2)
                    self.canvas.restore_region(self.BILT_BUFFER1)
                    self.marker_draw(event.xdata, event.ydata)
                    self.canvas.blit(self.subplot.bbox)
                else:
                    # Select segment if not using brush/eraser
                    target = self.getLoaded().findSegFromPoint(event.xdata, event.ydata)
                    segment.clearBufferAndDeselect()
                    if target:
                        target.selected = True
                        segment.setBuffer(target)

    def onCanvasRelease(self, event: MouseEvent):
        self.press = False
        if self.gui.getFuncButton().bboxButtonPressed():
            anchor.clearBuffer()
        elif self.gui.getFuncButton().segButtonPressed():
            if self.gui.getSeasoning().burshButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
                final = list(zip(self.xs, self.ys))
                for marker in self.markers:
                    marker.remove()
                self.markers.clear()
                for x, y in final:
                    segment.getBuffer().update_mask(x, y, self.gui.getSeasoning().get_marker_size())
                segment.getBuffer().recal_patch()
                self.canvas.draw_idle()
                self.xs.clear()
                self.ys.clear()
            elif self.gui.getSeasoning().eraserButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
                final = list(zip(self.xs, self.ys))
                for marker in self.markers:
                    marker.remove()
                self.markers.clear()
                for x, y in final:
                    segment.getBuffer().update_mask(x, y, self.gui.getSeasoning().get_marker_size(), erase=True)
                segment.getBuffer().recal_patch()
                self.canvas.draw_idle()
                self.xs.clear()
                self.ys.clear()

    def onCanvasDrag(self, event: MouseEvent):
        if not self.press: 
            return

        if self.gui.getFuncButton().bboxButtonPressed():
            a = anchor.getBuffer()
            b = box.getBuffer()
            if not (a and a.selected and b and b.selected): 
                return
            if event.inaxes != b.rect.axes: 
                return
            
            x0, y0 = b.rect.get_xy()
            w, h = b.rect.get_width(), b.rect.get_height()
            
            if a.location == "bottom-left":
                nx, ny = event.xdata, event.ydata
                w += x0 - nx
                h += y0 - ny
                b.rect.set_xy((nx, ny))
            elif a.location == "bottom-right":
                w = event.xdata - x0
                h += y0 - event.ydata
                b.rect.set_xy((x0, event.ydata))
            elif a.location == "top-right":
                w = event.xdata - x0
                h = event.ydata - y0
            elif a.location == "top-left":
                nx = event.xdata
                h = event.ydata - y0
                w += x0 - nx
                b.rect.set_xy((nx, y0))
            elif a.location == "pos-anchor":
                dx = event.xdata - (b.rect.get_x() + w/2)
                dy = event.ydata - (b.rect.get_y() + h)
                b.rect.set_xy((x0 + dx, y0 + dy))
            
            b.rect.set_width(w)
            b.rect.set_height(h)
            b.anchorUpdate()
            self.canvas.draw()

        elif self.gui.getFuncButton().segButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
            current_x, current_y = event.xdata, event.ydata
            if current_x is None or current_y is None: 
                return

            if self.gui.getSeasoning().burshButtonPressed() or self.gui.getSeasoning().eraserButtonPressed():
                if self.xs and self.ys:
                    lx, ly = self.xs[-1], self.ys[-1]
                    distance = np.hypot(current_x - lx, current_y - ly)
                    if distance > 5:
                        n = int(distance // 1)
                        x_vals = np.linspace(lx, current_x, n + 1)
                        y_vals = np.linspace(ly, current_y, n + 1)
                        for x, y in zip(x_vals, y_vals):
                            self.xs.append(x)
                            self.ys.append(y)
                            self.bufferSetCurrent(1)
                            self.canvas.restore_region(self.BILT_BUFFER1)
                            self.marker_draw(x, y)
                            self.canvas.blit(self.subplot.bbox)
                else:
                    self.xs.append(current_x)
                    self.ys.append(current_y)
                    self.bufferSetCurrent(1)
                    self.canvas.restore_region(self.BILT_BUFFER1)
                    self.marker_draw(current_x, current_y)
                    self.canvas.blit(self.subplot.bbox)

                segment.getBuffer().update_mask(
                    current_x, current_y, self.gui.getSeasoning().get_marker_size(),
                    erase=self.gui.getSeasoning().eraserButtonPressed()
                )

    def marker_draw(self, x, y):
        circle = Circle((x, y), self.gui.getSeasoning().get_marker_size(), color='red', alpha=0.01)
        self.markers.append(circle)
        self.subplot.add_patch(circle)
        self.subplot.draw_artist(circle)

    def isLoaded(self) -> bool:
        return self.__onLoad is not None
    def getLoaded(self):
        return self.__onLoad
    def setLoaded(self, abs):
        self.__onLoad = abs
    def clearLoaded(self):
        self.__onLoad = None
    
    @staticmethod
    def isLeftClick(event: MouseEvent) -> bool:
        return event.button == 1
    @staticmethod
    def isLeftClick(event: MouseEvent) -> bool:
        return event.button == 1
