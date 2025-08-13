# fishgui/gui/canvas_view.py
import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
from matplotlib.backend_bases import MouseEvent
from .toolbar import FishToolBar
from ..model.segment import segment
from ..model.shapes import box, anchor

class stove:
    BILT_BUFFER1 = None
    BILT_BUFFER2 = None
    BILT_BUFFER3 = None

    def __init__(self, gui):
        self.gui = gui
        self.pit = tk.Frame(self.gui.getLowerFrame().getFrameA(), background="black")
        self.sep = tk.Frame(self.gui.getLowerFrame().getFrameA(), width=1, bd=0, relief=tk.SUNKEN, bg="black")

        self.ax_img = None
        self.figure = Figure(figsize=(3, 3), dpi=200)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.subplot = self.figure.add_subplot(111); self.subplot.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.figure, self.pit)
        self.canvas.mpl_connect("button_press_event", self.onCanvasClick)
        self.canvas.mpl_connect("button_release_event", self.onCanvasRelease)
        self.canvas.mpl_connect("motion_notify_event", self.onCanvasDrag)
        self.toolbar = FishToolBar(self.canvas, self.pit, self.gui); self.toolbar.update()

        self.tb_pointer = Circle((0, 0), 15, linewidth=0.5, edgecolor='cyan', facecolor='none')
        self.xs, self.ys = [], []
        self.markers = []
        self.press = False
        self.__onLoad = None

    def get_tb_pointer(self):
        self.tb_pointer.set_radius(self.gui.getSeasoning().get_marker_size())
        return self.tb_pointer

    @property
    def biltbg(self):
        return self.canvas.copy_from_bbox(self.subplot.bbox)

    def bufferSetCurrent(self, buffer_id):
        if buffer_id == 1: self.BILT_BUFFER1 = self.biltbg
        elif buffer_id == 2: self.BILT_BUFFER2 = self.biltbg
        elif buffer_id == 3: self.BILT_BUFFER3 = self.biltbg

    def pack(self):
        self.pit.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sep.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.BOTH)

    def cook(self, abs_obj):
        self.setLoaded(abs_obj)
        self.ax_img = self.subplot.imshow(self.getLoaded().getImgNumpyRGB())
        self.subplot.set_axis_off()
        self.canvas.draw()

    def dump(self):
        self.clearLoaded(); self.subplot.clear(); self.subplot.set_axis_off(); self.canvas.draw()

    def adjust_contrast(self, factor):
        img = self.getLoaded().getImgNumpyRGB().astype(np.float32) / 255.0
        img = np.clip(0.5 + factor * (img - 0.5), 0, 1)
        self.ax_img.set_data((img * 255).astype(np.uint8)); self.canvas.draw()

    def adjust_brightness(self, factor):
        img = self.getLoaded().getImgNumpyRGB().astype(np.float32) / 255.0
        img = np.clip(factor + img, 0, 1)
        self.ax_img.set_data((img * 255).astype(np.uint8)); self.canvas.draw()

    def onCanvasClick(self, event: MouseEvent):
        self.press = True
        if not stove.isLeftClick(event): return
        if event.inaxes != self.subplot: return

        if self.gui.getFuncButton().bboxButtonPressed():
            if box.getBuffer() and box.getBuffer().selected:
                name = box.getBuffer().anchorContains(event.xdata, event.ydata)
                if name:
                    target = box.getBuffer().anchors[name]; target.selected = True; anchor.setBuffer(target); return
            target = self.getLoaded().findBoxFromPoint(event.xdata, event.ydata)
            box.clearBufferAndDeselect()
            if target:
                self.toolbar.deactivate_all_tools(); target.selected = True; box.setBuffer(target)

        elif self.gui.getFuncButton().segButtonPressed():
            if self.gui.getSeasoning().burshButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
                self.xs, self.ys = [event.xdata], [event.ydata]
                self.bufferSetCurrent(1); self.bufferSetCurrent(2)
                self.canvas.restore_region(self.BILT_BUFFER1)
                self.marker_draw(event.xdata, event.ydata)
                self.canvas.blit(self.subplot.bbox)
            elif self.gui.getSeasoning().eraserButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
                self.xs, self.ys = [event.xdata], [event.ydata]
                self.bufferSetCurrent(1); self.bufferSetCurrent(2)
                self.canvas.restore_region(self.BILT_BUFFER1)
                self.marker_draw(event.xdata, event.ydata)
                self.canvas.blit(self.subplot.bbox)
            else:
                target = self.getLoaded().findSegFromPoint(event.xdata, event.ydata)
                segment.clearBufferAndDeselect()
                if target: target.selected = True; segment.setBuffer(target)

    def onCanvasRelease(self, event: MouseEvent):
        self.press = False
        if self.gui.getFuncButton().bboxButtonPressed():
            anchor.clearBuffer(); return

        if not self.gui.getFuncButton().segButtonPressed(): return
        if segment.getBuffer() and segment.getBuffer().selected:
            for m in self.markers: m.remove()
            self.markers.clear()
            segment.getBuffer().recal_patch()
            self.canvas.draw_idle()
            self.xs.clear(); self.ys.clear()

    def onCanvasDrag(self, event: MouseEvent):
        if not self.press: return

        if self.gui.getFuncButton().bboxButtonPressed():
            a, b = anchor.getBuffer(), box.getBuffer()
            if not (a and a.selected and b and b.selected): return
            if event.inaxes != b.rect.axes: return
            x0, y0 = b.rect.get_xy(); w, h = b.rect.get_width(), b.rect.get_height()
            if a.location == "bottom-left":
                nx, ny = event.xdata, event.ydata; w += x0 - nx; h += y0 - ny; b.rect.set_xy((nx, ny))
            elif a.location == "bottom-right":
                w = event.xdata - x0; h += y0 - event.ydata; b.rect.set_xy((x0, event.ydata))
            elif a.location == "top-right":
                w = event.xdata - x0; h = event.ydata - y0
            elif a.location == "top-left":
                nx = event.xdata; h = event.ydata - y0; w += x0 - nx; b.rect.set_xy((nx, y0))
            elif a.location == "pos-anchor":
                dx = event.xdata - (b.rect.get_x() + w/2); dy = event.ydata - (b.rect.get_y() + h)
                b.rect.set_xy((x0 + dx, y0 + dy))
            b.rect.set_width(w); b.rect.set_height(h); b.anchorUpdate(); self.canvas.draw()

        elif self.gui.getFuncButton().segButtonPressed() and segment.getBuffer() and segment.getBuffer().selected:
            current_x, current_y = event.xdata, event.ydata
            if current_x is None or current_y is None: return

            if self.gui.getSeasoning().burshButtonPressed() or self.gui.getSeasoning().eraserButtonPressed():
                if self.xs and self.ys:
                    lx, ly = self.xs[-1], self.ys[-1]
                    d = np.hypot(current_x - lx, current_y - ly)
                    if d > 5:
                        n = int(d // 1)
                        x_vals = np.linspace(lx, current_x, n + 1)
                        y_vals = np.linspace(ly, current_y, n + 1)
                        for x, y in zip(x_vals, y_vals):
                            self.xs.append(x); self.ys.append(y)
                            self.bufferSetCurrent(1)
                            self.canvas.restore_region(self.BILT_BUFFER1)
                            self.marker_draw(x, y)
                            self.canvas.blit(self.subplot.bbox)
                else:
                    self.xs.append(current_x); self.ys.append(current_y)
                    self.bufferSetCurrent(1)
                    self.canvas.restore_region(self.BILT_BUFFER1)
                    self.marker_draw(current_x, current_y)
                    self.canvas.blit(self.subplot.bbox)

                segment.getBuffer().update_mask(
                    current_x, current_y, self.gui.getSeasoning().get_marker_size(),
                    erase=bool(self.gui.getSeasoning().eraserButtonPressed())
                )

    def marker_draw(self, x, y):
        circle = Circle((x, y), self.gui.getSeasoning().get_marker_size(), color='red', alpha=0.01)
        self.markers.append(circle); self.subplot.add_patch(circle); self.subplot.draw_artist(circle)

    def isLoaded(self): return self.__onLoad is not None
    def getLoaded(self): return self.__onLoad
    def setLoaded(self, abs_obj): self.__onLoad = abs_obj
    def clearLoaded(self): self.__onLoad = None

    @staticmethod
    def isLeftClick(event: MouseEvent): return event.button == 1
