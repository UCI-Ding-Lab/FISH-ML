from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from ..app import FishGUI

class FishToolBar(NavigationToolbar2Tk):
    def __init__(self, canvas, window, gui: FishGUI):
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
