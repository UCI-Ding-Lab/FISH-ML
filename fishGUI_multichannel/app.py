import configparser
import pathlib
import os
import tkinter as tk
from tkinter import messagebox
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
    datefmt="%H:%M:%S",
)

# --- GUI pieces ---
from .gui.frames import lf
from .gui.canvas_view import stove
from .gui.tools_pannel import seasoning    
from .gui.buttons import funcButton
from .gui.thumbnails import tifSequence
import fishCore 


class FishGUI:
    def __init__(self, root: tk.Tk):
        self._root = root
        self._root.title("FISH-ML GUI")
        self._root.geometry("870x1000")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # set config path
        proj_root = pathlib.Path(__file__).resolve().parents[1]
        config_path = None
        candidates = [
            proj_root / "config.ini",
            pathlib.Path.cwd() / "config.ini",
        ]
        for path in candidates:
            if path.exists():
                config_path = path
                break
        if not config_path:
            messagebox.showerror("Config Error", "config.ini not found!")
            raise FileNotFoundError("config.ini not found")

        # Instantiate backend with config file path 
        try:
            self._backend = fishCore.Fish(config_path)
            self._backend.set_model_version("3.50")
        except Exception as e:
            messagebox.showerror("Config Error", f"Failed to initialize backend: {e}")
            raise

        self._lower = lf(self)
        self._lower.pack()
        self._stove = stove(self)
        self._thumbs = tifSequence(self)
        self._buttons = funcButton(self)
        self._seasoning = seasoning(self)
        self._stove.pack()
        self._thumbs.pack()
        self._buttons.pack()
        self._seasoning.pack()

        # wait popup handle
        self._wait_win = None
        
        # Add delete key binding like in multichannel version
        self._root.bind('<BackSpace>', self.onDelete)
        self._root.focus_set()

    # ======= Small helpers used by submodules =======
    def getRoot(self) -> tk.Tk:
        return self._root

    def getBackEnd(self):
        return self._backend

    def getLowerFrame(self) -> lf:
        return self._lower

    def getStove(self) -> stove:
        return self._stove

    def getTifSequence(self) -> tifSequence:
        return self._thumbs

    def getFuncButton(self) -> funcButton:
        return self._buttons

    def getSeasoning(self) -> seasoning:
        return self._seasoning

    # ---- popups / status ----
    def popBox(self, level: str, title: str, msg: str):
        level = (level or "").lower()
        if level.startswith("i"):
            messagebox.showinfo(title, msg, parent=self._root)
        elif level.startswith("w"):
            messagebox.showwarning(title, msg, parent=self._root)
        else:
            messagebox.showerror(title, msg, parent=self._root)

    def indicateWait(self, label: str = "Working…"):
        # tiny modal-ish window to show work in progress
        if self._wait_win and tk.Toplevel.winfo_exists(self._wait_win):
            return
        self._wait_win = tk.Toplevel(self._root)
        self._wait_win.title("Please wait")
        self._wait_win.geometry("+%d+%d" % (self._root.winfo_rootx() + 80,
                                            self._root.winfo_rooty() + 80))
        self._wait_win.transient(self._root)
        self._wait_win.grab_set()
        tk.Label(self._wait_win, text=f"{label}…", padx=20, pady=14).pack()
        self._wait_win.update_idletasks()

    def dismissWait(self):
        if self._wait_win and tk.Toplevel.winfo_exists(self._wait_win):
            self._wait_win.grab_release()
            self._wait_win.destroy()
        self._wait_win = None

    def onDelete(self, event):
        """Delete operation (Backspace) - matches multichannel version"""
        from .gui.canvas_view import box, segment
        
        selected_box = box.getBuffer()
        selected_seg = segment.getBuffer()

        if selected_box:
            selected_box.delete()
        elif selected_seg:
            selected_seg.delete()
        else:
            self.popBox('w', 'No Selection', 'No bounding box or segmentation mask is selected.')

#to run the app: python -m fishGUI_multichannel.app
def main():
    root = tk.Tk()
    app = FishGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
