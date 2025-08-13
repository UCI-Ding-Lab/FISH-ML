# fishgui/app.py
import os
import tkinter as tk
from tkinter import messagebox

# --- GUI pieces ---
from .gui.frames import lf
from .gui.canvas_view import stove
from .gui.tools_pannel import seasoning    # (keep file name as you have it)
from .gui.buttons import funcButton
from .gui.thumbnails import tifSequence

# --- Model root import is NOT needed here; GUI files import model themselves
# --- Services are called from the GUI files (Progress, etc.)

# Your backend (fishCore.py) should expose class Fish with a .config dict
import fishCore  # stays at repo root next to FISH-ML code


class FishGUI:
    def __init__(self, root: tk.Tk):
        self._root = root
        self._root.title("FISH-ML GUI")
        self._root.geometry("1100x700")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # --- backend (SAM / DINO wrappers, config incl. icon_folder) ---
        # If fishCore exposes Fish(), instantiate; otherwise adapt as needed.
        self._backend = fishCore.Fish() if hasattr(fishCore, "Fish") else fishCore

        # --- layout frames (A,B,C) ---
        self._lower = lf(self)       # provides getFrameA/B/C()
        self._lower.pack()

        # --- widgets ---
        self._stove = stove(self)            # matplotlib canvas + handlers
        self._thumbs = tifSequence(self)     # thumbnail strip
        self._buttons = funcButton(self)     # Import / Select / BBOX / Segment / Export
        self._seasoning = seasoning(self)    # brush / eraser / sliders / save-load

        # --- pack order mirrors original UI ---
        self._stove.pack()
        self._thumbs.pack()
        self._buttons.pack()
        self._seasoning.pack()

        # wait popup handle
        self._wait_win = None

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


def main():
    root = tk.Tk()
    app = FishGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()



# $env:PYTHONPATH = (Get-Location).Path
# python -m fishgui.app
