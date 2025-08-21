# fishgui/services/progress.py
import pathlib, pickle, threading, concurrent.futures, time
from tkinter import filedialog, messagebox
from ..model.abstract import abstract
from ..model.shapes import box
from ..model.segment import segment
import matPacker  # keep as in your project

class Progress:
    @staticmethod
    def save(abstract_cls=abstract):
        """Serialize the current pool as a simple, portable list of dicts."""
        f = filedialog.asksaveasfilename(defaultextension=".pkl",
                                         filetypes=[("Pickle files", "*.pkl")],
                                         title="Save Session As")
        if not f:
            return
        data = []
        for a in abstract_cls.getPool():
            item = {
                "path": str(a.getAbsPath()),
                "bbox": [b.final for b in a.bbox] if not a.noBbox() else [],
                "seg":  [s._data.T for s in a.segment] if not a.noSegment() else [],
            }
            data.append(item)
        with open(f, "wb") as file:
            pickle.dump(data, file)
        messagebox.showinfo("Done", "Session saved as " + f)

    @staticmethod
    def load(gui, abstract_cls=abstract, abstract_ctor=abstract):
        """Restore from the list-of-dicts we save above."""
        f = filedialog.askopenfilename(filetypes=[("Progress files", "*.pkl")])
        if not f:
            return
        try:
            with open(f, "rb") as file:
                data = pickle.load(file)
            abstract_cls.getPool().clear()
            for item in data:
                p = pathlib.Path(item["path"])
                if not p.exists():
                    messagebox.showwarning("Warning", f"Image {p} not found!")
                    continue
                abs_obj = abstract_ctor(p, gui.getTifSequence().gallery_frame, gui)
                # Rebuild objects
                abs_obj.bbox = [box(b, gui) for b in item.get("bbox", [])]
                abs_obj.segment = [segment(gui, m) for m in item.get("seg", [])]
            abstract_cls.sendFirst()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load session: {e}")

    @staticmethod
    def export(gui):
        """Export selected items with explicit segments to MATLAB .mat."""
        toSave = [a for a in abstract.getPool() if a.selected and len(a.segmentExplict)]
        if not toSave:
            messagebox.showwarning("Nothing to export", "No selected images with segments.")
            return
        f = filedialog.asksaveasfilename(defaultextension=".mat",
                                         filetypes=[("Matlab files", "*.mat")],
                                         title="Export Results As")
        if not f:
            return
        names, xys, masks = [], [], []
        for a in toSave:
            names.append(str(a.getAbsPath()))
            xys.append([s.xy for s in a.segment])
            masks.append([s.box for s in a.segment])
        matPacker.create(names, xys, masks, f)

    @staticmethod
    def generateBbox(gui, abstracts):
        """Kick off background bbox generation for all items in 'abstracts'."""
        def generate_bboxes():
            def generate_bbox(a):
                start = time.time()
                _ = a.bbox  # triggers generation
                print(f"Generated bbox for {a.getAbsPath()} in {time.time() - start:.4f}s")
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(4, len(abstracts)))
            ) as ex:
                list(ex.map(generate_bbox, abstracts))
        threading.Thread(target=generate_bboxes, daemon=True).start()
save = Progress.save
load = Progress.load
export = Progress.export
generateBbox = Progress.generateBbox