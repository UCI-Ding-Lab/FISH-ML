import pathlib, pickle, threading, concurrent.futures, time
from tkinter import filedialog, messagebox
import matPacker
import re
from ..gui.thumbnails import abstract
from ..gui.canvas_view import box, segment
import pickle
import session_loader


class Progress:
    @staticmethod
    def save(abstract_cls=abstract):
        f = filedialog.asksaveasfilename(defaultextension=".pkl",
                                         filetypes=[("Pickle files", "*.pkl")],
                                         title="Save Session As")
        if not f:
            return
        data = abstract_cls.grabPool()        
        with open(f, "wb") as file:
            pickle.dump(data, file)
        messagebox.showinfo("Done", f"Session saved as {f}")

    @staticmethod
    def load(gui, abstract_class=abstract):
        f = filedialog.askopenfilename(filetypes=[("Progress files", "*.pkl")])
        if not f:
            return
        try:
            with open(f, "rb") as file:
                data = pickle.load(file)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read {f}:\n{e}")
            return

        abstract_class.getPool().clear() # ensure that only the data from the loaded session are present 
        for item in data:
            try:
                if hasattr(item, "L"):
                    nucleus_path, cyto_paths, bbox_list, seg_list = item.L()
                    if not nucleus_path.exists():
                        messagebox.showwarning("Missing file", f"Nucleus image not found:\n{nucleus_path}")
                        continue
                    missing_cyto = [str(p) for p in cyto_paths if not p.exists()]
                    if missing_cyto:
                        messagebox.showwarning("Missing file", f"Cyto image(s) not found:\n" + "\n".join(missing_cyto))
                    abs_obj = abstract_class(
                        sample_id=getattr(item, "sample_id", nucleus_path.stem),
                        nucleus_path=nucleus_path,
                        cyto_paths=cyto_paths,
                        gallery_frame=gui.getTifSequence().gallery_frame,
                        gui=gui
                    )
                    abs_obj.bbox = [box(b, gui) for b in (bbox_list or [])] 
                    abs_obj.segment = [segment(gui, m) for m in seg_list]
                else:
                    messagebox.showwarning("Unrecognized entry", f"Skipping unsupported item: {type(item)}")
            except Exception as e:
                messagebox.showwarning("Skipped one row", f"Reason:{e}")
        abstract_class.sendFirst()

    @staticmethod
    def export_finalized_masks(gui, abstract_class=abstract):
        filename = filedialog.asksaveasfilename(
            defaultextension=".mat",
            filetypes=[("Matlab files", "*.mat")],
            title="Export Finalized Masks As"
        )
        if not filename:
            messagebox.showinfo("Export Cancelled", "No file was selected for export.")
            return

        frames_to_export = [
            frame for frame in abstract_class.getPool()
            if frame.selected and getattr(frame, "finalized_mask", None)
        ]
        if not frames_to_export:
            messagebox.showinfo("No Data", "No frames are selected or have finalized masks for export.")
            return

        cytoplasm_filenames = []
        segmentation_masks = []
        for frame in frames_to_export:
            for cytoplasm_path in frame.getCytoplasmPaths():
                cytoplasm_filenames.append(str(cytoplasm_path))
                segmentation_masks.append(frame.finalized_mask)

        matPacker.create(cytoplasm_filenames, segmentation_masks, filename)
        
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