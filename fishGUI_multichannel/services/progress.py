import pathlib, pickle, threading, concurrent.futures, time
from tkinter import filedialog, messagebox
import matPacker
import re
from ..gui.thumbnails import abstract
from ..gui.canvas_view import box, segment

def _parse_sample_id(path: pathlib.Path) -> str:
    m = re.search(r"s(\d{1,4})", path.stem, re.IGNORECASE)
    return m.group(1) if m else ""

class Progress:
    @staticmethod
    def save(abstract_cls=abstract):
        f = filedialog.asksaveasfilename(defaultextension=".pkl",
                                         filetypes=[("Pickle files", "*.pkl")],
                                         title="Save Session As")
        if not f:
            return
        data = abstract_cls.grabPool()        
        with open(f, "wb") as fh:
            pickle.dump(data, fh)
        messagebox.showinfo("Done", f"Session saved as {f}")

    @staticmethod
    def load(gui, abstract_cls=abstract, abstract_ctor=abstract):
        f = filedialog.askopenfilename(filetypes=[("Progress files", "*.pkl")])
        if not f:
            return

        try:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read {f}:\n{e}")
            return

        abstract_cls.getPool().clear()

        def make_abs(nucleus: pathlib.Path, cyto_paths, sample_id: str):
            return abstract_ctor(sample_id, nucleus, cyto_paths,
                                gui.getTifSequence().gallery_frame, gui)

        for item in data:
            try:
                if isinstance(item, dict) and "nucleus" in item:
                    nucleus = pathlib.Path(item["nucleus"])
                    if not nucleus.exists():
                        messagebox.showwarning("Missing file",
                                            f"Nucleus image not found:\n{nucleus}")
                        continue

                    cyto_all = item.get("cyto", [])
                    cyto_paths = [pathlib.Path(p) for p in cyto_all if pathlib.Path(p).exists()]
                    missing = [p for p in cyto_all if not pathlib.Path(p).exists()]
                    if missing:
                        messagebox.showwarning("Some files missing",
                                            "Skipped missing cyto files:\n" + "\n".join(missing))

                    sample_id = item.get("sample_id") or _parse_sample_id(nucleus)
                    abs_obj = make_abs(nucleus, cyto_paths, sample_id)

                    # restore bbox (no generation)
                    abs_obj.bbox = [box(b, gui) for b in item.get("bbox", [])]

                    # restore per-channel seg caches (don’t call abs_obj.segment)
                    seg_by_channel = item.get("seg_by_channel", {})
                    seg_647 = [segment(gui, m) for m in seg_by_channel.get("647", [])]
                    seg_488 = [segment(gui, m) for m in seg_by_channel.get("488", [])]
                    setattr(abs_obj, "_abstract__seg_647", seg_647)
                    setattr(abs_obj, "_abstract__seg_488", seg_488)

                    # set the active list to the saved/available channel
                    sel = item.get("selected_channel")
                    if sel == "647" and seg_647:
                        abs_obj.segment = seg_647
                    elif sel == "488" and seg_488:
                        abs_obj.segment = seg_488
                    elif seg_647:  # fallback if nothing saved
                        abs_obj.segment = seg_647
                    elif seg_488:
                        abs_obj.segment = seg_488
                    continue

                if isinstance(item, dict) and "path" in item:
                    nucleus = pathlib.Path(item["path"])
                    if not nucleus.exists():
                        messagebox.showwarning("Missing file", f"Image not found:\n{nucleus}")
                        continue
                    abs_obj = make_abs(nucleus, [], _parse_sample_id(nucleus))
                    abs_obj.bbox = [box(b, gui) for b in item.get("bbox", [])]
                    if item.get("seg"):
                        abs_obj.segment = [segment(gui, m) for m in item["seg"]]
                    continue

                if hasattr(item, "L"):
                    path, bbox_list, seg_list = item.L()
                    nucleus = pathlib.Path(path)
                    if not nucleus.exists():
                        messagebox.showwarning("Missing file", f"Image not found:\n{nucleus}")
                        continue
                    abs_obj = make_abs(nucleus, [], _parse_sample_id(nucleus))
                    abs_obj.bbox = [box(b, gui) for b in (bbox_list or [])]
                    if seg_list:
                        abs_obj.segment = [segment(gui, m) for m in seg_list]
                    continue

                messagebox.showwarning("Unrecognized entry", f"Skipping unsupported item: {type(item)}")

            except Exception as e:
                messagebox.showwarning("Skipped one row", f"Reason: {e}")

        abstract_cls.sendFirst()

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