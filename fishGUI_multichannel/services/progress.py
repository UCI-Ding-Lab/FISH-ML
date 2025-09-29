import pathlib, pickle, threading, concurrent.futures, time
from tkinter import filedialog, messagebox
import matPacker
import re
from ..gui.thumbnails import abstract
from ..gui.canvas_view import box, segment
import pickle

class Progress:
    @staticmethod
    def save(abstract_class=abstract):
        # --- Helper functions ---
        def get_save_filename():
            return filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl")],
                title="Save Session As"
            )
        def get_session_data():
            return abstract_class.grabPool()
        def write_session_to_file(filename, session_data):
            with open(filename, "wb") as file:
                pickle.dump(session_data, file)
        def notify_save_complete(filename):
            messagebox.showinfo("Done", f"Session saved as {filename}")
        
        # --- Main Logic ---
        save_filename = get_save_filename()
        if not save_filename:
            return
        session_data = get_session_data()
        write_session_to_file(save_filename, session_data)
        notify_save_complete(save_filename)

    @staticmethod
    def load(gui, abstract_class=abstract):
        # --- Helper Functions ---
        def get_load_filename():
            return filedialog.askopenfilename(filetypes=[("Progress files", "*.pkl")])
        def read_session_from_file(filename):
            with open(filename, "rb") as file:
                return pickle.load(file)
        def clear_current_pool():
            abstract_class.getPool().clear()
        def create_abstract_object(nucleus_path, cyto_paths, bbox_list, seg_list, item, gui):
            abs_obj = abstract_class(
                sample_id=getattr(item, "sample_id", nucleus_path.stem),
                nucleus_path=nucleus_path,
                cyto_paths=cyto_paths,
                gallery_frame=gui.getTifSequence().gallery_frame,
                gui=gui
            )
            abs_obj.bbox = [box(b, gui) for b in (bbox_list or [])]
            abs_obj.segment = [segment(gui, m) for m in seg_list]
            return abs_obj
        def notify_missing_file(message):
            messagebox.showwarning("Missing file", message)
        def notify_unrecognized_entry(item_type):
            messagebox.showwarning("Unrecognized entry", f"Skipping unsupported item: {item_type}")
        def notify_skipped_row(error):
            messagebox.showwarning("Skipped one row", f"Reason:{error}")
        def notify_load_error(filename, error):
            messagebox.showerror("Error", f"Could not read {filename}:\n{error}")
        def send_first():
            abstract_class.sendFirst()

        # --- Main Logic ---
        load_filename = get_load_filename()
        if not load_filename: return
        try:
            session_data = read_session_from_file(load_filename)
        except Exception as error:
            notify_load_error(load_filename, error)
            return

        clear_current_pool()
        for item in session_data:
            try:
                if hasattr(item, "L"):
                    nucleus_path, cytoplasm_paths, bbox_list, seg_list = item.L()
                    if not nucleus_path.exists():
                        notify_missing_file(f"Nucleus image not found:\n{nucleus_path}")
                        continue
                    missing_cytoplasm_file = [str(p) for p in cytoplasm_paths if not p.exists()]
                    if missing_cytoplasm_file:
                        notify_missing_file(f"Cytoplasm image(s) not found:\n" + "\n".join(missing_cytoplasm_file))
                    create_abstract_object(nucleus_path, cytoplasm_paths, bbox_list, seg_list, item, gui)
                else:
                    notify_unrecognized_entry(type(item))
            except Exception as error:
                notify_skipped_row(error)
        send_first()

    @staticmethod
    def export_finalized_masks(gui, abstract_class=abstract):
        # --- Helper functions ---
        def get_export_filename():
            return filedialog.asksaveasfilename(
                defaultextension=".mat",
                filetypes=[("Matlab files", "*.mat")],
                title="Export Finalized Masks As"
            )
        def get_frames_to_export():
            return [
                frame for frame in abstract_class.getPool()
                if frame.selected and getattr(frame, "finalized_mask", None)
            ]
        def notify_export_cancelled():
            messagebox.showinfo("Export Cancelled", "No file was selected for export.")
        def notify_no_data():
            messagebox.showinfo("No Data", "No frames are selected or have finalized masks for export.")
        def collect_export_data(frames_to_export):
            cytoplasm_filenames = []
            segmentation_masks = []
            for frame in frames_to_export:
                for cytoplasm_path in frame.getCytoplasmPaths():
                    cytoplasm_filenames.append(str(cytoplasm_path))
                    segmentation_masks.append(frame.finalized_mask)
            return cytoplasm_filenames, segmentation_masks
        def write_mat_file(cytoplasm_filenames, segmentation_masks, filename):
            matPacker.create(cytoplasm_filenames, segmentation_masks, filename)
        

        # --- Main Logic ---
        export_filename = get_export_filename()
        if not export_filename:
            notify_export_cancelled()
            return
        frames_to_export = get_frames_to_export()
        if not frames_to_export:
            notify_no_data()
            return
        cytoplasm_filenames, segmentation_masks = collect_export_data(frames_to_export)
        write_mat_file(cytoplasm_filenames, segmentation_masks, export_filename)

    @staticmethod
    def generateBbox(gui, abstracts):
        def generate_bbox_for_object(abstract_object):
            # --- Helper functions ---
            start_time = time.time()
            _ = abstract_object.bbox
            end_time = time.time()
            print(f"Generated bbox for {abstract_object.getAbsPath()} in {end_time - start_time:.4f} seconds")
        def generate_bboxes_in_thread():
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(4, len(abstracts)))
            ) as executor:
                list(executor.map(generate_bbox_for_object, abstracts))
        
        # --- Main Logic ---
        threading.Thread(target=generate_bboxes_in_thread, daemon=True).start()

# Shortcut aliases for Progress class methods allowing direct calls
save = Progress.save
load = Progress.load
export = Progress.export_finalized_masks
generateBbox = Progress.generateBbox