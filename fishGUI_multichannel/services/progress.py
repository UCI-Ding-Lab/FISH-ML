import pickle, threading, concurrent.futures, time
from tkinter import filedialog, messagebox
from ..gui.abstract import abstract
from ..gui.canvas_view import box, segment
from .bundle_data import bundle
from .matPacker import create
from ..gui.fishGUI import FishGUI

class Progress:
    @staticmethod
    def save():
        """
        Save current session data including all selected frames, its boundary boxes and 
        segmentation masks if exist.
        """
        filename = filedialog.asksaveasfilename(defaultextension=".pkl", 
                                         filetypes=[("Pickle files", "*.pkl")],
                                         title="Save Session As")
        if not filename: 
            return
        list_of_bundled_data = abstract.grabPool() # grabPool() returns a list of bundled data (includes file paths, bbox and masks for all abstract objects)
        with open(filename, "wb") as file:
            pickle.dump(list_of_bundled_data, file)
        messagebox.showinfo("Done", "Session saved as " + filename)    
        
    def load(gui: FishGUI):
        """
        Load previous session data including all selected frames, its boundary boxes and
        segmentation masks if exist. 
        """
        # --- Helper Functions ---
        def get_filename_to_load():
            return filedialog.askopenfilename(filetypes=[("Progress files", "*.pkl")])
        
        def read_data_from_file(filename):
            try:
                with open(filename, "rb") as file:
                    return pickle.load(file) # returns list of bundled data 
            except Exception as error:
                messagebox.showerror("Error", f"Could not read {filename}:\n{error}")
                return None
        
        def clear_previous_session():
            """
            - getPool() returns a list of abstract objects in current session
            - clear() empties this list and references pointing to each objects are removed as well, 
              deleting all objects in current session automatically
            """
            abstract.getPool().clear()
            
        def return_valid_paths(nucleus_path, cytoplasm_paths):
            if not nucleus_path.exists():
                messagebox.showwarning("Missing file", "Nucleus image not found:\n{nucleus_path}")
                return None
        
            missing_cytoplasm_file = [str(path) for path in cytoplasm_paths if not path.exists()]
            if missing_cytoplasm_file:
                messagebox.showwarning("Missing file", "Cytoplasm image(s) not found:\n" + "\n".join(missing_cytoplasm_file))
                return None
            
            return nucleus_path, cytoplasm_paths
            
        def create_abstract_object(nucleus_path, cyto_paths, bbox_list, seg_list, gui: FishGUI):
            abstract_object = abstract(
                nucleus_path=nucleus_path,
                cyto_paths=cyto_paths,
                gallery_frame=gui.getTifSequence().gallery_frame,
                gui=gui
            )
            abstract_object.bbox = [box(b, gui) for b in bbox_list]
            abstract_object.segment = [segment(gui, m) for m in seg_list]
            return abstract_object

        # --- Main Logic ---
        load_filename = get_filename_to_load()
        if not load_filename: return
        session_data = read_data_from_file(load_filename)
        if session_data is None: return
        clear_previous_session()
        for item in session_data:
            try:
                single_bundle : bundle = item
                nucleus_path, cytoplasm_paths, bbox_list, seg_list = single_bundle.extract_data_from_bundles() 
                valid_paths = return_valid_paths(nucleus_path, cytoplasm_paths) 
                if valid_paths is None:
                    continue
                create_abstract_object(nucleus_path, cytoplasm_paths, bbox_list, seg_list, gui)
            except Exception as error:
                messagebox.showwarning("Skipped one row", f"Reason:{error}")
        abstract.sendFirst()

    @staticmethod
    def generateBbox(gui: FishGUI, list_of_abstract_objects: list[abstract]):
        """
        Generates bounding boxes using multithreading
        Called in gui/buttons.py, IMPORT_call method
        This ensures that boundary boxes are generate once images are imported
        """
        # --- Helper functions ---
        def generate_bbox_for_object(single_abstract_object: abstract):
            start_time = time.time()
            _ = single_abstract_object.bbox
            end_time = time.time()
            print(f"Generated bbox for {single_abstract_object.getNucleusPath()} in {end_time - start_time:.4f} seconds")

        def generate_bboxes():
            max_workers = min(3, len(list_of_abstract_objects))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(generate_bbox_for_object, list_of_abstract_objects)

        # --- Main Logic ---
        threading.Thread(target=generate_bboxes, daemon=True).start() # Start the thread of generating bboxes