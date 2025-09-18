import threading
import logging
from ..services.bundle_data import bundle
from ..services.apply_channel_mask import apply_channel_mask_to_frames
from ..gui.abstract import abstract
import os
from concurrent.futures import ThreadPoolExecutor

class SessionManager:
    __pool = []
    __buffer = None

    """
    Responsible for handling operations that affect the entire session
    or pool of abstract objects.
    """
    # --- Pool Management ---
    @classmethod
    def addToPool(cls, abstract_object):
        """
        Called when a new frame is loaded. 
        Adds this new abstract object to the current pool.
        """
        cls.__pool.append(abstract_object)

    @classmethod
    def getPool(cls):
        """
        Returns the list of all abstract objects currently managed in the session.
        """
        return cls.__pool

    @classmethod
    def setBuffer(cls, abstract_object: abstract):
        """
        Set one frame (abstract object) to focus
        """
        cls.__buffer = abstract_object

    @classmethod
    def getBuffer(cls):
        """
        Get focused frame        
        """
        return cls.__buffer

    # --- Selection/Focus Management ---
    @classmethod
    def selectAll(cls):
        """
        Marks all objects in the pool as selected.
        Used when the user turns the "select" mode on.
        """
        for abstract_object in cls.getPool():
            abstract_object.selected = True

    @classmethod
    def removeUnselected(cls):
        """
        Resets the thumbnail for all objects and deletes the thumbnail
        for unselected ones. Then refocuses.
        """
        for abstract_object in cls.getPool():
            abstract_object.thumbnail = "default"
            if not abstract_object.selected:
                del abstract_object.thumbnail
        cls.sendFocused()

    @classmethod
    def sendFirst(cls):
        """
        Focuses the first selected object in the pool
        """
        for abstract_object in cls.getPool():
            if not isinstance(abstract_object, abstract):
                logging.debug(f"Object {abstract_object} is not an instance of abstract. Skipping.")
                return
            if abstract_object.selected:
                abstract_object.on_click(None)
                return
            
    @classmethod
    def sendFocused(cls):
        """
        Focuses the currently buffered object, or the first selected one
        if none is buffered. 
        """
        current = cls.getBuffer()
        if not isinstance(current, abstract):
            logging.debug(f"Object {current} is not an instance of abstract. Skipping.")
            return
        if current: current.on_click(None)
        else: cls.sendFirst()

    @classmethod
    def remove_segmentation_selection(cls):
        """
        Deselects all frames for segmentation in the pool.
        Ensures a clean state for new operations
        """
        for abstract_object in cls.getPool():
            if not isinstance(abstract_object, abstract):
                logging.debug(f"Object {abstract_object} is not an instance of abstract. Skipping.")
                continue 
            abstract_object.selected_for_segmentation = False  

    # --- Save/Segment/Batch Operations ---
    @classmethod
    def saveBboxChanges(cls):
        """
        Ensures GUI is now in a view-only mode. 
        Called when:
        - The user exits BBOX mode 
        - The user clicks on "Save" and Progress.save is called

        Note for Shizuka:
        Checkout ADDBOX_CALL in tools_panel.py
        Checkout abstract.py def bbox and its setter
        Checkout box.py and anchor.py
        1st line: this simply hides the overlay. the new bbox is saved in self.__bbox in abstract.py
        """
        cls.getBuffer().drawBbox = False 
        cls.sendFocused()

    @classmethod
    def saveSegChanges(cls):
        """
        Ensures GUI is now in a view-only mode. 
        Called when:
        - The user exits SEGMENT mode 
        - The user clicks on "Save" and Progress.save is called
        """
        cls.getBuffer().drawSegmentation = False
        cls.sendFocused()

    @classmethod
    def grabPool(cls) -> list['bundle']:
        """
        Overview: 
            Collects all selected frames and pacakges their data using the bundle class in services/bundle_data.
            Iterates over all abstract objects in the pool, and for each selected frame,
            creates a bundle object containing nucleus path, cytoplasm paths, revised bounding boxes,
            and revised segmentation masks. This is useful for saving session state and loading data.

        Returns:
            list[bundle]: A list of bundle objects, one for each selected frame.
        """
        result = []
        for abstract_object in cls.getPool():
            if not isinstance(abstract_object, abstract):
                logging.debug(f"Object {abstract_object} is not an instance of abstract. Skipping.")
                continue 
            if abstract_object.selected:
                seg_647 = [s._segment__data.T for s in abstract_object._get_seg_list_for_channel("647")]
                seg_488 = [s._segment__data.T for s in abstract_object._get_seg_list_for_channel("488")]
                seg_dict = {"647": seg_647, "488": seg_488}
                bundled_info_for_save = bundle(
                    abstract_object.sample_id,
                    nucleus_path=abstract_object.getNucleusPath(),
                    cyto_paths=list(abstract_object.getCytoplasmPaths()),
                    bbox=abstract_object.boundingBoxRevised,
                    segment=seg_dict
                )
                result.append(bundled_info_for_save)
        return result
    
    @classmethod
    def apply_channel_mask_to_frames(cls, source_channel, selected_frames, target_channels, on_done=None):
        apply_channel_mask_to_frames(cls, source_channel, selected_frames, target_channels, on_done=on_done)

    @classmethod
    def segment_selected(cls, gui):
        """
        Runs segmentation on all frames that the user has marked as "selected for segmentation"
        """
        selected_frames = cls._get_selected_frames()
        ready, not_ready = cls._split_by_bbox_generated(selected_frames)
        if not_ready:
            names = ", ".join(getattr(obj, "sample_id", "?") for obj in not_ready)
            gui.popBox("w", "BBOX Not Ready",
                    f"Skipping segmentation for: {names} (BBOX still not ready).")
        if not ready:
            return

        # max_workers = min(5, os.cpu_count() or 1)  # Limit to 5 or number of CPUs
        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     futures = [executor.submit(cls._segment_each, abs_obj, gui) for abs_obj in ready]
            
        # gui.popBox("i", "Segmentation", f"Started segmentation for {len(selected_frames)} images.")

        # # --- Monitor threads and turn off segmentation selection when done ---
        # def monitor_threads():
        #     for f in futures:
        #         f.result()  
        #     # Turn off the segmentation selection button in the GUI
        #     gui.getFuncButton().toggle["SEGMENTATION_SELECTION"].set(0)

        threads = []
        for abs_obj in selected_frames:
            t = threading.Thread(target=cls._segment_each, args=(abs_obj,gui), daemon=True)
            t.start()
            threads.append(t)

        gui.popBox("i", "Segmentation", f"Started segmentation for {len(selected_frames)} images.")

        def monitor_threads():
            for t in threads:
                t.join()
            gui.getFuncButton().toggle["SEGMENTATION_SELECTION"].set(0)

        threading.Thread(target=monitor_threads, daemon=True).start()
    

    # --- Helper function for segment_selected method ---
    @classmethod
    def _get_selected_frames(cls):
        """
        Returns frames that are selected for segmentation
        """
        selected_frames = [
            abstract_object for abstract_object in cls.getPool()
            if isinstance(abstract_object, abstract) and abstract_object.selected_for_segmentation
        ]
        logging.debug(
            f"Segmenting {len(selected_frames)} images: "
            f"{[str(abstract_object.getNucleusPath().name) for abstract_object in selected_frames]}"
        )
        return selected_frames

    @classmethod
    def _split_by_bbox_generated(cls, frames):
        ready = []
        not_ready = []
        for obj in frames:
            if obj.bbox_generated:
                ready.append(obj)
            else:
                not_ready.append(obj)
        return ready, not_ready
    
    @classmethod
    def _ui_show_segmented(cls, a, gui):
        if a.selected_for_segmentation: 
            a.thumbnail = "segmentation_selected_and_segmented" # orange, blue and green dots
        else:
            a.thumbnail = "segmented" # 
        a.selected_for_segmentation = False
        if a is cls.getBuffer() and gui.getFuncButton().segButtonPressed():
            a.drawSegmentation = True

    @classmethod # TODO - Chceck this method again after abstract.py
    def _segment_each(cls, abs_obj: abstract, gui):
        """
        Runs segmentation for each channel in the frame
        """
        thread_name = threading.current_thread().name
        print(f"[DEBUG] Thread {thread_name} STARTED for sample {abs_obj.sample_id}")
        import time
        start = time.time()
        
        for channel in abs_obj.available_channels:
            seg_list = abs_obj._get_seg_list_for_channel(channel)
            if seg_list:
                abs_obj.seg = seg_list 
                abs_obj.segment_generated = True
            else:
                _ = abs_obj.segment # If segmentation masks isn't present yet, run segmentation for the channel
                abs_obj._set_seg_list_for_channel(channel, abs_obj.seg)
            gui.getRoot().after(0, lambda a=abs_obj: cls._ui_show_segmented(a, gui)) # ensure threading safety and responsiveness
        end = time.time()
        print(f"[DEBUG] Thread {thread_name} FINISHED for sample {abs_obj.sample_id} in {end-start:.2f}s")
    



    
