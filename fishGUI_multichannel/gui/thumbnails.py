import tkinter
import pathlib
import re
import logging
from ..services.session_manager import SessionManager

logger = logging.getLogger(__name__) 

"""
Manages the gallery of TIFF image sequences for the GUI.
Handles scrolling, thumbnail display, and grouping by sample/channel.
"""
class tifSequence():
    def __init__(self, gui):
        self.gui = gui
        container = gui.getLowerFrame().getFrameB()
        self.base = tkinter.Canvas(container, height=74)

        self.scrollbar = tkinter.Scrollbar(container, orient=tkinter.HORIZONTAL, command=self.base.xview)
        self.base.configure(xscrollcommand=self.scrollbar.set)

        self.gallery_frame = tkinter.Frame(self.base)
        self.base.create_window((0, 0), window=self.gallery_frame, anchor="nw")

        self.base.bind("<Configure>", lambda e: self.update_scrollregion())
        self.base.bind_all("<MouseWheel>", self.on_mouse_wheel)
        self.base.bind_all("<Button-4>", self.on_mouse_wheel)
        self.base.bind_all("<Button-5>", self.on_mouse_wheel)

    def update_scrollregion(self):
        self.base.update_idletasks()
        self.base.config(scrollregion=self.base.bbox("all"))

    def on_mouse_wheel(self, event):
        if event.num == 4:  # Linux scrolling up
            self.base.xview_scroll(-1, "units")
        elif event.num == 5:  # Linux scrolling down
            self.base.xview_scroll(1, "units")
        elif event.delta:  # Windows/macOS
            if event.delta > 0:
                self.base.xview_scroll(-1, "units")
            else:
                self.base.xview_scroll(1, "units")
        
    def pack(self):
        self.base.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        self.scrollbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    
    def unpack(self):
        self.base.pack_forget()
        self.scrollbar.pack_forget()
    
    def resetPosition(self):
        self.base.xview_moveto(0)
        self.base.yview_moveto(0)

    # Called in buttons.py, IMPORT_call method    
    def addToGallery(self, tif_files: list):
        from .abstract import abstract # prevent circular imports
        logger.debug(f"addToGallery → starting with {len(tif_files)} files")

        def parse_sampleID_and_channel(path: pathlib.Path):
            stem = path.stem
            sample_match = re.search(r"s(\d{1,4})", stem, re.IGNORECASE)
            channel_match = re.search(r"w[-_]?(?:.*?)?(DAPI|488|647)", stem, re.IGNORECASE)
            if not (sample_match and channel_match):
                logger.warning(f"addToGallery → skipping {stem!r}, couldn't parse s### or w###")
                return None, None
            sample_id = sample_match.group(1)
            channel_name = channel_match.group(1).upper()
            return sample_id, channel_name

        def group_files_by_sample_and_channel(file_paths: list):
            """Group files into a dict[sample_id][channel_name] = path."""
            grouped = {}
            for file_path in file_paths:
                path = pathlib.Path(file_path)
                sample_id, channel_name = parse_sampleID_and_channel(path)
                if sample_id and channel_name:
                    grouped.setdefault(sample_id, {})[channel_name] = path
            return grouped

        def get_cytoplasm_paths(channels: dict):
            """Return list of cytoplasm channel paths (647, 488) if present."""
            cyto_paths = []
            if "647" in channels:
                cyto_paths.append(channels["647"])
            if "488" in channels:
                cyto_paths.append(channels["488"])
            return cyto_paths

        grouped = group_files_by_sample_and_channel(tif_files)
        logger.debug(f"addToGallery → grouped into samples: {list(grouped.keys())}")

        for sample_id, channels in grouped.items():
            nucleus_path = channels.get("DAPI")
            if nucleus_path is None:
                logger.warning(f"addToGallery → sample {sample_id} has no DAPI, skipping")
                continue

            cyto_paths = get_cytoplasm_paths(channels)
            logger.info(f"addToGallery → instantiating abstract for sample {sample_id}")
            abs_obj = abstract(
                sample_id,
                nucleus_path,
                cyto_paths,
                self.gallery_frame,
                self.gui
            )
            self.gui.getSeasoning().update_channel_menu(abs_obj.available_channels)

        SessionManager.sendFirst()
        self.update_scrollregion()