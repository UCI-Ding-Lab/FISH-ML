from ..gui.canvas_view import segment
from tkinter import messagebox

# --- Helper Functions ---
def get_frame_indices(selected_frames, frame_pool):
    if selected_frames == "all":
        return range(len(frame_pool))
    return sorted(list(selected_frames))
def get_target_channels(frame, target_channels):
    if target_channels == "all_channels":
        return list(getattr(frame, "available_channels", []))
    return list(target_channels)
def ensure_source_channel_segmented(frame, source_channel):
    if not frame._get_seg_list_for_channel(source_channel):
        current_focused_channel = getattr(frame, "selected_channel", None)
        frame.selected_channel = source_channel
        _ = frame.segment
        frame.selected_channel = current_focused_channel or source_channel
def build_finalized_mask_from_source_channel(frame, source_channel):
    segmentation_objects = frame._get_seg_list_for_channel(source_channel)
    finalized_mask_list = [
        segmentation_object._segment__data.T
        for segmentation_object in segmentation_objects
    ] if segmentation_objects else []
    frame.set_finalized_mask(finalized_mask_list)
    return finalized_mask_list
def apply_finalized_mask_to_target_channels(frame, target_channels, finalized_mask_list):
    for target_channel in target_channels:
        frame._set_seg_list_for_channel(
            target_channel,
            [segment(frame.gui, mask) for mask in finalized_mask_list] if finalized_mask_list else []
        )
    frame._abstract__seg = frame._get_seg_list_for_channel(frame.selected_channel)
    frame.segment_generated = True
def update_ui_for_focused_frame(frame, abstract_cls):
    if frame is abstract_cls.getBuffer() and frame.gui.getFuncButton().segButtonPressed():
        frame.drawSegmentation = True

# --- Main Logic ---
def apply_channel_mask_to_frames(abstract_cls, source_channel, selected_frames, target_channels, gui=None):
    frame_pool = abstract_cls.getPool()
    frame_indices = get_frame_indices(selected_frames, frame_pool)
    skipped_frames = []
    for i in frame_indices:
        current_frame = frame_pool[i]
        if not current_frame.bbox_generated:
            skipped_frames.append(getattr(current_frame, "sample_id", f"idx{i}"))
            continue
        ensure_source_channel_segmented(current_frame, source_channel)
        finalized_mask_list = build_finalized_mask_from_source_channel(current_frame, source_channel)
        apply_finalized_mask_to_target_channels(
            current_frame,
            get_target_channels(current_frame, target_channels),
            finalized_mask_list
        )
        update_ui_for_focused_frame(current_frame, abstract_cls)
    if skipped_frames:
        messagebox.showinfo("Skipped Frames", f"No bounding box for: {', '.join(skipped_frames)}")