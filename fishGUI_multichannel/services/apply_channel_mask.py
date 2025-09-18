from ..gui.canvas.segment import segment
from tkinter import messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import traceback
import time

# --------- Debug print (non-interleaved across threads) ----------
_print_lock = threading.Lock()
def dprint(msg: str):
    with _print_lock:
        print(msg, flush=True)

# --- Helper Functions ---
def get_frame_indices(selected_frames, frame_pool):
    if selected_frames == "all":
        return range(len(frame_pool))
    return sorted(list(selected_frames))

def get_target_channels(frame, target_channels):
    if target_channels == "all_channels":
        return list(getattr(frame, "available_channels", []))
    return list(target_channels)

def _get_gui_root(abstract_cls, frame_pool):
    """Find a Tk root safely (best-effort)."""
    buf = getattr(abstract_cls, "getBuffer", lambda: None)()
    root = None
    if buf is not None and getattr(buf, "gui", None):
        root = buf.gui.getRoot()
    if root is None:
        for f in frame_pool:
            if getattr(f, "gui", None):
                root = f.gui.getRoot()
                break
    return root

# --------- COMPUTE-ONLY helpers (safe off-main-thread) ----------
def ensure_source_channel_segmented(frame, source_channel):
    dprint(f"[DEBUG] ensure_source_channel_segmented: frame={getattr(frame, 'sample_id', '?')}, source_channel={source_channel}")
    if not frame._get_seg_list_for_channel(source_channel):
        current_focused_channel = getattr(frame, "selected_channel", None)
        # Pure data attribute; OK in worker thread:
        frame.selected_channel = source_channel
        dprint(f"[DEBUG]   segmenting channel {source_channel} for frame {getattr(frame, 'sample_id', '?')}")
        _ = frame.segment  # should trigger compute-only segmentation for selected channel
        frame.selected_channel = current_focused_channel or source_channel
        dprint(f"[DEBUG]   segmentation done for channel {source_channel} for frame {getattr(frame, 'sample_id', '?')}")

def build_finalized_mask_from_source_channel(frame, source_channel):
    dprint(f"[DEBUG] build_finalized_mask_from_source_channel: frame={getattr(frame, 'sample_id', '?')}, source_channel={source_channel}")
    segmentation_objects = frame._get_seg_list_for_channel(source_channel)
    finalized_mask_list = [
        segmentation_object._segment__data.T
        for segmentation_object in (segmentation_objects or [])
    ]
    # Data only; OK in worker:
    frame.set_finalized_mask(finalized_mask_list)
    dprint(f"[DEBUG]   built {len(finalized_mask_list)} masks for frame {getattr(frame, 'sample_id', '?')}")
    return finalized_mask_list

# --------- UI helpers (scheduled onto main thread ONLY) ----------
def _apply_masks_on_main(frame, target_channels, finalized_mask_list):
    """Runs on main thread: creates segment() objects and updates frame state."""
    try:
        dprint(f"[DEBUG] apply_finalized_mask_to_target_channels (main): frame={getattr(frame,'sample_id','?')}, targets={target_channels}")
        for target_channel in target_channels:
            dprint(f"[DEBUG]   applying mask to channel {target_channel}")
            frame._set_seg_list_for_channel(
                target_channel,
                [segment(frame.gui, mask) for mask in finalized_mask_list] if finalized_mask_list else []
            )
        frame._abstract__seg = frame._get_seg_list_for_channel(frame.selected_channel)
        frame.segment_generated = True
    except Exception as e:
        dprint(f"[ERROR] _apply_masks_on_main failed for {getattr(frame,'sample_id','?')}: {e}\n{traceback.format_exc()}")

def _update_ui_for_focused_frame_on_main(frame, focused_frame, seg_mode_on):
    try:
        dprint(f"[DEBUG] update_ui_for_focused_frame (main): frame={getattr(frame, 'sample_id', '?')}")
        if frame is focused_frame and seg_mode_on:
            frame.drawSegmentation = True
    except Exception as e:
        dprint(f"[ERROR] _update_ui_for_focused_frame_on_main failed: {e}\n{traceback.format_exc()}")

# --------- Public API (safe; non-blocking UI) ----------
def apply_channel_mask_to_frames(
    abstract_cls,
    source_channel,
    selected_frames,
    target_channels,
    on_done=None,   # <-- NEW: will be called on the Tk main thread when all frames are done
):
    """
    Compute segmentation/masks in background threads; apply results and update UI on Tk main thread.
    Calls on_done() on the main thread after all frames are processed (including skipped reporting).
    """
    frame_pool = abstract_cls.getPool()
    frame_indices = list(get_frame_indices(selected_frames, frame_pool))
    total = len(frame_indices)

    dprint(f"[DEBUG] apply_channel_mask_to_frames: source_channel={source_channel}, "
           f"selected_frames={selected_frames}, target_channels={target_channels}, total_frames={total}")

    # Capture Tk state ONCE (this function is called from main thread via button)
    focused_frame = abstract_cls.getBuffer()
    try:
        seg_mode_on = bool(focused_frame and focused_frame.gui.getFuncButton().segButtonPressed())
    except Exception:
        seg_mode_on = False

    root = _get_gui_root(abstract_cls, frame_pool)
    if root is None:
        dprint("[WARN] apply_channel_mask_to_frames: no Tk root found; UI updates will run inline (best-effort).")

    # ---------------- Coordinator (runs off the main thread) ----------------
    def _run():
        start_all = time.perf_counter()
        skipped_frames = []
        results = []

        def _compute_for_index(i):
            """Worker thread: compute-only; NO Tk calls."""
            frame = frame_pool[i]
            sid = getattr(frame, "sample_id", f"idx{i}")
            tname = threading.current_thread().name
            t0 = time.perf_counter()
            dprint(f"[DEBUG] [{tname}] START compute idx={i} sid={sid}")

            try:
                if not getattr(frame, "bbox_generated", False):
                    dprint(f"[DEBUG] [{tname}] SKIP (no bbox) sid={sid}")
                    return ("skipped", sid, frame, None, None)

                # Compute-only steps:
                ensure_source_channel_segmented(frame, source_channel)
                masks = build_finalized_mask_from_source_channel(frame, source_channel)
                targets = get_target_channels(frame, target_channels)

                dt = time.perf_counter() - t0
                dprint(f"[DEBUG] [{tname}] DONE compute sid={sid} took={dt:.2f}s")
                return ("ok", sid, frame, masks, targets)

            except Exception as e:
                dprint(f"[ERROR] [{tname}] compute failed for {sid}: {e}\n{traceback.format_exc()}")
                return ("error", sid, frame, None, None)

        # Run the compute phase in a small pool
        max_workers = min(5, os.cpu_count() or 1)
        dprint(f"[DEBUG] ThreadPoolExecutor: max_workers={max_workers}, frames={total}")
        try:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ApplyMask") as ex:
                futures = {ex.submit(_compute_for_index, i): i for i in frame_indices}
                for fut in as_completed(futures):
                    results.append(fut.result())
        except Exception as e:
            dprint(f"[ERROR] executor failure: {e}\n{traceback.format_exc()}")

        # ---------------- Main-thread applier ----------------
        def _apply_results_on_main():
            for status, sid, frame, masks, targets in results:
                if status == "skipped":
                    skipped_frames.append(sid)
                    continue
                if status == "error":
                    continue

                # Tk work: create segment objects & update overlay state safely
                _apply_masks_on_main(frame, targets, masks)
                _update_ui_for_focused_frame_on_main(frame, focused_frame, seg_mode_on)
                dprint(f"[DEBUG] process_frame: APPLIED (main) sid={sid}")

            if skipped_frames:
                try:
                    messagebox.showinfo("Skipped Frames", f"No bounding box for: {', '.join(skipped_frames)}")
                except Exception:
                    dprint(f"[INFO] Skipped Frames: {', '.join(skipped_frames)}")

            elapsed = time.perf_counter() - start_all
            dprint(f"[DEBUG] apply_channel_mask_to_frames: DONE (processed={len(results)}/{total}, elapsed={elapsed:.2f}s)")

            # >>> Notify caller that everything has finished (still on the main thread)
            if on_done:
                try:
                    on_done()
                except Exception:
                    dprint("[WARN] on_done callback raised:\n" + traceback.format_exc())

        # Schedule UI work on Tk main thread
        if root is not None:
            root.after(0, _apply_results_on_main)
        else:
            # Best-effort fallback if no root was found
            _apply_results_on_main()

    # Launch the coordinator in the background (non-blocking)
    threading.Thread(target=_run, daemon=True, name="ApplyChannelMaskCoordinator").start()
# def apply_channel_mask_to_frames(abstract_cls, source_channel, selected_frames, target_channels, on_done=None):
#     """
#     Orchestrates mask application:
#       - workers compute segmentation/masks (no Tk)
#       - UI updates (segment objects, thumbnails, messageboxes) happen via root.after on main thread
#     Returns immediately; the work continues in background.
#     """
#     frame_pool = abstract_cls.getPool()
#     frame_indices = list(get_frame_indices(selected_frames, frame_pool))
#     dprint(f"[DEBUG] apply_channel_mask_to_frames: source_channel={source_channel}, selected_frames={selected_frames}, target_channels={target_channels}")
#     dprint(f"[DEBUG] frame_indices={frame_indices}")

#     # Capture any Tk state ONCE on the main thread before we go async
#     focused_frame = abstract_cls.getBuffer()
#     seg_mode_on = False
#     try:
#         # segButtonPressed() touches Tk IntVar; call it here on main thread
#         seg_mode_on = bool(focused_frame and focused_frame.gui.getFuncButton().segButtonPressed())
#     except Exception:
#         # If called off-main, just assume False
#         seg_mode_on = False

#     root = _get_gui_root(abstract_cls, frame_pool)
#     if root is None:
#         dprint("[WARN] apply_channel_mask_to_frames: no Tk root found; running compute only.")
    
#     # Background coordinator thread
#     def _run():
#         skipped_frames = []
#         start_all = time.time()

#         # Inner worker: compute-only per frame
#         def _compute_for_index(i):
#             current_frame = frame_pool[i]
#             sid = getattr(current_frame, 'sample_id', f'idx{i}')
#             dprint(f"[DEBUG] process_frame (compute): index={i}, sample_id={sid}")
#             if not getattr(current_frame, "bbox_generated", False):
#                 dprint(f"[DEBUG]   SKIPPED (no bbox): {sid}")
#                 return ("skipped", sid, current_frame, None, None)

#             try:
#                 ensure_source_channel_segmented(current_frame, source_channel)
#                 masks = build_finalized_mask_from_source_channel(current_frame, source_channel)
#                 targets = get_target_channels(current_frame, target_channels)
#                 return ("ok", sid, current_frame, masks, targets)
#             except Exception as e:
#                 dprint(f"[ERROR] compute failed for {sid}: {e}\n{traceback.format_exc()}")
#                 return ("error", sid, current_frame, None, None)

#         # Use a limited pool for CPU-bound tasks
#         max_workers = min(5, os.cpu_count() or 1)
#         dprint(f"[DEBUG] ThreadPoolExecutor: max_workers={max_workers}")

#         results = []
#         try:
#             with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ApplyMask") as ex:
#                 futures = {ex.submit(_compute_for_index, i): i for i in frame_indices}
#                 for fut in as_completed(futures):
#                     status, sid, frame, masks, targets = fut.result()
#                     results.append((status, sid, frame, masks, targets))
#         except Exception as e:
#             dprint(f"[ERROR] executor failure: {e}\n{traceback.format_exc()}")

#         # Schedule UI updates on main thread as results arrive
#         def _apply_results_on_main():
#             for status, sid, frame, masks, targets in results:
#                 if status == "skipped":
#                     skipped_frames.append(sid)
#                     continue
#                 if status == "error":
#                     # already logged
#                     continue
#                 # Apply masks + maybe show overlay (for focused frame if seg mode on)
#                 _apply_masks_on_main(frame, targets, masks)
#                 _update_ui_for_focused_frame_on_main(frame, focused_frame, seg_mode_on)
#                 dprint(f"[DEBUG] process_frame: DONE {sid}")

#             # Report any skipped frames
#             if skipped_frames:
#                 try:
#                     messagebox.showinfo("Skipped Frames", f"No bounding box for: {', '.join(skipped_frames)}")
#                 except Exception:
#                     # In case messagebox cannot be shown (rare), just log.
#                     dprint(f"[INFO] Skipped Frames: {', '.join(skipped_frames)}")

#             elapsed = time.time() - start_all
#             dprint(f"[DEBUG] apply_channel_mask_to_frames: DONE in {elapsed:.2f}s")

#         if root is not None:
#             # Execute UI work on the main thread
#             root.after(0, _apply_results_on_main)
#         else:
#             # No root: run UI part inline (best effort; not ideal)
#             _apply_results_on_main()

#     # Launch coordinator
#     threading.Thread(target=_run, daemon=True, name="ApplyChannelMaskCoordinator").start()
