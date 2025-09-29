import numpy as np
from skimage import filters, morphology, segmentation
from skimage.restoration import estimate_sigma
from scipy import ndimage as ndi
import cv2
import logging

from ..gui.canvas.segment import segment
from ..utils.image_preprocessing import (
    remove_outliers,
    normalize_to_uint8,
    clahe,
    postproc_mask,  
    gradient,
    mask_to_bbox       
)

logger = logging.getLogger(__name__)

def watershed_segment_with_centers(cyt_img: np.ndarray,
                                    centers: list[tuple[float,float]]
                                    ) -> list[np.ndarray]:
    """
    Segment cytoplasm with image using watershed, seeded at the nucleus center
    Returns a list of binary masks. 
    """
    thresh = filters.threshold_otsu(cyt_img)
    binary = morphology.remove_small_holes(cyt_img > thresh, area_threshold=1000)
    binary = morphology.remove_small_objects(binary, min_size=1000)
    dist   = ndi.distance_transform_edt(binary)

    markers = np.zeros(cyt_img.shape, np.int32)
    for i,(cx,cy) in enumerate(centers, start=1):
        xi, yi = int(round(cx)), int(round(cy))
        if 0<=yi<cyt_img.shape[0] and 0<=xi<cyt_img.shape[1]:
            markers[yi,xi] = i

    elev   = -dist + 5*filters.sobel(cyt_img)
    labels = segmentation.watershed(elev, markers=markers, mask=binary)

    masks = []
    for lab in range(1, labels.max()+1):
        m = (labels==lab)
        if m.sum()>0: masks.append(postproc_mask(m))
    return masks

def run_basic_watershed(
    nucleus_img: np.ndarray,
    cyto_647: np.ndarray,
    cyto_488: np.ndarray,
    gui,
    selected_channel: str
) -> tuple[list[segment], list[segment]]:
    """
    Perform segmentation for both channels and return (seg_647, seg_488)
    """
    boxes = gui.getBackEnd().AppIntDINOwrapper(nucleus_img)
    centers = [((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in boxes]

    # process 647 first (cyto1), then 488 (cyto2)
    # TODO - O(n^2) -- consider improving time complexity
    seg_647, seg_488 = [], []
    for image, channel in [(cyto_647, "647"), (cyto_488, "488")]:
        if image is None:
            continue
        # image = remove_outliers(image) # TODO check with margaret - is this necessary?
        # image = normalize_to_uint8(image) # remove_outlier does nt guarantee that the output is 0-255 so calling this method once again is necessary

        if channel == "647":
            clahe_img = clahe(image, clip_limit=2.0, tile_size=(8,8))
            grad  = gradient(clahe_img, ksize=5)
            proc = clahe_img
            rgb  = np.stack([clahe_img, clahe_img, grad], axis=-1)

        else:  # chan == "488"
            rem = remove_outliers(image) # TODO - check with margaret why we remove twice
            noramlized_img= normalize_to_uint8(rem) 
            cyt_clahe = clahe(noramlized_img, clip_limit=4.0, tile_size=(8,8))

            sigma_est = estimate_sigma(cyt_clahe, channel_axis=None, average_sigmas=True)
            sigma_norm = sigma_est + 3.0
            sigma_weak = sigma_est - 10.0

            cyt_bilat = cv2.bilateralFilter(cyt_clahe, d=9, sigmaColor=sigma_norm, sigmaSpace=15, borderType=cv2.BORDER_REFLECT_101)
            cyt_edge_preserved = cv2.edgePreservingFilter(cyt_clahe, flags=1, sigma_s=sigma_norm, sigma_r=0.4)
            cyt_bilat_edge = cv2.edgePreservingFilter(cyt_bilat, flags=1, sigma_s=sigma_weak, sigma_r=0.4)

            laplacian = cv2.Laplacian(noramlized_img, cv2.CV_64F)
            laplacian = cv2.convertScaleAbs(laplacian)
            cyt_blended = cv2.addWeighted(cyt_bilat, 0.8, laplacian, 0.2, 0)

            proc = cyt_bilat_edge
            rgb  = np.stack([cyt_blended, cyt_bilat, cyt_edge_preserved], axis=-1)

        ws_masks = watershed_segment_with_centers(proc, centers)
        bboxes = [mask_to_bbox(m) for m in ws_masks]
        bboxes = [b for b in bboxes if b is not None]

        channel_masks = []
        for bb in bboxes:
            try:
                box_input = [[[float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]]]
                sets = gui.getBackEnd().finetune.AppIntPREDICTCytoplasmWrapper(rgb, box_input)
                if sets is not None and len(sets) > 0 and sets[0] is not None and len(sets[0]) > 0:
                    best = max(sets[0], key=lambda m: m.sum())
                    channel_masks.append(postproc_mask(best))
            except Exception as e:
                logger.error(f"SAM refine failed on {channel} box {bb}: {str(e)}")
        
        if channel == "647":
            seg_647 = [segment(gui, m) for m in channel_masks]
        else:
            seg_488 = [segment(gui, m) for m in channel_masks]
        
    return seg_647, seg_488


# TODO remove if unnecessary
# def watershed_segment(cyt_img: np.ndarray,
#                       centers,
#                       thresh_method: str = "otsu") -> list[np.ndarray]:
#     thresh = filters.threshold_otsu(cyt_img) if thresh_method == "otsu" else np.percentile(cyt_img, 30)
#     binary = cyt_img > thresh
#     binary = morphology.remove_small_holes(binary, area_threshold=1000)
#     binary = morphology.remove_small_objects(binary, min_size=1000)
#     dist = ndi.distance_transform_edt(binary)

#     markers = np.zeros(cyt_img.shape, dtype=np.int32)
#     for idx, (cx, cy) in enumerate(centers, start=1):
#         xi, yi = int(round(cx)), int(round(cy))
#         if 0 <= yi < cyt_img.shape[0] and 0 <= xi < cyt_img.shape[1]:
#             markers[yi, xi] = idx

#     labels = segmentation.watershed(-dist, markers, mask=binary)
#     return [postproc_mask(labels == i) for i in range(1, len(centers)+1)]
