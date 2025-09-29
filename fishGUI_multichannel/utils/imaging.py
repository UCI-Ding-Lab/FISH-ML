# fishgui/utils/imaging.py
import cv2
import numpy as np

def clahe(img, clip_limit=4.0, tile_size=(8, 8)):
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    print("debugging clahe!!!!!!!!!!")
    return c.apply(img)

def normalize_to_uint8(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def grayscale_to_rgb(grayscale_img) -> np.ndarray:
    img_normalized = cv2.normalize(grayscale_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
