import pathlib
import cv2
import numpy as np

def grayscale_to_rgb(grayscale_img) -> np.ndarray:
    img_normalized = cv2.normalize(grayscale_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
    brightness_factor = 1
    return np.clip(img_rgb * brightness_factor, 0, 255).astype(np.uint8)

folder = pathlib.Path("assets\demotifs")

for tif_file in folder.glob("*.tif"):
    grayscale_img = cv2.imread(str(tif_file), cv2.IMREAD_UNCHANGED)
    rgb_img = grayscale_to_rgb(grayscale_img)
    jpg_file = tif_file.with_suffix(".jpg")
    cv2.imwrite(str(jpg_file), rgb_img)