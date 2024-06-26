import fishCore
import pathlib
import cv2
from PIL import Image
import numpy as np

fishcore = fishCore.fishcore(pathlib.Path("./config.ini"))
fishcore.set_modle_version("2.1")
img = cv2.cvtColor(np.array(Image.open(pathlib.Path("./img/test_1.jpg"))), cv2.COLOR_GRAY2RGB).astype(np.uint8)
fishcore.predict(img)