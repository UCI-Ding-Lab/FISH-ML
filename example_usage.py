import fishCore
import pathlib
from PIL import Image
import numpy as np

fishcore = fishCore.fishcore(pathlib.Path("./config.ini"))
fishcore.set_modle_version("2.1")
img = np.array(Image.open(pathlib.Path("./assets/tif/1-50_Hong/MAX_KOa_w1-359 DAPI_s032.tif")))
fishcore.predict(img)