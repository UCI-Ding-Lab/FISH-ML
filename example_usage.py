import fishCore
import pathlib
from PIL import Image
import numpy as np
import train.dataset_prep as dprep
import train.dataset_proc as dproc

fishcore = fishCore.Fish(pathlib.Path("./config.ini"))
fishcore.set_modle_version("3.20")
img = np.array(Image.open(pathlib.Path("./assets/tif/1-50_Hong/MAX_KOa_w1-359 DAPI_s032.tif")))
mat = dprep.get_masks_from_mat(pathlib.Path("./assets/tif/1-50_Hong/1-50_finished.mat"), "Tracked")
gt = mat["masks"][mat["name"].index("MAX_KOa_w1-359 DAPI_s032.tif")]
fishcore.sam.info(img, gt)