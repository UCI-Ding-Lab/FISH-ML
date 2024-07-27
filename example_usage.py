import fishCore
import pathlib
from PIL import Image
import numpy as np
import train.dataset_prep as dprep
import train.dataset_proc as dproc

fishcore = fishCore.Fish(pathlib.Path("./config.ini"))
fishcore.set_model_version("3.50")
img = np.array(Image.open(pathlib.Path("./assets/tif/1-50_Hong/MAX_KOa_w1-359 DAPI_s032.tif")))
img2 = np.array(Image.open(pathlib.Path("./assets/tif/201-250_Hong/MAX_CTLa_w1-359 DAPI_s026.tif")))
mat = dprep.get_masks_from_mat(pathlib.Path("./assets/tif/1-50_Hong/1-50_finished.mat"), "Tracked")
gt_masks = mat["masks"][mat["name"].index("MAX_KOa_w1-359 DAPI_s032.tif")]
gt_pos = mat["xy"][mat["name"].index("MAX_KOa_w1-359 DAPI_s032.tif")]
fishcore.finetune.info(img, gt_masks, gt_pos)

