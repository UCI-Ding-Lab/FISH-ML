import fishCore
import pathlib
import warnings
from PIL import Image
import numpy as np
import train.dataset_prep as dprep
import train.dataset_proc as dproc

# Suppress specific warnings
warnings.filterwarnings("ignore")

fishcore = fishCore.Fish(pathlib.Path("./config.ini"))
fishcore.set_model_version("3.50")

validate_folder = fishcore.config["validate"]["tif_folder"].lower()
tif_files = sorted(list(pathlib.Path(validate_folder).glob("*.tif")))
if not tif_files:
    raise FileNotFoundError("No .tif files found in the folder.")
mat_file = next(pathlib.Path(validate_folder).glob("*.mat"))
if not mat_file:
    raise FileNotFoundError("No .mat file found in the folder.")

mat = dprep.get_masks_from_mat(pathlib.Path(mat_file), "Tracked")
# "Tracked" is the name of the field in the .mat file for 1-50_Hong/1-50_finished.mat

output_folder = fishcore.config["validate"]["output_folder"].lower()
for tif_file in tif_files:
    img_name = tif_file.name
    if img_name in mat["name"]:
        img = np.array(Image.open(tif_file))
        gt_masks = mat["masks"][mat["name"].index(img_name)]
        gt_pos = mat["xy"][mat["name"].index(img_name)]
        fishcore.finetune.validate(img_name, img, gt_masks, gt_pos)
        print(f"Processed {img_name}")
print("Processing complete.")

