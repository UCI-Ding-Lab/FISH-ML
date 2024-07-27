import numpy as np
from datasets import Dataset
from PIL import Image
import pathlib
from transformers import SamProcessor
from transformers import SamModel, SamConfig
from torch.utils.data import DataLoader
from patchify import patchify
import os
import torch
from torch.optim import Adam
import monai
from transformers import SamModel, SamProcessor
import fishLoader as fish
import dataset_proc as ppf
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
import logging
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import platform
import matplotlib.pyplot as plt
import cv2

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def plt_result(result, img):
    plt.figure(figsize=(20,20))
    plt.imshow(img)
    plt.axis('off')
    show_anns(result)

def grayscale_to_rgb(grayscale_imgs):
    # Initialize an empty array to hold the RGB images
    # The new shape will have an extra dimension for the channels at the end
    rgb_imgs = np.zeros(
        (grayscale_imgs.shape[0], grayscale_imgs.shape[1], 3),
        dtype=np.uint8,
    )

    grayscale_img_uint8 = (grayscale_imgs / 256).astype(np.uint8)
    rgb_imgs = np.stack((grayscale_img_uint8,) * 3, axis=-1)

    return rgb_imgs

def get_top_brightness_points(np_image, percentage):
    # Convert the image to grayscale by averaging the RGB channels
    grayscale_image = np.mean(np_image, axis=2)
    
    # Determine the threshold for the top percentage of brightness
    threshold_value = np.percentile(grayscale_image, 100 - percentage)
    
    # Create a mask for the brightest areas
    bright_areas_mask = grayscale_image > threshold_value
    return bright_areas_mask

def appl_exp(ori):
    exposure_factor = 18
    exposed_image_array = np.clip(ori * exposure_factor, 0, 65535)
    return exposed_image_array

target = pathlib.Path("./201-250_Hong/MAX_CTLa_w1-359 DAPI_s012.tif")
img_org = Image.open(target)
img = cv2.cvtColor(np.array(img_org), cv2.COLOR_GRAY2RGB).astype(np.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"

CP = "fish_v2.1.pth"
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

fish = SamModel(config=model_config)
fish.load_state_dict(torch.load(pathlib.Path("./checkpoints/"+CP), map_location=torch.device('cpu')))
fish.to(device)


idx = 10
dataset = Dataset.load_from_disk(pathlib.Path("./data"))
example_image = dataset[idx]["image"]
np_image = np.array(example_image)

# Define grid parameters
array_size = 256
grid_size = 3
x = np.linspace(0, array_size-1, grid_size)
y = np.linspace(0, array_size-1, grid_size)
xv, yv = np.meshgrid(x, y)
xv_list = xv.tolist()
yv_list = yv.tolist()
input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]

# Get top brightness points mask
percentage = 30  # Percentage for top brightest points
bright_areas_mask = get_top_brightness_points(np_image, percentage)

# Filter the input points to keep only the brightest ones
filtered_input_points = []
for row in input_points:
    filtered_row = []
    for point in row:
        if bright_areas_mask[point[1], point[0]]:
            filtered_row.append(point)
        else:
            filtered_row.append([-1, -1])  # Placeholder for non-bright points
    filtered_input_points.append(filtered_row)

# Convert filtered_input_points to a numpy array with the same shape as input_points
filtered_input_points = np.array(filtered_input_points)

# Print some results to verify
print("Input points:")
print(np.array(input_points))
print("\nFiltered input points:")
print(filtered_input_points)

# Visualize the original image and the filtered grid points
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(appl_exp(np_image))

plt.subplot(1, 2, 2)
plt.title("Brightest Points Grid")
plt.imshow(appl_exp(np_image))
for row in filtered_input_points:
    for point in row:
        if point[0] != -1 and point[1] != -1:  # Only plot valid points
            plt.plot(point[0], point[1], 'ro')

plt.show()

input_points = torch.tensor(filtered_input_points).view(1, 1, grid_size*grid_size, 2)
inputs = processor(example_image, input_points=input_points, return_tensors="pt")

inputs = {k: v.to(device) for k, v in inputs.items()}
fish.eval()


with torch.no_grad():
  outputs = fish(**inputs, multimask_output=False)

single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
single_patch_prediction = (single_patch_prob > 0.5).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(appl_exp(np.array(example_image)), cmap='gray')
axes[0].set_title("Image")

axes[1].imshow(single_patch_prob)
axes[1].set_title("Probability Map")

axes[2].imshow(single_patch_prediction, cmap='gray')
axes[2].set_title("Prediction")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.show()