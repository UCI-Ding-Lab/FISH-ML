import numpy as np
import pathlib
import os

import logging

from datasets import Dataset
from PIL import Image
from patchify import patchify
import fishLoader as fish
import matplotlib.pyplot as plt


def process_tiff_images(folder_path, exposure_factor=30):
    # Create a Path object for the folder
    folder = pathlib.Path(folder_path)

    # List to store the processed image arrays
    processed_images = []

    # Collect all TIFF files, considering both '.tif' and '.tiff' extensions
    file_names = os.listdir(folder)
    tiff_files = [
        file for file in file_names if file.endswith(".tif") or file.endswith(".tiff")
    ]
    tiff_files.sort()

    # Iterate over each sorted TIFF file
    for file_path in tiff_files:
        with Image.open(folder_path + "/" + file_path) as img:
            # Convert the image to a NumPy array
            image_array = np.array(img)

            # Adjust exposure and clip values
            exposed_image_array = np.clip(image_array * exposure_factor, 0, 65535)

            # Append the processed image array to the list
            processed_images.append(exposed_image_array)

    return processed_images

def grayscale_to_rgb(grayscale_imgs):
    # Initialize an empty array to hold the RGB images
    # The new shape will have an extra dimension for the channels at the end
    rgb_imgs = np.zeros(
        (grayscale_imgs.shape[0], grayscale_imgs.shape[1], grayscale_imgs.shape[2], 3),
        dtype=np.uint8,
    )

    for i in range(grayscale_imgs.shape[0]):
        # Duplicate the grayscale image data across three channels
        grayscale_img_uint8 = (grayscale_imgs[i] / 256).astype(np.uint8)
        rgb_imgs[i] = np.stack((grayscale_img_uint8,) * 3, axis=-1)

    return rgb_imgs

def rgb_to_grayscale(images):
    # Initialize an empty array to hold the grayscale images
    grayscale_imgs = np.zeros(
        (images.shape[0], images.shape[1], images.shape[2]), dtype=np.uint8
    )

    for i in range(images.shape[0]):
        # Convert each RGB image to a PIL Image, then convert to grayscale
        pil_img = Image.fromarray(images[i].astype("uint8"), "RGB")
        gray_img = pil_img.convert("L")
        # Convert back to numpy array and store in the grayscale_imgs array
        grayscale_imgs[i] = np.array(gray_img)

    return grayscale_imgs

if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
    )
    patch_size = 256
    overlap = 0.5
    step = int(patch_size * (1 - overlap))
    exposure_factor = 1
    logging.info(f"Patch size: {patch_size}, Step: {step}, Patch Overlap: {overlap}")
    logging.info(f"Exposure factor: {exposure_factor}")


    logging.info(f"Processing images and masks...")
    images_2 = process_tiff_images("51-100_Hong", exposure_factor=exposure_factor)
    images_3 = process_tiff_images("151-200_Hong", exposure_factor=exposure_factor)
    images_4 = process_tiff_images("201-250_Hong", exposure_factor=exposure_factor)
    images = np.concatenate((images_2, images_3, images_4))
    filtered_masks_2, valid_indices_2 = fish.get_masks_from_mat(
        "51-100_Hong/51-100_finished.mat", "Tracked"
    )
    filtered_masks_3, valid_indices_3 = fish.get_masks_from_mat(
        "151-200_Hong/151-200_finished.mat", "Tracked_151200"
    )
    filtered_masks_4, valid_indices_4 = fish.get_masks_from_mat(
        "201-250_Hong/201-250_finished.mat", "Tracked_201250"
    )
    filtered_masks = np.concatenate((filtered_masks_2, filtered_masks_3, filtered_masks_4))
    valid_indices = np.concatenate((valid_indices_2, valid_indices_3, valid_indices_4))
    filtered_images = np.array(
        [images[i] for i in valid_indices]
    )  # filters out the images that don't have masks
    grayscale_images = filtered_images
    logging.info(f"Done")
    """
    Processing part checked, image array shape is (150, 2048, 2048), 150 images in total
    2024 May 16
    """


    logging.info(f"Patching Images...")
    all_img_patches = []
    for img in range(grayscale_images.shape[0]):
        large_image = grayscale_images[img]
        patches_img = patchify(
            large_image, (patch_size, patch_size), step=step
        )  # Step=256 for 256 patches means no overlap
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :]
                all_img_patches.append(single_patch_img)
    images = np.array(all_img_patches)
    logging.info(f"Done")

    logging.info(f"Patching Masks...")
    all_mask_patches = []
    for img in range(filtered_masks.shape[0]):
        large_mask = filtered_masks[img]
        patches_mask = patchify(
            large_mask, (patch_size, patch_size), step=step
        )  # Step=256 for 256 patches means no overlap
        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i, j, :, :]
                single_patch_mask = (single_patch_mask).astype(np.uint8)
                all_mask_patches.append(single_patch_mask)
    masks = np.array(all_mask_patches)
    logging.info(f"Done")

    logging.info(f"Filtering out empty masks...")
    # Create a list to store the indices of non-empty masks
    valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
    # Filter the image and mask arrays to keep only the non-empty pairs
    images_ready = images[valid_indices]
    masks_ready = masks[valid_indices]
    logging.info(f"Done")

    logging.info(f"Converting grayscale images to RGB...")
    rgb_images_ready = grayscale_to_rgb(images_ready)
    logging.info(f"Done")

    logging.info(f"Forming dataset dictionary...")
    # Convert the NumPy arrays to Pillow images, storing them in a dictionary
    dataset_dict = {
        "image": [Image.fromarray(img.astype(np.uint8)) for img in rgb_images_ready],
        "label": [Image.fromarray(mask) for mask in masks_ready],
    }
    # Create the dataset using the datasets.Dataset class
    dataset = Dataset.from_dict(dataset_dict)
    logging.info(f"Done")

    logging.info(f"Saving dataset to disk...")
    dataset.save_to_disk(pathlib.Path("./data"))
    logging.info(f"Done")