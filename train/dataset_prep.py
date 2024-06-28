import numpy as np
import pathlib
import os
import logging
from datasets import Dataset
from PIL import Image
from patchify import patchify
import scipy.io

def get_masks_from_mat(mat_file, dir_name):
    def load_matlab_data(mat_file):
        file = pathlib.Path(mat_file)
        mat = scipy.io.loadmat(file)
        return mat
    def convert(all_frames):
        converted: list[dict] = []
        for i in range(all_frames.shape[0]):
            cur = dict()
            # use try catch to avoid error on edge cases
            try:
                for j in range(100):
                    cur[j] = all_frames[i][0, 0][j][0]
            except IndexError:
                pass
            converted.append(cur)
        return converted
    def getCellsField(singleFrame, cellsInfoIndex=12):
        keys = list(singleFrame.keys())
        cell_data = None
        for i in range(len(keys)):
            buffer = singleFrame[keys[i]]
            if buffer.shape != () and buffer.shape[0] != 1:
                cell_data = buffer

        if not isinstance(cell_data, list) and not hasattr(cell_data, "__iter__"):
            return []

        cells = []
        for i in range(len(cell_data)):
            info = []
            try:
                cell = (
                    cell_data[i][0, 0]
                    if hasattr(cell_data[i], "size") and cell_data[i].size > 0
                    else None
                )
                if cell is None:
                    continue

                for j in range(min(cellsInfoIndex, len(cell))):
                    info.append(cell[j])
            except (TypeError, IndexError) as e:
                continue

            if info:
                cells.append(info)
        return cells
    def get_all_cells_pos(all_cells_in_mask_1):
        all_cell_positions = []

        for cell in all_cells_in_mask_1:
            cell_position = cell[1]
            y_pos, x_pos = cell_position[0][0], cell_position[0][1]
            all_cell_positions.append((y_pos, x_pos))

        return all_cell_positions
    def patch_cells_into_picture_exc(all_cells_in_mask_1, all_cell_positions):
        picture = np.zeros((2048, 2048), dtype=int)

        for cell_mask, pos in zip(all_cells_in_mask_1, all_cell_positions):
            mask = cell_mask[0]
            if np.all(mask == 0):
                continue

            y_pos, x_pos = pos
            mask_height, mask_width = mask.shape
            binary_mask = mask > 0
            y_end, x_end = min(y_pos + mask_height, 2048), min(x_pos + mask_width, 2048)
            picture[y_pos:y_end, x_pos:x_end] |= binary_mask[
                : y_end - y_pos, : x_end - x_pos
            ]

        if np.all(picture == 0):
            return None

        return picture
    
    mat = load_matlab_data(mat_file)
    all_frames = mat[dir_name][0]  # all the masks corresponding to 101 tifs
    all_frames_converted = convert(all_frames)
    all_masks = []
    mask_index = []
    for i in range(len(all_frames_converted)):
        mask = all_frames_converted[i]
        all_cells_in_mask = getCellsField(mask)
        all_cell_positions = get_all_cells_pos(all_cells_in_mask)
        picture = patch_cells_into_picture_exc(all_cells_in_mask, all_cell_positions)
        all_masks.append(picture) if picture is not None else None
        mask_index.append(i) if picture is not None else None

    all_masks = np.array(all_masks)
    return all_masks, mask_index

def process_tiff_images(folder_path, exposure_factor=30):
    folder = pathlib.Path(folder_path)
    processed_images = []
    file_names = os.listdir(folder)
    tiff_files = [
        file for file in file_names if file.endswith(".tif") or file.endswith(".tiff")
    ]
    tiff_files.sort()
    for file_path in tiff_files:
        with Image.open(folder_path + "/" + file_path) as img:
            image_array = np.array(img)
            exposed_image_array = np.clip(image_array * exposure_factor, 0, 65535)
            processed_images.append(exposed_image_array)

    return processed_images

def grayscale_to_rgb(grayscale_imgs):
    rgb_imgs = np.zeros(
        (grayscale_imgs.shape[0], grayscale_imgs.shape[1], grayscale_imgs.shape[2], 3),
        dtype=np.uint8,
    )

    for i in range(grayscale_imgs.shape[0]):
        grayscale_img_uint8 = (grayscale_imgs[i] / 256).astype(np.uint8)
        rgb_imgs[i] = np.stack((grayscale_img_uint8,) * 3, axis=-1)

    return rgb_imgs

def rgb_to_grayscale(images):
    grayscale_imgs = np.zeros(
        (images.shape[0], images.shape[1], images.shape[2]), dtype=np.uint8
    )

    for i in range(images.shape[0]):
        pil_img = Image.fromarray(images[i].astype("uint8"), "RGB")
        gray_img = pil_img.convert("L")
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
    data_path = None
    dataset_path = None
    
    logging.info(f"Patch size: {patch_size}, Step: {step}, Patch Overlap: {overlap}")
    logging.info(f"Exposure factor: {exposure_factor}")

    logging.info(f"Processing images and masks...")
    images_2 = process_tiff_images(data_path / "51-100_Hong", exposure_factor=exposure_factor)
    images_3 = process_tiff_images(data_path / "151-200_Hong", exposure_factor=exposure_factor)
    images_4 = process_tiff_images(data_path / "201-250_Hong", exposure_factor=exposure_factor)
    images = np.concatenate((images_2, images_3, images_4))
    filtered_masks_2, valid_indices_2 = get_masks_from_mat(
        data_path / "51-100_Hong" / "51-100_finished.mat", "Tracked"
    )
    filtered_masks_3, valid_indices_3 = get_masks_from_mat(
        data_path / "151-200_Hong" / "151-200_finished.mat", "Tracked_151200"
    )
    filtered_masks_4, valid_indices_4 = get_masks_from_mat(
        data_path / "201-250_Hong" / "201-250_finished.mat", "Tracked_201250"
    )
    filtered_masks = np.concatenate((filtered_masks_2, filtered_masks_3, filtered_masks_4))
    valid_indices = np.concatenate((valid_indices_2, valid_indices_3, valid_indices_4))
    filtered_images = np.array(
        [images[i] for i in valid_indices]
    )  # filters out the images that don't have masks
    grayscale_images = filtered_images
    logging.info(f"Done")
    
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
    valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
    images_ready = images[valid_indices]
    masks_ready = masks[valid_indices]
    logging.info(f"Done")

    logging.info(f"Converting grayscale images to RGB...")
    rgb_images_ready = grayscale_to_rgb(images_ready)
    logging.info(f"Done")

    logging.info(f"Forming dataset dictionary...")
    dataset_dict = {
        "image": [Image.fromarray(img.astype(np.uint8)) for img in rgb_images_ready],
        "label": [Image.fromarray(mask) for mask in masks_ready],
    }
    dataset = Dataset.from_dict(dataset_dict)
    logging.info(f"Done")

    logging.info(f"Saving dataset to disk...")
    dataset.save_to_disk(dataset_path)
    logging.info(f"Done")