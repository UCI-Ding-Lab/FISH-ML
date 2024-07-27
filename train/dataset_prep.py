import numpy as np
import pathlib
import logging
from datasets import Dataset
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt

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

            if info and not is_noise(info[0]):
                cells.append(info)
        return cells
    def get_all_cells_pos(all_cells_in_mask_1):
        all_cell_positions = []

        for cell in all_cells_in_mask_1:
            cell_position = cell[1]
            y_pos, x_pos = cell_position[0][0], cell_position[0][1]
            all_cell_positions.append((y_pos, x_pos))

        return all_cell_positions
    def patch_cells(all_cells_in_mask_1, all_cell_positions) -> np.ndarray:
        """use cell positions and regional mask, make a set of complete mask

        Args:
            all_cells_in_mask_1 (_type_): _description_
            all_cell_positions (_type_): _description_

        Returns:
            np.ndarray: shape (n,2048,2048), n is ttl number of cells
        """
        single_cell_data = np.zeros((len(all_cells_in_mask_1), 2048, 2048))
        counter = 0
        for cell_mask, pos in zip(all_cells_in_mask_1, all_cell_positions):
            mask = cell_mask[0]
            if np.all(mask == 0):
                continue
            y_pos, x_pos = pos
            mask_height, mask_width = mask.shape
            binary_mask = mask > 0
            y_end, x_end = min(y_pos + mask_height, 2048), min(x_pos + mask_width, 2048)
            single_cell_data[counter, y_pos:y_end, x_pos:x_end] = binary_mask[
                : y_end - y_pos, : x_end - x_pos
            ]
            counter += 1
        if len(single_cell_data) == 0:
            return None
        return single_cell_data
    def is_noise(array, target_shape=(3,3)):
        array_size = np.prod(array.shape)
        target_size = np.prod(target_shape)
        return array_size < target_size
    
    mat = load_matlab_data(mat_file)
    all_frames = mat[dir_name][0]  # all the 101 frames corresponding to 101 tifs, each frame contains various ground truth masks
    all_frames_converted = convert(all_frames)
    d = {"name":[],"image":[],"xy":[],"masks":[]}
    for i in range(len(all_frames_converted)):
        mask = all_frames_converted[i]
        all_cells_in_mask = getCellsField(mask)
        all_cell_positions = get_all_cells_pos(all_cells_in_mask)
        picture: np.ndarray = patch_cells(all_cells_in_mask, all_cell_positions)
        if picture is not None:
            name = str(mask[0][0][0]) if not isinstance(mask[0], str) else str(mask[1][0][0])
            d["name"].append(name)
            d["xy"].append(all_cell_positions)
            d["masks"].append([i[0] for i in all_cells_in_mask])

    return d

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    data_path = pathlib.Path("./assets/tif")
    dataset_path = "./assets/dataset/"

    logging.info(f"Processing mat and masks...")
    masks_2 = get_masks_from_mat(data_path/"51-100_Hong"/"51-100_finished.mat", "Tracked")
    masks_3 = get_masks_from_mat(data_path/"151-200_Hong"/"151-200_finished.mat", "Tracked_151200")
    masks_4 = get_masks_from_mat(data_path/"201-250_Hong"/"201-250_finished.mat", "Tracked_201250")
    logging.info(f"Done")

    logging.info(f"Converting grayscale images to RGB...")
    image_2 = {key: np.array(Image.open(data_path/"51-100_Hong"/key)) for key in masks_2["name"]}
    image_3 = {key: np.array(Image.open(data_path/"151-200_Hong"/key)) for key in masks_3["name"]}
    image_4 = {key: np.array(Image.open(data_path/"201-250_Hong"/key)) for key in masks_4["name"]}
    image = {**image_2, **image_3, **image_4}
    for n in masks_2["name"]:
        masks_2["image"].append(image[n])
    for n in masks_3["name"]:
        masks_3["image"].append(image[n])
    for n in masks_4["name"]:
        masks_4["image"].append(image[n])
    data = {"name": masks_2["name"] + masks_3["name"] + masks_4["name"],
            "image": masks_2["image"] + masks_3["image"] + masks_4["image"],
            "xy": masks_2["xy"] + masks_3["xy"] + masks_4["xy"],
            "masks": masks_2["masks"] + masks_3["masks"] + masks_4["masks"]}
    logging.info(f"Done")

    logging.info(f"Forming dataset dictionary...")
    dataset = Dataset.from_dict(data)
    logging.info(f"Done")
    
    if False:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(grayscale_to_rgb(data["image"][0]))
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        y_pos, x_pos = data["xy"][0][0]
        mask[y_pos:y_pos+data["masks"][0][0].shape[0], x_pos:x_pos+data["masks"][0][0].shape[1]] = data["masks"][0][0]
        mask = mask > 0
        ax[1].imshow(mask)
        plt.show()
    
    logging.info(f"Saving dataset to disk...")
    dataset.save_to_disk(dataset_path)
    logging.info(f"Done")