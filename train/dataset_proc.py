import numpy as np
from torch.utils.data import Dataset as torchDataset

def get_bounding_box(y_pos, x_pos, mask):
    x_min, y_min = x_pos, y_pos
    y_max, x_max = min(y_pos + mask.shape[0], 2048), min(x_pos + mask.shape[1], 2048)
    bbox = [x_min, y_min, x_max, y_max]
    return bbox

def grayscale_to_rgb(grayscale_img) -> np.ndarray:
    rgb_img = np.zeros((2048, 2048, 3), dtype=np.uint8)
    img_uint8 = np.clip(grayscale_img//256, 0, 255).astype(np.uint8)
    rgb_img[:, :, 0] = img_uint8
    rgb_img[:, :, 1] = img_uint8
    rgb_img[:, :, 2] = img_uint8
    dynamic_exp_factor = 255.0 / np.max(rgb_img[:, :, 0])
    brightened_image = np.clip(rgb_img * dynamic_exp_factor, 0, 255).astype(np.uint8)
    return brightened_image

def fill(y_pos, x_pos, mask, ground_truth_mask):
    y_end, x_end = min(y_pos + mask.shape[0], 2048), min(x_pos + mask.shape[1], 2048)
    ground_truth_mask[y_pos:y_end, x_pos:x_end] = mask[:y_end-y_pos, :x_end-x_pos]
    return ground_truth_mask

class fishDataset(torchDataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = grayscale_to_rgb(np.array(item["image"]))
        bbox = []
        ground_truth_mask = np.zeros((len(item["masks"]), 2048, 2048), dtype=np.uint8)
        for ind, xy in enumerate(item["xy"]):
            bbox.append(get_bounding_box(*xy, np.array(item["masks"][ind])))
            g = fill(*xy, np.array(item["masks"][ind]), np.zeros((2048, 2048), dtype=np.uint8))
            ground_truth_mask[ind] = g

        inputs = self.processor(image, input_boxes=[bbox], return_tensors="pt")
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs