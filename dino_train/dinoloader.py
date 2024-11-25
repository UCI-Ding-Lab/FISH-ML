from __future__ import annotations
import pprint
import pathlib
import pickle
import torch
from torch.utils.data import Dataset as torchDataset
import os
import numpy as np
from PIL import Image
from datasets import Dataset
import logging

class bundle():
    def __init__(self) -> None:
        self.path: pathlib.Path = None
        self.bbox: list[list] = None
        self.mask: list[np.ndarray] = None

class Dinoloader:
    @staticmethod
    def getBoxesFromPkl(file_path):
        """
        Extract bounding boxes from a pickle file and return them as a list.
        """
        print(os.getcwd())
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        
        boxes = [] # List to store bounding boxes
        image_paths = []
        names = []
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            
            for b in sorted(data, key=lambda x: x.path):
                bund: bundle = b
                bund_path = pathlib.Path(bund.path)
                boxes.append(bund.bbox)
                image_paths.append(bund.path)
                names.append(bund_path.name)
            
        except Exception as e:
            raise Exception(f"Failed to load bounding boxes from {file_path}: {e}")
        return names, image_paths, boxes
    
class DinoDataset(torchDataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # TODO: unfinished
        return

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(levelname)s] %(message)s")
    names, img_paths, boxes = Dinoloader.getBoxesFromPkl("/Users/huizhizhang/FISH-ML/demotifs/progress.pkl")
    images = [np.array(Image.open(path)) for path in img_paths]
    data = {"name": names, "image": images, "bbox": boxes}
    pprint.pprint(data)
    # dataset = Dataset.from_dict(data)
    # dataset.save_to_disk("./assets/dataset/")
    

