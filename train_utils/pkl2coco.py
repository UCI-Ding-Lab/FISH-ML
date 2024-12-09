from __future__ import annotations
import pprint
import pathlib
import pickle
import os
import numpy as np
from PIL import Image
import time
import json

class bundle():
    def __init__(self) -> None:
        self.path: pathlib.Path = None
        self.bbox: list[list] = None
        self.mask: list[np.ndarray] = None

class Pkl2coco():
    @staticmethod
    def getDataFromPkl(file_path):
        """
        Extract all the data from a pickle file and return them as lists.
        """
        print(os.getcwd())
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        
        boxes = [] # List to store bounding boxes
        image_paths = []
        names = []
        segmentations = []
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            
            for b in sorted(data, key=lambda x: x.path):
                bund: bundle = b
                bund_path = pathlib.Path(bund.path)
                boxes.append(bund.bbox) if bund.bbox is not None else boxes.append([])
                image_paths.append(bund.path)
                names.append(bund_path.name)
                segmentations.append(bund.segment) if bund.segment is not None else segmentations.append([])
            
        except Exception as e:
            raise Exception(f"Failed to load data from {file_path}: {e}")
        return names, image_paths, boxes, segmentations

@staticmethod
def create_coco_json(data):
    """
    Create a COCO JSON file from the data (returned by getDataFromPkl).
    """
    # COCO JSON format init:
    json = {
        "info": [], 
        "licenses": [], 
        "images": [], 
        "annotations": [], 
        "categories": []
    }
    # Add the info of this dataset:
    json["info"].append({"year": time.localtime().tm_year
                            ,"version": "1.0"
                            ,"description": "FISH Dataset"
                            ,"contributor": "Huizhi Zhang, Yicheng Ding"
                            ,"url": "https://github.com/UCI-Ding-Lab/FISH-ML"
                            ,"date_created": time.asctime()})
    
    # Add the categories of this dataset (things to be detected):
    json["categories"].append({
                "id": 1, 
                "name": "white cell nucleus", 
                "supercategory": "cell nucleus"})
    
    # Add the images and annotations:
    for i, (name, shape, box, segs) in enumerate(zip(data["image_name"]
                                                        ,data["image_shape"]
                                                        ,data["bbox"]
                                                        ,data["segmentations"])):
        img_id = i + 1
        json["images"].append({
            "id": img_id, 
            "file_name": name, 
            "height": shape[0],
            "width": shape[1]
        })
        for j, (b, s) in enumerate(zip(box, segs)):
            json["annotations"].append({
                "id": j + 1,
                "image_id": img_id,
                "category_id": 1,
                # "segmentation": s,
                "bbox": b,
                "area": b[2] * b[3]
                # "iscrowd": 0 if len(s) == 0 else 1
            })
    return json

if __name__ == "__main__":
    names, img_paths, boxes, segs = Pkl2coco.getDataFromPkl("/Users/huizhizhang/FISH-ML/demotifs/progress.pkl")
    images = [np.array(Image.open(path)) for path in img_paths]
    images_shape = [img.shape for img in images]
    data = {"image_name": names, "image_path": img_paths, "image_shape": images_shape, "bbox": boxes, "segmentations": segs}
    json_dict = create_coco_json(data)
    pprint.pprint(json_dict)
    export_path = "/Users/huizhizhang/FISH-ML/demotifs/train.json"
    with open(export_path, "w") as f:
        json.dump(json_dict, f)
    print(f"Exported COCO JSON file to {export_path}")
    
    