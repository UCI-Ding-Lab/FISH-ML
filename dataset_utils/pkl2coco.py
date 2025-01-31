from __future__ import annotations
import pprint
import pathlib
import pickle
import os
import re
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
                match = re.search(r"assets.*", bund.path)
                if match:
                    result = match.group()
                    print(result)
                result = result.replace("\\", "/") # windows to mac path conversion
                name = pathlib.Path(result).name.replace(".tif", ".jpg")
                boxes.append(bund.bbox) if bund.bbox is not None else boxes.append([])
                image_paths.append(result)
                names.append(name)
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
    
    # Add the licenses of this dataset:
    json["licenses"].append({"url": "https://www.ding.eng.uci.edu/"
                                ,"id": 1
                                ,"name": "FISH Lab License"})
    # Add the categories of this dataset (things to be detected):
    json["categories"].append({
                "id": 1, 
                "name": "white cell nucleus"})
    
    j = 0 # Annotation ID
    # Add the images and annotations:
    for i, (name, shape, box, segs) in enumerate(zip(data["image_name"]
                                                        ,data["image_shape"]
                                                        ,data["bbox"]
                                                        ,data["segmentations"])):
        json["images"].append({
            "id": i,
            "license": 1,
            "file_name": name, 
            "height": shape[0],
            "width": shape[1]
        })
        for _, b in enumerate(box):
            # convert b from [min_x, min_y, max_x, max_y] to [min_x, min_y, width, height]
            b = [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]
            json["annotations"].append({
                "id": j,
                "image_id": i,
                "category_id": 1,
                "segmentation": [],
                "bbox": b,
                "area": b[2] * b[3],
                "iscrowd": 0
            })
            j += 1
        i = i + 1
    return json

if __name__ == "__main__":
    names, img_paths, boxes, segs = Pkl2coco.getDataFromPkl("/Users/huizhizhang/FISH-ML/assets/demotifs/testpkl1.pkl")
    images = [np.array(Image.open(path)) for path in img_paths]
    images_shape = [img.shape for img in images]
    data = {"image_name": names, "image_path": img_paths, "image_shape": images_shape, "bbox": boxes, "segmentations": segs}
    json_dict = create_coco_json(data)
    pprint.pprint(json_dict)
    export_path = "/Users/huizhizhang/FISH-ML/assets/demotifs/train.json"
    with open(export_path, "w") as f:
        json.dump(json_dict, f)
    print(f"Exported COCO JSON file to {export_path}")
    
    