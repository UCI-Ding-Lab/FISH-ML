# Standard Library Imports
import pathlib
import logging
import configparser
from typing import Tuple

# Third-Party Imports
import numpy as np
import torch
from torchvision.ops import box_convert
from transformers import SamModel, SamConfig, SamProcessor
from PIL import Image
import cv2

# Local Application/Library Specific Imports
import groundingdino.datasets.transforms as T
import groundingdino.util.inference as dino
import groundingdino

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[1;91m',
        'RESET': '\033[0m'
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        record.msg = f"{log_color}{record.msg}{reset_color}"
        return super().format(record)

class Fish():
    def __init__(self,config: pathlib.Path) -> None:
        self.setup__config(config)
        self.setup__logger()
        self.setup__asset()
        self.setup__ai()
        self.finetune = self.Finetune(self)
    
    def setup__config(self, config):
        self.config = configparser.ConfigParser()
        self.config.read(config)
    def setup__logger(self):
        self.logger = logging.getLogger('fishcore')
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = ColoredFormatter("[%(asctime)s][%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            if "transformers" in logger.name.lower():
                logger.setLevel(logging.ERROR)
    def setup__asset(self):
        self.asset_folder_path = pathlib.Path(self.config["general"]["asset_folder_path"])
        self.supported_version = self.config["general"]["supported_version"].split(",")
        self.model_version = None
        self.model_path = None
    def setup__ai(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.gdino_config = pathlib.Path(groundingdino.__path__[0]) / self.config["dino"]["config"]
        self.gdino_weights = pathlib.Path(groundingdino.__path__[0]) / self.config["dino"]["weights"]
        self.gdino_model = dino.load_model(self.gdino_config, self.gdino_weights)

    
    def set_model_version(self,v):
        if v in self.supported_version:
            self.model_version = v
            self.model_path = self.asset_folder_path / "model" / f"fish_v{v}.pth"
            if self.model:
                del self.model
            self.model = SamModel(config=self.model_config)
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
            self.model.to(self.device)
            self.logger.info(f"CHECK: Model version {v} loaded")
        else:
            self.logger.error(f"CHECK: Version {v} is not supported")
    
    @staticmethod
    def helper__hdr2Rgb(hdr_image: np.ndarray, dynamic_range: int) -> np.ndarray:
        scale_factor = 255 / dynamic_range
        scaled_image = (hdr_image * scale_factor).astype(np.uint8)
        rgb_image = np.stack((scaled_image,) * 3, axis=-1)
        return rgb_image
    @staticmethod
    def helper__hdr2RgbNorm(hdr_image: np.ndarray, brightness_factor: int) -> np.ndarray:
        img_normalized = cv2.normalize(hdr_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
        return np.clip(img_rgb * brightness_factor, 0, 255).astype(np.uint8)
    @staticmethod
    def helper__rectArea(rect):
        x1, y1, x2, y2 = rect
        return (x2 - x1) * (y2 - y1)
    @staticmethod
    def helper__computeIntersectionArea(rect1, rect2):
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        if xi1 < xi2 and yi1 < yi2:
            return (xi2 - xi1) * (yi2 - yi1)
        else:
            return 0
        
    # @staticmethod
    # def helper__filterAlgorithm(image_source: np.ndarray, boxes: torch.Tensor) -> list:
    #     h, w, _ = image_source.shape
    #     boxes = boxes * torch.Tensor([w, h, w, h])
    #     xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.uint16).tolist()
    #     bboxes = []
    #     for box in xyxy:
    #         min_x, min_y, max_x, max_y = box
    #         area = (max_x - min_x) * (max_y - min_y)
    #         if area < 800 or area > 100000:
    #             continue
    #         if max_x - min_x < 40 or max_y - min_y < 40:
    #             continue
    #         bboxes.append(box)
    #     rects = np.array(bboxes)
    #     N = len(rects)
    #     to_delete = set()
    #     areas = np.array([Fish.helper__rectArea(rect) for rect in rects])
    #     for i in range(N):
    #         for j in range(i + 1, N):
    #             if j in to_delete:
    #                 continue
    #             intersection_area = Fish.helper__computeIntersectionArea(rects[i], rects[j])
    #             if intersection_area >= 0.9 * min(areas[i], areas[j]):
    #                 if areas[i] > areas[j]:
    #                     to_delete.add(i)
    #                 else:
    #                     to_delete.add(j)
    #     filtered_rects = [rect for k, rect in enumerate(rects) if k not in to_delete]
    #     return np.array(filtered_rects).tolist()
    
    @staticmethod
    def helper__filterAlgorithm(image_source: np.ndarray, boxes: torch.Tensor) -> list:
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.uint16).tolist()
        bboxes = []
        
        # Filtering bounding boxes based on area and size constraints
        for box in xyxy:
            min_x, min_y, max_x, max_y = box
            area = (max_x - min_x) * (max_y - min_y)
            
            # Skip boxes with area too small or too large
            if area < 800 or area > 100000:
                continue
            
            # Skip boxes with width or height too small
            if max_x - min_x < 40 or max_y - min_y < 40:
                continue
            
            # Skip boxes near the image edges (within 4 pixels)
            if min_x <= 4 or min_y <= 4 or max_x >= w - 4 or max_y >= h - 4:
                continue
            
            bboxes.append(box)
        
        rects = np.array(bboxes)
        N = len(rects)
        to_delete = set()
        areas = np.array([Fish.helper__rectArea(rect) for rect in rects])
        
        # Overlap filtering logic
        for i in range(N):
            for j in range(i + 1, N):
                if j in to_delete:
                    continue
                intersection_area = Fish.helper__computeIntersectionArea(rects[i], rects[j])
                if intersection_area >= 0.9 * min(areas[i], areas[j]):
                    if areas[i] > areas[j]:
                        to_delete.add(i)
                    else:
                        to_delete.add(j)
        
        # Final list of filtered bounding boxes
        filtered_rects = [rect for k, rect in enumerate(rects) if k not in to_delete]
        return np.array(filtered_rects).tolist()
    @staticmethod
    def helper__filterAlgorithm__big(image_source: np.ndarray, boxes: torch.Tensor, input_points: np.ndarray) -> list:
        h, w, _ = image_source.shape
        boxes = boxes * torch.tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        final_boxes = []
        for center in input_points:
            cx, cy = center
            min_dist = float('inf')
            best_box = None
            for box in xyxy:
                x1, y1, x2, y2 = box
                box_cx = (x1 + x2) / 2
                box_cy = (y1 + y2) / 2
                dist = np.sqrt((cx - box_cx)**2 + (cy - box_cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_box = box
            if best_box is not None:
                final_boxes.append(best_box.astype(np.uint16).tolist())

        return final_boxes

    @staticmethod
    def helper__imageTransform4Dino(img: np.ndarray) -> Tuple[np.array, torch.Tensor]:
        transform = T.Compose([T.RandomResize([800], max_size=1333),
                               T.ToTensor(),
                               T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        image_source = Image.fromarray(Fish.helper__hdr2RgbNorm(img, 1))
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed
    
    @staticmethod
    def dino_bbox(gdino_model, img: np.ndarray) -> dict:
        
        image_source, image = Fish.helper__imageTransform4Dino(img)

        TEXT_PROMPT = "white flower"
        BOX_TRESHOLD = 0.07
        TEXT_TRESHOLD = 0.05

        boxes, logits, phrases = dino.predict(
            model=gdino_model, 
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD,
            device="cpu"
        )
        
        finalized_bboxes = Fish.helper__filterAlgorithm(image_source, boxes)
        return finalized_bboxes
    
    @staticmethod
    def dino_bbox_big(gdino_model, img: np.ndarray, input_points: np.ndarray) -> dict:
        
        image_source, image = Fish.helper__imageTransform4Dino(img)

        TEXT_PROMPT = "white flower"
        BOX_TRESHOLD = 0.07
        TEXT_TRESHOLD = 0.05

        boxes, logits, phrases = dino.predict(
            model=gdino_model, 
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD,
            device="cpu"
        )
        
        finalized_bboxes = Fish.helper__filterAlgorithm__big(image_source, boxes, input_points)
        return finalized_bboxes
    
    def AppIntDINOwrapper(self, img: np.ndarray) -> list[list]:
        return Fish.dino_bbox(self.gdino_model, img)
    
    def AppIntDINOwrapperB(self, img: np.ndarray, input_points: np.ndarray) -> list[list]:
        return Fish.dino_bbox_big(self.gdino_model, img, input_points)
 
    class Finetune():
        def __init__(self, fish):
            self.fish: Fish = fish

        def predict(self, img: np.ndarray, bbox: list[list]=None):
            if not self.fish.model:
                self.fish.logger.error("PRED: Model not loaded")
                return
            if len(img.shape) != 2:
                self.fish.logger.error("PRED: Image should be in grayscale")
                return
            
            if not bbox:
                raw = Fish.dino_bbox(self.fish.gdino_config, self.fish.gdino_weights, img)
                self.fish.logger.info(f"CLU: found {len(raw['bright_points'])} bright points")
                self.fish.logger.info(f"CLU: found {len(raw['bboxes'])} bounding boxes")
            else:
                raw = {"bright_points": "OF", "clusters": "OF", "bboxes": bbox}
            
            img = Fish.helper__hdr2Rgb(img, int(self.fish.config["predict"]["dynamic_range"]))
            
            masks = None
            inputs = self.fish.processor(img,
                                    input_boxes=[[bbox for bbox in raw["bboxes"]]],
                                    return_tensors="pt",
                                    do_convert_rgb=False).to(self.fish.device)
            self.fish.model.eval()
            with torch.no_grad():
                outputs: dict = self.fish.model(**inputs, multimask_output=False)
                # !!!pred_masks shape: torch.Size([1, bbox, 1, 256, 256]) be4 squeeze
            self.fish.logger.info(f"PRED: generated {outputs.pred_masks.shape[1]} masks")
            
            masks: list = self.fish.processor.image_processor.post_process_masks(masks=outputs.pred_masks.cpu(),
                                                                                 original_sizes=inputs["original_sizes"].cpu(),
                                                                                 reshaped_input_sizes=inputs["reshaped_input_sizes"].cpu(),
                                                                                 mask_threshold=float(self.fish.config["predict"]["mask_threshold"]))  
            masks = masks[0].squeeze(1).numpy().astype(np.uint8)
            # masks (n, 2048, 2048)
            
            return raw, masks
        
        def AppIntPREDICTwrapper(self, img: np.ndarray, bbox: list[list]=None) -> np.ndarray:
            return self.predict(img, bbox)[1]
        