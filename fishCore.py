# Standard Library Imports
import os
import json
import pathlib
import logging
import configparser
from typing import Tuple, List

# Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
import torch
from torchvision.ops import box_convert
from transformers import SamModel, SamConfig, SamProcessor, pipeline
from scipy.ndimage import binary_erosion
from PIL import Image

# Local Application/Library Specific Imports
import GroundingDINO.groundingdino.datasets.transforms as T
import GroundingDINO.groundingdino.util.inference as dino
from segment_anything import SamPredictor, sam_model_registry
import archive.demo as demo

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
        # config.ini
        self.config = configparser.ConfigParser()
        self.config.read(config)
        # logger
        self.logger = logging.getLogger('fishcore')
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = ColoredFormatter("[%(asctime)s][%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        # asset
        self.asset_folder_path = pathlib.Path(self.config["general"]["asset_folder_path"])
        self.supported_version = self.config["general"]["supported_version"].split(",")
        self.model_version = None
        self.model_path = None
        # ai
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        # module
        self.finetune = self.Finetune(self)
        self.sam = self.Sam(self)
        
        #self.assets_validation()
    
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
    
    def assets_validation(self):
        """
        unfinished
        """
        model_folder = self.asset_folder_path / "model"
        for v in self.supported_version:
            if not (model_folder / f"fish_{v}.pth").exists():
                self.logger.error(f"CHECK: Model file for version {v} is missing")
    
    @staticmethod
    def hdr_to_rgb(hdr_image: np.ndarray, dynamic_range: int) -> np.ndarray:
        scale_factor = 255 / dynamic_range
        scaled_image = (hdr_image * scale_factor).astype(np.uint8)
        rgb_image = np.stack((scaled_image,) * 3, axis=-1)
        return rgb_image
    
    @staticmethod
    def prepare_input(cls: type, img: np.ndarray) -> dict:
        def get_top_brightness_points(np_image, percentage):
            threshold_value = np.percentile(np_image, float(100 - percentage))
            bright_areas_mask = np_image > threshold_value
            return bright_areas_mask
        def cluster_points(points, eps, min_samples):
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            labels = db.labels_
            unique_labels = set(labels)
            clusters = [points[labels == k] for k in unique_labels if k != -1]
            return clusters
        def get_bounding_box(points):
            min_x = np.min(points[:, 1])
            max_x = np.max(points[:, 1])
            min_y = np.min(points[:, 0])
            max_y = np.max(points[:, 0])
            return [min_x, min_y, max_x, max_y]
        bright_areas_mask = get_top_brightness_points(img, float(cls.config["predict"]["brightest_percentage"]))
        bright_points = np.column_stack(np.where(bright_areas_mask))
        clusters = cluster_points(bright_points,
                                  eps=float(cls.config["predict"]["dbscan_eps"]),
                                  min_samples=int(cls.config["predict"]["dbscan_min_samples"]))
        grouped_input_points = [cluster.tolist() for cluster in clusters]
        bounding_boxes = [get_bounding_box(np.array(cluster)) for cluster in grouped_input_points]
        finalized_bboxes = Fish.finalize_bbox(bounding_boxes,
                                              int(cls.config["predict"]["bbox_expand_px"]),
                                              int(cls.config["predict"]["bbox_area_threshold"]))
        return {"bright_points": bright_points, "clusters": clusters, "bboxes": finalized_bboxes}
    
    @staticmethod
    def dino_bbox(config: str, weights: str, img: np.ndarray) -> dict:
        def load_image(img: np.ndarray) -> Tuple[np.array, torch.Tensor]:
            def grayscale_to_rgb(grayscale_img) -> np.ndarray:
                rgb_img = np.zeros((2048, 2048, 3), dtype=np.uint8)
                img_uint8 = np.clip(grayscale_img//256, 0, 255).astype(np.uint8)
                rgb_img[:, :, 0] = img_uint8
                rgb_img[:, :, 1] = img_uint8
                rgb_img[:, :, 2] = img_uint8
                dynamic_exp_factor = 255.0 / np.max(rgb_img[:, :, 0])
                brightened_image = np.clip(rgb_img * dynamic_exp_factor, 0, 255).astype(np.uint8)
                return brightened_image
            
            transform = T.Compose([T.RandomResize([800], max_size=1333),
                                   T.ToTensor(),
                                   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
            image_source = Image.fromarray(grayscale_to_rgb(img))
            image = np.asarray(image_source)
            image_transformed, _ = transform(image_source, None)
            return image, image_transformed
        
        def finalization(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> list:
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.uint16).tolist()
            bboxes = []
            for box in xyxy:
                min_x, min_y, max_x, max_y = box
                area = (max_x - min_x) * (max_y - min_y)
                if area < 200 or area > 160000:
                    continue
                bboxes.append(box)
            bboxes = demo.filter_rects(np.array(bboxes)).tolist()
            return bboxes
        
        model = dino.load_model(config, weights)
        image_source, image = load_image(img)

        TEXT_PROMPT = "white flower"
        BOX_TRESHOLD = 0.02
        TEXT_TRESHOLD = 0.25

        boxes, logits, phrases = dino.predict(
            model=model, 
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD,
            device="cpu"
        )
        
        finalized_bboxes = finalization(image_source, boxes, logits, phrases)
        return {"bright_points": "DINO", "clusters": "DINO", "bboxes": finalized_bboxes}
        
    @staticmethod
    def finalize_bbox(orignal_bbox: list, expand_fct: int, bbox_area_threshold: int):
        bboxes = []
        for box in orignal_bbox:
            min_x, min_y, max_x, max_y = box
            area = (max_x - min_x) * (max_y - min_y)
            if area < bbox_area_threshold:
                continue
            expanded_bbox = [np.clip(min_x - expand_fct, 0, 2048),
                             np.clip(min_y - expand_fct, 0, 2048),
                             np.clip(max_x + expand_fct, 0, 2048),
                             np.clip(max_y + expand_fct, 0, 2048)]
            bboxes.append(expanded_bbox)
        return bboxes
    
    @staticmethod
    def cal_iou(result: np.ndarray, gt: list, gt_pos: list) -> Tuple[float, np.ndarray]:
        """
        Calculate the Intersection over Union (IoU) between a predicted mask and multiple ground truth masks.

        Args:
            result (np.ndarray): Prediction mask of shape (2048, 2048) containing binary contour lines.
            gt (list): A list of small ground truth masks, each represented as a numpy array.
            gt_pos (list): A list of positions corresponding to the locations of the ground truth masks in the format (y_start, x_start).

        Returns:
            Tuple[float, np.ndarray]: The IoU score and the combined ground truth mask.
        """
        whole_gt = np.zeros(result.shape, dtype=np.uint8)
        for pos, gt in zip(gt_pos, gt):
            y_strt, x_strt = pos
            h, w = gt.shape
            y_end, x_end = min(y_strt + h, 2048), min(x_strt + w, 2048)
            whole_gt[y_strt:y_end, x_strt:x_end] |= gt[:y_end-y_strt, :x_end-x_strt]
        intersection = np.logical_and(result, whole_gt)
        union = np.logical_or(result, whole_gt)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score, whole_gt
    
    class Sam():
        def __init__(self, fish):
            self.fish: Fish = fish
        
        def predict(self, img: np.ndarray) -> np.ndarray:
            if len(img.shape) != 2:
                self.fish.logger.error("PRED: Image should be in grayscale")
                return
            
            raw = Fish.prepare_input(self.fish,img)
            self.fish.logger.info(f"CLU: found {len(raw['bright_points'])} bright points")
            self.fish.logger.info(f"CLU: found {len(raw['bboxes'])} bounding boxes")
            
            img = Fish.hdr_to_rgb(img, int(self.fish.config["predict"]["dynamic_range"]))
            masks = None
            
            sam = sam_model_registry["vit_h"](checkpoint=self.fish.asset_folder_path/"sam"/"sam_vit_h_4b8939.pth")
            predictor = SamPredictor(sam)
            sam.to(device="cpu")
            predictor.set_image(img)
            input_boxes = torch.tensor(raw["bboxes"], device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img.shape[:2])
            masks, _, _ = predictor.predict_torch(point_coords=None,
                                                  point_labels=None,
                                                  boxes=transformed_boxes,
                                                  multimask_output=False)
            masks = masks.squeeze(1).cpu().numpy().astype(np.uint8)
            # masks (n, 2048, 2048)
            
            self.fish.logger.debug(f"PRED: combination started")
            result = np.zeros(img.shape[:2], dtype=np.uint8)
            for ind, bbox in enumerate(raw["bboxes"]):
                min_x, min_y, max_x, max_y = bbox
                if max_x <= min_x or max_y <= min_y:
                    continue
                seg = masks[ind]
                seg = seg[min_y:max_y, min_x:max_x]
                if self.fish.config["predict"]["mask_erosion"].lower() == "true":
                    er_factor = int(self.fish.config["predict"]["erosion_factor"])
                    eroded_mask = binary_erosion(seg, structure=np.ones((er_factor, er_factor)))
                    seg = seg - eroded_mask
                result[min_y:max_y, min_x:max_x] = seg
            self.fish.logger.info("PRED: combination completed")
            
            return raw, masks, result
        
        def info(self, img: np.ndarray, gt: np.ndarray) -> None:
            raw, masks, result = self.predict(img)
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(img, cmap='gray')
            
            if self.fish.config["info"]["overlay"].lower() == "true":
                overlay = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
                overlay[result == 1] = np.array([255, 255, 0, 255])
                ax.imshow(overlay)
            if self.fish.config["info"]["bbox_preview"].lower() == "true":
                for bbox in raw["bboxes"]:
                    min_x, min_y, max_x, max_y = bbox
                    rect = patches.Rectangle((min_x, min_y),
                                             max_x - min_x,
                                             max_y - min_y,
                                             linewidth=float(self.fish.config["info"]["bbox_preview_line_width"]),
                                             edgecolor='r',
                                             facecolor='none')
                    ax.add_patch(rect)
                if self.fish.config["info"]["show_cluster"].lower() == "true":
                    for cluster in raw["clusters"]:
                        ax.scatter(cluster[:, 1], cluster[:, 0], s=1, c='b')
                if self.fish.config["info"]["show_brightest_point"].lower() == "true":
                    ax.scatter(raw["bright_points"][:, 1], raw["bright_points"][:, 0], s=0.2, c='g')
            plt.show()
    
    class Finetune():
        def __init__(self, fish):
            self.fish: Fish = fish

        def predict(self, img: np.ndarray):
            if not self.fish.model:
                self.fish.logger.error("PRED: Model not loaded")
                return
            if len(img.shape) != 2:
                self.fish.logger.error("PRED: Image should be in grayscale")
                return
            
            # raw = Fish.prepare_input(self.fish, img)
            raw = Fish.dino_bbox(self.fish.config["dino"]["config"],
                                 self.fish.config["dino"]["weights"],
                                 img)
            self.fish.logger.info(f"CLU: found {len(raw['bright_points'])} bright points")
            self.fish.logger.info(f"CLU: found {len(raw['bboxes'])} bounding boxes")
            
            img = Fish.hdr_to_rgb(img, int(self.fish.config["predict"]["dynamic_range"]))
            
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
            
            self.fish.logger.debug(f"PRED: combination started")
            line_result = np.zeros(img.shape[:2], dtype=np.uint8)
            area_result = np.zeros(img.shape[:2], dtype=np.uint8)
            for ind, bbox in enumerate(raw["bboxes"]):
                min_x, min_y, max_x, max_y = bbox
                if max_x <= min_x or max_y <= min_y:
                    continue
                seg = masks[ind]
                seg = seg[min_y:max_y, min_x:max_x]
                area_result[min_y:max_y, min_x:max_x] = seg
                if self.fish.config["info"]["mask_erosion"].lower() == "true":
                    er_factor = int(self.fish.config["predict"]["erosion_factor"])
                    eroded_mask = binary_erosion(seg, structure=np.ones((er_factor, er_factor)))
                    contour_seg = seg - eroded_mask
                    line_result[min_y:max_y, min_x:max_x] = contour_seg
            self.fish.logger.info("PRED: combination completed")
            
            return raw, masks, line_result, area_result
            # line_result is the prediction with contour only, area_result is concrete prediction
        
        def info(self, img: np.ndarray, gt: list, gt_pos: list) -> None:
            raw, masks, line_result, area_result = self.predict(img)
            _, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(img, cmap='gray')
            if self.fish.config["info"]["mask_erosion"].lower() == "true":
                result = line_result
            else:
                result = area_result
            
            if self.fish.config["info"]["overlay"].lower() == "true":
                overlay = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
                overlay[result == 1] = np.array([255, 255, 0, 255])
                ax.imshow(overlay)
            if self.fish.config["info"]["show_iou"].lower() == "true":
                iou, _ = Fish.cal_iou(area_result, gt, gt_pos)
                ax.text(0.05, 0.95, f'IoU: {iou:.2f}', transform=ax.transAxes
                                                    , fontsize=self.fish.config["info"]["iou_font"].lower()
                                                    , verticalalignment='top'
                                                    , bbox=dict(facecolor='white', alpha=0.5))
            if self.fish.config["info"]["bbox_preview"].lower() == "true":
                for bbox in raw["bboxes"]:
                    min_x, min_y, max_x, max_y = bbox
                    rect = patches.Rectangle((min_x, min_y),
                                             max_x - min_x,
                                             max_y - min_y,
                                             linewidth=float(self.fish.config["info"]["bbox_preview_line_width"]),
                                             edgecolor='r',
                                             facecolor='none')
                    ax.add_patch(rect)
                if self.fish.config["info"]["show_cluster"].lower() == "true":
                    for cluster in raw["clusters"]:
                        ax.scatter(cluster[:, 1], cluster[:, 0], s=1, c='b')
                if self.fish.config["info"]["show_brightest_point"].lower() == "true":
                    ax.scatter(raw["bright_points"][:, 1], raw["bright_points"][:, 0], s=0.2, c='g')
            plt.show()
        
        def validate(self, img_name, img: np.ndarray, gt: list, gt_pos: list) -> None:
            raw, masks, line_result, area_result = self.predict(img)
            fig_ax, ax = plt.subplots(1, figsize=(8, 8))
            fig_bx, bx = plt.subplots(1, figsize=(8, 8))
            ax.imshow(img, cmap='gray')
            bx.imshow(img, cmap='gray')
            if self.fish.config["info"]["mask_erosion"].lower() == "true":
                result = line_result
            else:
                result = area_result
            
            if self.fish.config["info"]["overlay"].lower() == "true":
                overlay = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
                overlay[result == 1] = np.array([255, 255, 0, 255])
                ax.imshow(overlay)
            if self.fish.config["info"]["show_iou"].lower() == "true":
                iou, whole_gt = Fish.cal_iou(area_result, gt, gt_pos)
                bx.imshow(whole_gt, alpha=0.5)
                ax.text(0.05, 0.95, f'IoU: {iou:.2f}', transform=ax.transAxes
                                                    , fontsize=self.fish.config["info"]["iou_font"].lower()
                                                    , verticalalignment='top'
                                                    , bbox=dict(facecolor='white', alpha=0.5))
            if self.fish.config["info"]["bbox_preview"].lower() == "true":
                for bbox in raw["bboxes"]:
                    min_x, min_y, max_x, max_y = bbox
                    rect = patches.Rectangle((min_x, min_y),
                                             max_x - min_x,
                                             max_y - min_y,
                                             linewidth=float(self.fish.config["info"]["bbox_preview_line_width"]),
                                             edgecolor='r',
                                             facecolor='none')
                    ax.add_patch(rect)
                if self.fish.config["info"]["show_cluster"].lower() == "true":
                    for cluster in raw["clusters"]:
                        ax.scatter(cluster[:, 1], cluster[:, 0], s=1, c='b')
                if self.fish.config["info"]["show_brightest_point"].lower() == "true":
                    ax.scatter(raw["bright_points"][:, 1], raw["bright_points"][:, 0], s=0.2, c='g')
            if self.fish.config["validate"]["to_output"].lower() == "true":
                output_folder = pathlib.Path(self.fish.config["validate"]["output_folder"])
                os.makedirs(output_folder, exist_ok=True)
                ax_plot_path = os.path.join(output_folder, f'{img_name}_a.png') # prediction
                bx_plot_path = os.path.join(output_folder, f'{img_name}_b.png') # ground truth
                fig_ax.savefig(ax_plot_path)
                fig_bx.savefig(bx_plot_path)
                plt.close(fig_ax)
                plt.close(fig_bx)

            else:
                plt.show()
        