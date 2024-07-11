import pathlib
import logging
import configparser
from sklearn.cluster import DBSCAN
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import SamModel, SamConfig, SamProcessor, pipeline
from scipy.ndimage import binary_erosion
from segment_anything import SamPredictor, sam_model_registry

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

class fishcore():
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
        
        #self.assets_validation()
    
    def set_modle_version(self,v):
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
        """unfinsihsed
        """
        modle_folder = self.asset_folder_path / "modle"
        for v in self.supported_version:
            if not (modle_folder / f"fish_{v}.pth").exists():
                self.logger.error(f"CHECK: Model file for version {v} is missing")
    
    def predict(self, img: np.ndarray) -> np.ndarray:
        """clustering bright points to bbox, predict masks, combine masks

        Args:
            img (np.ndarray): grey scale only, 2dim

        Returns:
            np.ndarray: 3dim [n, h, w] masks
        """
        if not self.model:
            self.logger.error("PRED: Modle not loaded")
            return
        if len(img.shape) != 2:
            self.logger.error("PRED: Image should be in grayscale")
            return
        
        def hdr_to_rgb(hdr_image, dynamic_range):
            scale_factor = 255 / dynamic_range
            scaled_image = (hdr_image * scale_factor).astype(np.uint8)
            rgb_image = np.stack((scaled_image,) * 3, axis=-1)
            return rgb_image
        
        def prepare_input(img):
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
            
            bright_areas_mask = get_top_brightness_points(img, float(self.config["predict"]["brightest_percentage"]))
            bright_points = np.column_stack(np.where(bright_areas_mask))
            clusters = cluster_points(bright_points, eps=float(self.config["predict"]["dbscan_eps"]), min_samples=int(self.config["predict"]["dbscan_min_samples"]))
            grouped_input_points = [cluster.tolist() for cluster in clusters]
            bounding_boxes = [get_bounding_box(np.array(cluster)) for cluster in grouped_input_points]
            return bright_points, clusters, bounding_boxes
        
        def finalize_bbox(orignal_bbox, expand_fct, bbox_area_threshold):
            bboxes = []
            for box in orignal_bbox:
                min_x, min_y, max_x, max_y = box
                area = (max_x - min_x) * (max_y - min_y)
                if area < bbox_area_threshold:
                    continue
                expanded_bbox = [min_x - expand_fct,
                                min_y - expand_fct,
                                max_x + expand_fct,
                                max_y + expand_fct]
                bboxes.append(expanded_bbox)
            return bboxes
        
        raw = prepare_input(img)
        self.logger.info(f"CLU: found {len(raw[0])} bright points")
        bboxes = finalize_bbox(raw[-1],
                               int(self.config["predict"]["bbox_expand_px"]),
                               int(self.config["predict"]["bbox_area_threshold"]))
        self.logger.info(f"CLU: found {len(bboxes)} bounding boxes")
        
        prepro_img_3_chan = hdr_to_rgb(img, int(self.config["predict"]["dynamic_range"]))
        masks = None
        
        if self.config["sam"]["enable"].lower() == "true":
            sam = sam_model_registry["vit_h"](checkpoint=self.asset_folder_path/"sam"/"sam_vit_h_4b8939.pth")
            predictor = SamPredictor(sam)
            sam.to(device="cpu")
            predictor.set_image(prepro_img_3_chan)
            input_boxes = torch.tensor(bboxes, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, prepro_img_3_chan.shape[:2])
            masks, _, _ = predictor.predict_torch(point_coords=None,
                                                  point_labels=None,
                                                  boxes=transformed_boxes,
                                                  multimask_output=False)
            masks = masks.squeeze(1).cpu().numpy().astype(np.uint8)
            # masks (n, 2048, 2048)
        
        if self.config["predict"]["enable"].lower() == "true":
            inputs = self.processor(prepro_img_3_chan,
                                    input_boxes=[[bbox for bbox in bboxes]],
                                    return_tensors="pt",
                                    do_convert_rgb=False).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs: dict = self.model(**inputs,
                                        multimask_output=False)
                # !!!pred_masks shape: torch.Size([1, bbox, 1, 256, 256]) be4 squeeze
            self.logger.info(f"PRED: generated {outputs.pred_masks.shape[1]} masks")
            
            
            masks: list = self.processor.image_processor.post_process_masks(masks=outputs.pred_masks.cpu(),
                                                                            original_sizes=inputs["original_sizes"].cpu(),
                                                                            reshaped_input_sizes=inputs["reshaped_input_sizes"].cpu(),
                                                                            mask_threshold=float(self.config["predict"]["mask_threshold"]))  
            masks = masks[0].squeeze(1).numpy().astype(np.uint8)
            # masks (n, 2048, 2048)
        
        if masks is None:
            self.logger.error("CHECK: enable at least one mask generation method")
            return
        
        self.logger.debug(f"PRED: combination started")
        result = np.zeros(img.shape[:2], dtype=np.uint8)
        for ind, bbox in enumerate(bboxes):
            min_x, min_y, max_x, max_y = bbox
            if max_x <= min_x or max_y <= min_y:
                continue
            seg = masks[ind]
            seg = seg[min_y:max_y, min_x:max_x]
            if self.config["predict"]["mask_erosion"].lower() == "true":
                er_factor = int(self.config["predict"]["erosion_factor"])
                eroded_mask = binary_erosion(seg, structure=np.ones((er_factor, er_factor)))
                seg = seg - eroded_mask
            result[min_y:max_y, min_x:max_x] = seg
        self.logger.info("PRED: combination completed")

        if self.config["show"]["enable"].lower() == "true":
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(img, cmap='gray')
            if self.config["show"]["overlay"].lower() == "true":
                overlay = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
                overlay[result == 1] = np.array([255, 255, 0, 255])
                ax.imshow(overlay)
            if self.config["bbox_preview"]["enable"].lower() == "true":
                for bbox in bboxes:
                    min_x, min_y, max_x, max_y = bbox
                    rect = patches.Rectangle((min_x, min_y),
                                            max_x - min_x,
                                            max_y - min_y,
                                            linewidth=float(self.config["bbox_preview"]["line_width"]),
                                            edgecolor='r',
                                            facecolor='none')
                    ax.add_patch(rect)
                if self.config["bbox_preview"]["show_cluster"].lower() == "true":
                    for cluster in raw[1]:
                        ax.scatter(cluster[:, 1], cluster[:, 0], s=1, c='b')
                if self.config["bbox_preview"]["show_brightest_point"].lower() == "true":
                    ax.scatter(raw[0][:, 1], raw[0][:, 0], s=0.2, c='g')
            plt.show()
        
        return masks, result

    def info(self, img: np.ndarray, gt: np.ndarray) -> None:
        def cal_iou(result: np.ndarray, gt: np.ndarray):
            intersection = np.logical_and(result, gt)
            union = np.logical_or(result, gt)
            iou_score = np.sum(intersection) / np.sum(union)
            return iou_score
        
        masks, result = self.predict(img)
        iou = cal_iou(result, gt)
        self.logger.info(f"DB: IOU score: {iou}")
    
    def pipe(self, img: np.ndarray):
        """not working

        Args:
            img (np.ndarray): rgb
        """
        if not self.model:
            self.logger.error("PRED: Modle not loaded")
            return
        
        def show_mask(mask, ax, random_color=False):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
        
        generator =  pipeline("mask-generation",
                              points_per_batch = 256,
                              model=self.model,
                              device=self.device,
                              image_processor=self.processor.current_processor)
        
        outputs = generator(img, points_per_batch = 256)
        plt.imshow(img)
        ax = plt.gca()
        self.logger.info(f"PRED: generated {len(outputs['masks'])} masks")
        for mask in outputs["masks"]:
            show_mask(mask, ax=ax, random_color=True)
        plt.axis("off")
        plt.show()
        