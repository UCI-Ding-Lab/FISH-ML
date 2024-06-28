import pathlib
import logging
import configparser
from sklearn.cluster import DBSCAN
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import SamModel, SamConfig, SamProcessor
from scipy.ndimage import binary_erosion

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
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        
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
        modle_folder = self.asset_folder_path / "modle"
        for v in self.supported_version:
            if not (modle_folder / f"fish_{v}.pth").exists():
                self.logger.error(f"CHECK: Model file for version {v} is missing")
    
    def predict(self, img: np.ndarray):
        if not self.model:
            self.logger.error("PRED: Modle not loaded")
            return
        
        def prepare_input(img):
            def get_top_brightness_points(np_image, percentage):
                grayscale_image = np.mean(np_image, axis=2)
                threshold_value = np.percentile(grayscale_image, 100 - percentage)
                bright_areas_mask = grayscale_image > threshold_value
                return bright_areas_mask
            
            def cluster_points(points, eps=3, min_samples=5):
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
            
            bright_areas_mask = get_top_brightness_points(img, int(self.config["predict"]["brightest_percentage"]))
            bright_points = np.column_stack(np.where(bright_areas_mask))
            clusters = cluster_points(bright_points, eps=int(self.config["predict"]["dbscan_eps"]), min_samples=int(self.config["predict"]["dbscan_min_samples"]))
            grouped_input_points = [cluster.tolist() for cluster in clusters]
            bounding_boxes = [get_bounding_box(np.array(cluster)) for cluster in grouped_input_points]
            return bounding_boxes
        
        bboxes = prepare_input(img)
        self.logger.info(f"PRED: found {len(bboxes)} bounding boxes")
        
        # all together
        inputs = self.processor(img, input_boxes=[[bbox for bbox in bboxes]], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
            # !!!pred_masks shape: torch.Size([1, 43, 1, 256, 256]) be4 squeeze
        
        self.logger.info(f"PRED: generated {outputs.pred_masks.shape[1]} masks")
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        masks = masks[0].squeeze(1).numpy().astype(np.uint8)
        result = np.zeros(img.shape[:2], dtype=np.uint8)
        
        for ind, bbox in enumerate(bboxes):
            min_x, min_y, max_x, max_y = bbox
            
            # ignore noise
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
            if self.config["show"]["overlay"].lower() == "true":
                img[result == 1] = [255,0,0]
                plt.imshow(img)
            else:
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                if self.config["show"]["enhancement"].lower() == "true":
                    axes[0].imshow(np.clip(np.array(img) * float(self.config["show"]["exposure_factor"]), 0, 65535), cmap='gray')
                else:
                    axes[0].imshow(img)
                axes[0].set_title("Image")
                axes[1].imshow(result, cmap='gray')
                axes[1].set_title("Mask")
            plt.show()
        
        return masks
        
        