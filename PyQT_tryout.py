# Import & Settings
import sys
import scipy.io
import json
import numpy as np
import pathlib
import os
import platform
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import Qt

class ImageOverlayWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageOverlayWidget, self).__init__(parent)
        self.opacity = 0.5
    
    def set_bot_pixmap(self, path):
        self.bottomPixmap = QPixmap(path)
        self.update()
    def set_top_pixmap(self, qimage):
        self.topPixmap = QPixmap.fromImage(qimage)
        self.update()
    def set_opacity(self, opacity):
        self.opacity = opacity
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.bottomPixmap)
        painter.setOpacity(self.opacity)
        painter.drawPixmap(self.rect(), self.topPixmap)

class FISH_APP(QMainWindow):
    # Global Variables
    SAM_CHECKPOINT = 'sam_vit_h_4b8939.pth'
    MODEL_TYPE = 'vit_h'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TITLE = "FISH APP"
    IMG = "img/test_1.jpg"

    def __init__(self, debug=False):
        super().__init__()
        self.setWindowTitle(self.TITLE)
        self.sam = None
        self.last_load_size = None
        self.cuda_supportive_test() if debug else None
        self.init_sam()

        self.overlay = ImageOverlayWidget()
        self.run()
        self.setCentralWidget(self.overlay)
        self.overlay.update()
        self.update()
    
    def cuda_supportive_test(self) -> None:
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
        print("CUDA is available:", torch.cuda.is_available())
        print("Device:", torch.cuda.get_device_name()) if (self.DEVICE == 'cuda') else print("Device: CPU")
    
    def init_sam(self) -> None:
        self.sam = sam_model_registry[self.MODEL_TYPE](checkpoint=self.SAM_CHECKPOINT)
        self.sam.to(device=self.DEVICE)
    
    def load_image(self, path) -> np.ndarray:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.resize(image.shape[1], image.shape[0])
        return image
    
    def sam_amg_predict(self, image: np.ndarray) -> list[dict]:
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        masks = mask_generator.generate(image)
        return masks
    
    def mask_to_matrix(self, masks) -> np.ndarray:
        img = np.ones((2048, 2048, 4), dtype=np.float32)
        img[:, :, 3] = 0
        for single_cell in masks:
            m = single_cell['segmentation']
            color_mask = np.concatenate([np.random.random(3).astype(np.float32), [1]]).astype(np.float32)
            img[m] = color_mask
        return img
    
    def matrix_to_qimage(self, matrix) -> QImage:
        if matrix.dtype == np.float32:
            matrix = (matrix * 255).astype(np.uint8)
        height, width, channels = matrix.shape
        bytes_per_line = width * channels
        return QImage(matrix.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
    
    def run(self) -> None:
        masks = self.sam_amg_predict(self.load_image(self.IMG))
        img = self.mask_to_matrix(masks)
        qimage = self.matrix_to_qimage(img)

        self.overlay.set_bot_pixmap(self.IMG)
        self.overlay.set_top_pixmap(qimage)
        self.overlay.set_opacity(1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = FISH_APP(debug=False)
    main.show()
    sys.exit(app.exec_())