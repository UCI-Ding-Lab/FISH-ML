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

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QSlider, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest

class ImageOverlayWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageOverlayWidget, self).__init__(parent)
        self.loaded = False
        self.loading = False
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
        textColor = Qt.white
        painter.setPen(textColor)
        painter.setFont(QFont('Arial', 20))
        if not self.loaded:
            if self.loading:
                painter.drawText(self.rect(), Qt.AlignCenter, "Loading...")
                return
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return
        painter.drawPixmap(self.rect(), self.bottomPixmap)
        painter.setOpacity(self.opacity)
        painter.drawPixmap(self.rect(), self.topPixmap)
    
    def exportToImage(self, filename):
        pixmap = QPixmap(self.size())
        self.render(pixmap)
        pixmap.save(filename)
        

class FISH_APP(QMainWindow):
    # Global Variables
    SAM_CHECKPOINT = 'sam_vit_h_4b8939.pth'
    MODEL_TYPE = 'vit_h'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TITLE = "FISH APP"

    def __init__(self, debug=False):
        super().__init__()
        self.setWindowTitle(self.TITLE)
        self.setGeometry(100, 100, 800, 800)
        self.sam = None
        self.last_load_size = None
        self.cuda_supportive_test() if debug else None
        self.init_sam()
        self.overlay = ImageOverlayWidget()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.layout.addWidget(self.overlay)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(int(self.overlay.opacity * 100))
        self.slider.valueChanged[int].connect(self.changeOpacity)
        self.layout.addWidget(self.slider) # for the sake of testing, hard coded

        self.setCentralWidget(self.central_widget)
        self.init_menu()
        self.overlay.update()
        self.update()
    
    def init_menu(self) -> None:
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu('File')
        
        self.menu_file.addAction('Open')
        self.menu_file.addAction('Save')
        self.menu_file.triggered.connect(self.menu_file_action)
    
    def menu_file_action(self, action) -> None:
        if action.text() == 'Open':
            self.open_img()
        elif action.text() == 'Save':
            self.save_img()

    def open_img(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        self.run(path)
    
    def save_img(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        self.overlay.exportToImage(path)
    
    def changeOpacity(self, value):
        self.overlay.set_opacity(value / 100.0)

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
        #self.resize(image.shape[1], image.shape[0])
        return image
    
    def sam_amg_predict(self, image: np.ndarray) -> list[dict]:
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        masks = mask_generator.generate(image)
        return masks
    
    def mask_to_matrix(self, masks) -> np.ndarray:
        img = np.ones((2048, 2048, 4), dtype=np.float32)
        img[:, :, 3] = 0
        
        # Calculate the size of each mask
        mask_sizes = [np.sum(single_cell['segmentation']) for single_cell in masks]
        
        # Identify the largest mask
        largest_mask_index = np.argmax(mask_sizes)
        
        # Apply colors and opacity
        for index, single_cell in enumerate(masks):
            m = single_cell['segmentation']
            
            if index == largest_mask_index:
                color_mask = np.array([0, 0, 0, 0], dtype=np.float32)  # Set color and make invisible
            else:
                color_mask = np.concatenate([np.random.random(3).astype(np.float32), [1]]).astype(np.float32)  # Random color and fully opaque
            
            img[m] = color_mask
        
        return img
    
    def matrix_to_qimage(self, matrix) -> QImage:
        if matrix.dtype == np.float32:
            matrix = (matrix * 255).astype(np.uint8)
        
        height, width, channels = matrix.shape
        bytes_per_line = width * channels

        return QImage(matrix.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
    
    def run(self, img) -> None:
        # Stat update
        self.overlay.loaded = False
        self.overlay.loading = True
        self.overlay.update()
        QTest.qWait(300)

        # ML
        masks = self.sam_amg_predict(self.load_image(img))
        img_mtx = self.mask_to_matrix(masks)
        qimage = self.matrix_to_qimage(img_mtx)

        # Stat update w/o display
        self.overlay.loading = False
        self.overlay.loaded = True

        # Display
        self.overlay.set_bot_pixmap(img)
        self.overlay.set_top_pixmap(qimage)
        self.overlay.set_opacity(0.5)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = FISH_APP(debug=False)
    main.show()
    sys.exit(app.exec_())