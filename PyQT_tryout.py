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

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


# Global Variables
DEVICE = 'cuda' if (platform.system() == "Windows") else 'cpu'
SAM_CHECKPOINT = 'sam_vit_h_4b8939.pth'
MODEL_TYPE = 'vit_h'

# Functions for SAM (predictor_tryout.ipynb)
def supportTest(DEVICE):
    import torch
    import torchvision
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name()) if (DEVICE == 'cuda') else print("Device: CPU")
supportTest(DEVICE)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# Functions for PyQT (new)
def matrix_to_qimage(matrix):
    # Assuming matrix is a 3D numpy array with shape (height, width, 4) and dtype=np.float32 or np.uint8
    if matrix.dtype == np.float32:  # Assuming your array is in [0, 1] for floats
        matrix = (matrix * 255).astype(np.uint8)
    height, width, channels = matrix.shape
    bytes_per_line = width * channels  # 4 bytes per pixel for RGBA
    return QImage(matrix.data, width, height, bytes_per_line, QImage.Format_RGBA8888)

def display_image(qimage):
    app = QApplication(sys.argv)

    # Create a QMainWindow or any other QWidget as the main window
    window = QMainWindow()
    window.setWindowTitle('Mask Display')

    # Use a QLabel to display the image
    label = QLabel()
    label.setPixmap(QPixmap.fromImage(qimage))

    # Set the QLabel as the central widget of the QMainWindow
    window.setCentralWidget(label)

    window.show()
    sys.exit(app.exec_())

def main():
    sam_checkpoint = SAM_CHECKPOINT
    model_type = MODEL_TYPE
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)

    # Read image from folder
    image = cv2.imread('img/test_1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)

    # Predict
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    # Ensure the dtype is float to accommodate the color values and alpha channel
    img = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4), dtype=np.float32)
    img[:, :, 3] = 0  # Set alpha channel to fully transparent

    for ann in sorted_masks:
        m = ann['segmentation']
        # Ensure the color_mask is an array of floats for the RGBA channels
        color_mask = np.concatenate([np.random.random(3).astype(np.float32), [0.35]]).astype(np.float32)
        img[m] = color_mask

    print(img.shape)  # (2048, 2048, 4), confirming the final image shape

    qimage = matrix_to_qimage(img)
    display_image(qimage)

if __name__ == "__main__":
    main()