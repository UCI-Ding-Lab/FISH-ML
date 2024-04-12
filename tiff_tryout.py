from PIL import Image
import pathlib
import numpy as np
import matplotlib.pyplot as plt

with Image.open(pathlib.Path("151-200_Hong/MAX_KO2_w1-359 DAPI_s055.tif")) as img:
        image_array = np.array(img)
        exposure_factor = 30
        exposed_image_array = np.clip(image_array * exposure_factor, 0, 65535)
        exposed_image = Image.fromarray(exposed_image_array)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=65535)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(exposed_image, cmap='gray', vmin=0, vmax=65535)
        plt.title('Exposed Image (+8 EV)')
        plt.axis('off')
        plt.show()