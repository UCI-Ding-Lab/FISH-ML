# FISH-ML for Cell Image Segmentation

This repo is a unified framework for Cell Image Segmentation of image using **Segment Anything Model** and **Grounding DINO** which utilizes **Transformers** and **PyTorch** as Backbone.

## Data Preparation
For zero-shot segmentation, make sure your images are in the following format:
 - 2048x2048x1 (grayscale) 
 - 16 bit color space
 - .tif

## Environment Setup
### Basic Installation
Please import `env.yaml` in Anaconda to install your environment.
Also, here is another way of using conda:
```bash
conda create -n fish python=3.11.7
conda activate fish
pip install -r requirements.txt
```

### Download assets folder
You can download the assets here: https://drive.google.com/drive/folders/1lmUERNGg93F5DzTO0FOVQYpI0nNKP2eg?usp=drive_link
Please place the `/assets` folder in the root of this repo:
```
/FISH-ML # current repo
. 
├── /archive
├── /assets
│   ├── /dataset
│   ├── /icon
│   │   ├── brush.png
│   │   └── eraser.png
│   ├── /model
│   │   └── fish_v3.50.pth
│   ├── /tif
│   │   ├── /1-50_Hong
│   │   ├── /51-100_Hong
│   │   ├── /151-200_Hong
│   │   └── /201-250_Hong
│   └── /validate_output
├── /train
├── README.md
├── requirements.txt
├── env.yaml
├── config.ini
...
```

### Install SAM and DINO
Make sure you clone or download the repo for SAM and DINO from
> SAM: https://github.com/facebookresearch/segment-anything/tree/main/segment_anything

> GroundingDINO: https://github.com/IDEA-Research/GroundingDINO

> DINO ckpt: https://huggingface.co/ShilongLiu/GroundingDINO/blob/main/groundingdino_swint_ogc.pth

and place them at the following location:
``` python
/FISH-ML # current repo
. 
├── /segment_anything # repo of SAM
├── /GroundingDINO # repo of DINO
│   └── groundingdino
│       └── weights
│           └── groundingdino_swint_ogc.pth # make sure you have this file here
├── /archive
├── /assets
├── /train
├── README.md
├── requirements.txt
├── env.yaml
├── config.ini
├── example_usage.py
├── fish.ui
├── fishGUI.py
├── fishCore.py
├── validate.py
...
```

## Run FishUI
After you successfully install all envs needed, you can run FishUI by simply running `fishGUI.py`, here is an example of how to run it in your terminal:
```bash
# don't forget to run the file under fish environment
/Users/anaconda3/envs/FISH/bin/python ./fishGUI.py   
```
After you run it, there should be a new UI window popped up, and you can begin your image segmentation!!! We are now using fish_v3.50.pth as our SAM model.

## Other notes
Some of our SAM Finetuned Checkpoints (for testing only):
 - `fish_v1.1.pth` - 1 ep, 50 images, 256 p_size, 0.5 overlap, 11250 patches, 0.89 loss
 - `fish_v1.100.pth` - 100 EP, 50 images, 256 p_size, 0.5 overlap, 11250 patches, 0.4655 loss
 - `fish_v2.1.pth` - 1 ep, 150 images, 256 p_size, 0.5 overlap, 33750 patches, 0.63 loss
 - `fish_v3.50.pth` - 50 ep, 150 images, whole_image_input