# FISH-ML for Cell Image Segmentation

This repo is a unified framework for Cell Image Segmentation of image using **Segment Anything Model** and **Grounding DINO** which utilizes **Transformers** and **PyTorch** as Backbone.

## Our models
Some of our SAM Finetuned Checkpoints (for testing only):
 - `fish_v1.1.pth` - 1 ep, 50 images, 256 p_size, 0.5 overlap, 11250 patches, 0.89 loss
 - `fish_v1.100.pth` - 100 EP, 50 images, 256 p_size, 0.5 overlap, 11250 patches, 0.4655 loss
 - `fish_v2.1.pth` - 1 ep, 150 images, 256 p_size, 0.5 overlap, 33750 patches, 0.63 loss
 - `fish_v3.50.pth` - 50 ep, 150 images, whole_image_input
