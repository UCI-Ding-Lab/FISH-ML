import numpy as np
import pathlib
import os
import time
import torch
import monai
import logging

from datasets import Dataset
from PIL import Image
from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from patchify import patchify
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
from statistics import mean
from torch.nn.functional import threshold, normalize

import fishLoader as fish
import dataset_proc as ppf


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(levelname)s] %(message)s")

    logging.info(f"Loading dataset...")
    dataset = Dataset.load_from_disk("./assets/dataset/")
    logging.info(f"Done")

    logging.info(f"Loading trainer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = ppf.fishDataset(dataset=dataset, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    batch = next(iter(train_dataloader))
    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    logging.info(f"Done")

    logging.info(f"Training in progress...")
    # Initialize TensorBoard SummaryWriter
    log_dir = f"runs/experiment_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True,
                                       squared_pred=True,
                                       reduction="mean")
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    epoch_mean_losses = []  # List to store mean losses for each epoch
    global_step = 0  # Initialize global step counter

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            # pm: (1,n,1,256,256)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            # gtm: (1,n,2048,2048)
            ground_truth_masks_resized = F.interpolate(ground_truth_masks,
                                                       size=(256, 256),
                                                       mode='bilinear',
                                                       align_corners=False).unsqueeze(2)
            # gtmr: (1,n,1,256,256) check
            loss = seg_loss(predicted_masks, ground_truth_masks_resized)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            # Log the loss to TensorBoard for every step
            writer.add_scalar('Loss/step', loss.item(), global_step)
            global_step += 1  # Increment global step

        mean_epoch_loss = mean(epoch_losses)
        epoch_mean_losses.append(mean_epoch_loss)
        logging.info(f"EPOCH: {epoch} | Mean loss: {mean(epoch_losses)}")
        writer.add_scalar('MeanLoss/epoch', mean_epoch_loss, epoch)
    logging.info(f"Done")


    logging.info(f"Saving model...")
    # Save the model to the file
    torch.save(model.state_dict(), "./fish_tb_1.0.pth")
    writer.close()
    logging.info(f"Model saved successfully!")