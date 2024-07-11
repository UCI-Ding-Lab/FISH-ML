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
    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True,
                                       squared_pred=True,
                                       reduction="mean")
    num_epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
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
        logging.info(f"EPOCH: {epoch} | Mean loss: {mean(epoch_losses)}")
    logging.info(f"Done")


    logging.info(f"Saving model...")
    # Save the model to the file
    torch.save(model.state_dict(), "./fish_v2.1.pth")
    logging.info(f"Model saved successfully!")