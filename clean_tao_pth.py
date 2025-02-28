import torch

# Load TAO-trained model checkpoint
tao_pth_path = "./GroundingDINO/groundingdino/weights/commercial.pth"
# tao_pth_path = "./GroundingDINO/groundingdino/weights/groundingdino_swint_ogc.pth"
checkpoint = torch.load(tao_pth_path, map_location="cpu")

# Identify the state_dict
if "model" in checkpoint:
    state_dict = checkpoint["model"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint  # Direct model state_dict

# Remove 'model.model.' prefix
new_state_dict = {}
for key in state_dict.keys():
    new_key = key.replace("model.model.", "module.")
    new_state_dict[new_key] = state_dict[key]

checkpoint["model"] = new_state_dict

# Save the cleaned checkpoint
cleaned_checkpoint_path = "./GroundingDINO/groundingdino/weights/commercial_clean.pth"
torch.save(checkpoint, cleaned_checkpoint_path)
print(f"Checkpoint cleaned and saved at {cleaned_checkpoint_path}")
