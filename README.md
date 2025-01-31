# FISH-ML: DAPI Cell Image Segmentation

FISH-ML is a unified framework for DAPI channel segmentation utilizing the **Segment Anything Model (SAM)** and **Grounding DINO**. This repository leverages **Transformers** and **PyTorch** to provide an efficient, scalable, and high-performance segmentation pipeline.

---

## 📂 Dataset Preparation
For zero-shot segmentation, ensure that your images meet the following specifications:
- **Image dimensions:** 2048 × 2048 × 1
- **Color space:** 16-bit grayscale
- **Format:** `.tif`

---

## 🛠️ Environment Setup

### Initial Installation
Follow these steps to set up the environment for the first time. If you have already completed the setup, please do not overwrite your existing environment and proceed to the [Run FishUI](#-run-fishui) section.

#### Conda Installation
You can install the necessary dependencies using the provided `env.yaml` file in Anaconda. Alternatively, follow these steps:
```bash
conda create -n fish python=3.11.7
conda activate fish
pip install -r requirements.txt
```
If all of these doesn't work, manually install all packages in `requirements.txt`.

### 🔽 Downloading Required Assets
The required assets can be downloaded from the following link:
[Google Drive Assets Folder](https://drive.google.com/drive/folders/1lmUERNGg93F5DzTO0FOVQYpI0nNKP2eg?usp=drive_link)

To gain access, please request permission via email: **hzhang952@wisc.edu**.

Once downloaded, place the `/assets` folder in the root directory of the repository:
```
/FISH-ML  # Root directory
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

---

## 🏗️ Installing SAM and Grounding DINO
To ensure seamless operation, you need to clone or download the repositories for **SAM** and **Grounding DINO**:

- **Segment Anything Model (SAM):** [GitHub Repository](https://github.com/facebookresearch/segment-anything/tree/main/segment_anything)
- **Grounding DINO:** [GitHub Repository](https://github.com/IDEA-Research/GroundingDINO)
- **Grounding DINO Checkpoint:** [Hugging Face Model](https://huggingface.co/ShilongLiu/GroundingDINO/blob/main/groundingdino_swint_ogc.pth)

Once downloaded, ensure that they are placed in the following structure within your project:
```
/FISH-ML  # Root directory
├── /segment_anything  # SAM repository
│   ├── /modeling
│   ├── /utils
|   ...
├── /GroundingDINO  # Grounding DINO repository
│   ├── groundingdino
│   │   └── weights
│   │       └── groundingdino_swint_ogc.pth  # Required checkpoint
│   ...
├── /archive
├── /dataset_utils
├── /assets
├── /sam_train
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

---

## 🚀 Running FishUI
After successfully setting up the environment and installing all required dependencies, you can launch **FishUI** using the following steps:
```bash
# Activate the environment
conda activate fish

# Run FishUI
/Users/anaconda3/envs/FISH/bin/python ./fishGUI.py   
```

Once the script executes, a UI window will appear, allowing you to begin segmenting your images effortlessly. Currently, **fish_v3.50.pth** is used as the SAM model, while **Grounding DINO checkpoints** are still under training.

---

## 📝 Notes on Model Checkpoints
Below are some of our SAM fine-tuned checkpoints for reference (for testing purposes only):

| Checkpoint         | Epochs | Images | Patch Size | Overlap | Patches | Loss  |
|--------------------|--------|--------|------------|---------|---------|-------|
| `fish_v1.1.pth`   | 1      | 50     | 256        | 0.5     | 11,250  | 0.89  |
| `fish_v1.100.pth` | 100    | 50     | 256        | 0.5     | 11,250  | 0.4655|
| `fish_v2.1.pth`   | 1      | 150    | 256        | 0.5     | 33,750  | 0.63  |
| `fish_v3.50.pth`  | 50     | 150    | Whole Image Input | - | - | - |

🔹 **Note:** Ensure that you are using version **3.50** for **SAM**.

---

## 🎯 Future Enhancements
We are actively improving this repository and training additional **Grounding DINO checkpoints** for even better detection performance. Stay tuned for updates!

For any issues, feature requests, or contributions, feel free to open an issue or submit a pull request. 🚀


## 📜 License
This project is developed under [UCI Ding Lab](https://www.ding.eng.uci.edu).
