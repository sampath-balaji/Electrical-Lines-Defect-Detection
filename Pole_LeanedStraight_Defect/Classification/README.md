# Electric Pole Classification — Leaned vs Straight vs Rejected

This repository contains training code for **image-level classification** of electric utility poles into:
- `Leaned`
- `Straight`
- `Rejected`

This complements the YOLOv12-based [object detection module](https://github.com/sampath-balaji/Electrical-Lines-Defect-Detection/tree/main/Pole_LeanedStraight_Defect/ObjectDetection). The model used is a fine-tuned [DINOv2 ViT-B/14](https://github.com/facebookresearch/dinov2) vision transformer from Meta AI.

---

## Dataset

Dataset hosted on Hugging Face:  
👉 [Classification Dataset](https://huggingface.co/datasets/sampath-balaji/Electrical-Lines-Defect-Detection/tree/main/Poles_LeanedStraight/Classification)

### Structure:
```bash
Classification/
├── train/
│   ├── Leaned/
│   ├── Straight/
│   └── Rejected/
├── val/
│   ├── Leaned/
│   ├── Straight/
│   └── Rejected/
├── test/
│   ├── Leaned/
│   ├── Straight/
│   └── Rejected/
```
- Labels were assigned by 3 human annotators via majority voting.
- No augmentations applied to the dataset itself.
- Splits: Train 75%, Val 15%, Test 10%

### Labeling Process

Each image in the dataset was labeled independently by **three human annotators**, each assigning one of the following class labels:
- `Leaned` – the pole visibly leans from vertical
- `Straight` – the pole appears upright
- `Rejected` – unclear cases (e.g., cropped tops/bottoms, extreme tilt, occlusions)

The final label for each image was determined using a **majority voting approach**, where at least 2 out of 3 annotators agreed on the class.

A CSV file [`image_labels_with_majority.csv`](./image_labels_with_majority.csv) is included in this repo, containing:
- `filename`: Image file name
- `Straight`, `Leaned`, `Rejected`: Labels assigned by each of the three annotators
- `majority_label`: The agreed label (via majority vote) used in the dataset

This CSV allows users to audit the labeling process or re-define label selection strategies if needed.

## Directory Structure
```bash
Pole_LeanedStraight_Defect/
└── Classification/
    ├── train.py           # Training script
    ├── saved_models/      # Folder created for model checkpoints
    ├── visualizations/    # Folder created for output images
```
## Setup: Clone the Dataset
To run training, first clone the dataset and organize it as expected:
```bash
# Step 1: Install Git LFS
git lfs install

# Step 2: Clone the dataset repo
git clone https://huggingface.co/datasets/sampath-balaji/Electrical-Lines-Defect-Detection

# Step 3: Rename or move it to match expected path
mv Electrical-Lines-Defect-Detection ../your_path
```
Ensure the folder structure is:
```bash
..Poles_LeanedStraight/Classification/
├── train/
│   ├── Leaned/
│   ├── Straight/
│   └── Rejected/
├── val/
│   ├── Leaned/
│   ├── Straight/
│   └── Rejected/
├── test/
│   ├── Leaned/
│   ├── Straight/
│   └── Rejected/
```

## Train the Model
```bash
pip install torch torchvision matplotlib tqdm
```
Run training:
```bash
python train.py
```
This will:
- Train for 40 epochs
- Save best model to: ```saved_models/best_dino_model.pt```
- Save prediction visualizations to: ```visualizations/```
Alternatively, a Jupyter notebook ([`train_dino_with_outputs.ipynb`](https://github.com/sampath-balaji/Electrical-Lines-Defect-Detection/blob/main/Pole_LeanedStraight_Defect/Classification/train_dino_with_outputs.ipynb)) is also included to run the training pipeline and visualize predictions interactively.

## Validation Accuracy
Best validation accuracy: 84.14%
(Trained using DINOv2 ViT-B/14 on RTX 4070 Ti SUPER on [JOHNAIC](https://von-neumann.ai/))

## Sample Visualizations
<p align="center"> <img src="https://raw.githubusercontent.com/sampath-balaji/Electrical-Lines-Defect-Detection/refs/heads/main/Pole_LeanedStraight_Defect/Classification/assets/val_viz_ep40.png" width="600"/> </p>
