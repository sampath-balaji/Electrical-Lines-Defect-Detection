# Electric Pole Classification — Leaned vs Straight vs Rejected

This repository contains training code for **image-level classification** of electric utility poles into:
- `Leaned`
- `Straight`
- `Rejected`

It is part of the APEPDCL Line Quality Monitoring project, complementing the YOLOv12-based object detection module. The model used is a fine-tuned [DINOv2 ViT-B/14](https://github.com/facebookresearch/dinov2) vision transformer from Meta AI.

---

## Dataset

Dataset hosted on Hugging Face:  
👉 [ElectricPoles_Classification_StraightLeaned](https://huggingface.co/datasets/sampath-balaji/ElectricPoles_Classification_StraightLeaned)

### Structure:
```bash
ElectricPoles_Classification_StraightLeaned/
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
git clone https://huggingface.co/datasets/sampath-balaji/ElectricPoles_Classification_StraightLeaned

# Step 3: Rename or move it to match expected path
mv ElectricPoles_Classification_StraightLeaned ../your_path
```
Ensure the folder structure is:
```bash
../ElectricPoles_Classification_StraightLeaned/
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

## Validation Accuracy
Best validation accuracy: 84.14%
(Trained using DINOv2 ViT-B/14 on RTX 4070 Ti SUPER on [JOHNAIC](https://von-neumann.ai/)

## Sample Visualizations
<p align="center"> <img src="https://raw.githubusercontent.com/sampath-balaji/electrical-line-defects/refs/heads/main/Pole_LeanedStraight_Defect/Classification/assets/val_viz_ep40.png" width="600"/> </p>
