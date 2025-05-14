#  Electric Pole Defect Detection — Leaned vs Straight

This repository contains training and evaluation code for detecting **leaned vs straight electric poles** using **YOLOv12** object detection. It was developed as part of the APEPDCL Line Quality Monitoring project using real-world field data.

This repo also includes image-level classification accuracy evaluation scripts based on YOLO predictions.

---

##  Dataset

Dataset hosted on Hugging Face:  
👉 [Object Detection Dataset](https://huggingface.co/datasets/sampath-balaji/Electrical-Lines-Defect-Detection/tree/main/Poles_LeanedStraight/ObjectDetection)

- 1804 total images from 3 districts in Andhra Pradesh
- Format: YOLOv12-style `.jpg` images and `.txt` annotations
- Labels:
  - `0`: Leaned_Pole
  - `1`: Straight_Pole
- Splits:
  - Train: 1444 images  
  - Val: 181 images  
  - Test: 179 images

---

## 📁 GitHub Repository Structure

```
Electrical-Lines-Defect-Detection/
└── Pole_LeanedStraight_Defect/
  └── ObjectDetection/
    ├── train.py # YOLOv12 training script
    ├── run_inference_and_eval_val.py # Inference + metrics on val set
    ├── run_inference_and_eval_test.py # Inference + metrics on test set
    ├── TrainAndEval.ipynb # Jupyter notebook with training and eval pipeline set
    ├── object_detection_dataset_bboxes.json
    ├── README.md (this file)
  └── assets/
```

---

## 📥 Setup: Clone the Dataset
To run training and evaluation, you must first clone the dataset locally and place it in the expected directory:
```bash
# Step 1: Install Git LFS if not already installed
git lfs install

# Step 2: Clone the dataset repo
git clone https://huggingface.co/datasets/sampath-balaji/Electrical-Lines-Defect-Detection

# Step 3: Move or rename it to match expected path
mv Electrical-Lines-Defect-Detection /path/to/dataset/Electrical-Lines-Defect-Detection
```
Ensure the folder contains the following structure:
```bash
../Poles_LeanedStraight/ObjectDetection/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
```
You may edit the path in ```train.py```, ```run_inference_and_eval_val.py```, and ```run_inference_and_eval_test.py``` if your local directory differs.

## Training (Use the Jupyter notebook with the pipeline ready / the script)

```bash
python train.py
```
Training by default uses:
- Base model: yolo12m.pt
- Batch size: 16
- Image size: 640x640
- Epochs: 200
- Optimizer: auto
- Device: cuda

  ## 🔍Inference & Evaluation
  ###  Run on validation set:
  ```bash
  python run_inference_and_eval_val.py
  ```
  ###  Run on test set:
  ```bash
  python run_inference_and_eval_test.py
  ```

## 📊 YOLOv12 Object Detection Performance
#### Hardware Used for training model: NVIDIA GeForce RTX 4070 Ti SUPER on [JOHNAIC](https://von-neumann.ai/index.html)
###  Validation Set
| Class          | Precision | Recall    | mAP\@0.5  | mAP\@0.5:0.95 |
| -------------- | --------- | --------- | --------- | ------------- |
| Leaned\_Pole   | 0.894     | 0.894     | 0.963     | 0.730         |
| Straight\_Pole | 0.914     | 0.875     | 0.934     | 0.572         |
| **Overall**    | **0.904** | **0.884** | **0.949** | **0.651**     |
##### Speed: 2.6ms/inference, 0.2ms/postprocess per image
###  Test Set
| Class          | Precision | Recall    | mAP\@0.5  | mAP\@0.5:0.95 |
| -------------- | --------- | --------- | --------- | ------------- |
| Leaned\_Pole   | 0.911     | 0.807     | 0.928     | 0.734         |
| Straight\_Pole | 0.922     | 0.879     | 0.968     | 0.630         |
| **Overall**    | **0.917** | **0.843** | **0.948** | **0.682**     |
##### Speed: 2.3ms/inference, 0.2ms/postprocess per image

## 📊 Image-Level Classification Performance
#### Evaluated using YOLO prediction outputs aggregated per image.
### Validation Set
| Class          | Accuracy | Precision | Recall | F1 Score |
| -------------- | -------- | --------- | ------ | -------- |
| Leaned\_Pole   | 94.48%   | 93.75%    | 98.36% | 96.00%   |
| Straight\_Pole | 93.92%   | 92.00%    | 93.24% | 92.62%   |

### Test Set
| Class          | Accuracy | Precision | Recall | F1 Score |
| -------------- | -------- | --------- | ------ | -------- |
| Leaned\_Pole   | 91.62%   | 89.77%    | 92.94% | 91.33%   |
| Straight\_Pole | 93.85%   | 93.02%    | 98.36% | 95.62%   |

## 📄 License

- **Code:** MIT License  
- **Dataset:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
  Available at: [Hugging Face Dataset](https://huggingface.co/datasets/sampath-balaji/Electrical-Lines-Defect-Detection/tree/main/Poles_LeanedStraight/ObjectDetection)

## Sample predictions
<p align="center">
  <img src="https://raw.githubusercontent.com/sampath-balaji/Electrical-Lines-Defect-Detection/refs/heads/main/Pole_LeanedStraight_Defect/ObjectDetection/assets/output.jpeg" width="600"/>
</p>
