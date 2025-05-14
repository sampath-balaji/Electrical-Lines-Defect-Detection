# ⚡ Electric Line Defect Detection

- This repository presents an end-to-end computer vision pipeline for detecting and classifying electrical line defects.

- Developed as part of the APEPDCL Line Quality Monitoring System, this project includes both source code and real-world datasets, and is released to encourage open-source contributions, collaboration among people interested in AI.
  
-  **This ongoing project is supervised and mentored by [Sasank Chilamkurthy](https://chsasank.com/about/), whose expertise has guided its development.** 

## 📂 Project Modules
### Project Modules (Current Phase)


### - Object Detection  
📁 [`Pole_LeanedStraight_Defect/ObjectDetection`](./Pole_LeanedStraight_Defect/ObjectDetection)

- Detects leaned and straight poles via bounding boxes  
- Model: YOLOv12  
- Dataset: 1810 annotated images  
- 📊 Includes training + inference + metrics evaluation  
- 📄 [Full Documentation →](https://github.com/sampath-balaji/Electrical-Lines-Defect-Detection/tree/main/Pole_LeanedStraight_Defect/ObjectDetection)

---

### - Image Classification  
📁 [`Pole_LeanedStraight_Defect/Classification`](./Pole_LeanedStraight_Defect/Classification)

- Classifies whole pole images into: `Leaned`, `Straight`, or `Rejected`  
- Model: DINOv2 ViT-B/14  
- Dataset: Folder-based structure + labeling CSV  
- 📊 Best validation accuracy: **84.14%**  
- 📄 [Full Documentation →](https://github.com/sampath-balaji/Electrical-Lines-Defect-Detection/tree/main/Pole_LeanedStraight_Defect/Classification)

---

## - Datasets (Open Source)

| Dataset | Type | Hugging Face Link |
|--------|------|-------------------|
| **Object Detection** | YOLOv12-format | [Object Detection Dataset](https://huggingface.co/datasets/sampath-balaji/Electrical-Lines-Defect-Detection/tree/main/Poles_LeanedStraight/ObjectDetection) |
| **Image Classification** | Folder-based | [Classification](https://huggingface.co/datasets/sampath-balaji/Electrical-Lines-Defect-Detection/tree/main/Poles_LeanedStraight/Classification) |

---

## - Project Structure

```bash
Electrical-Lines-Defect-Detection/
├── Pole_LeanedStraight_Defect/
│   ├── ObjectDetection/
│   │   ├── assets/                         # Sample output images
│   │   ├── README.md
│   │   ├── TrainAndEval.ipynb              # Jupyter notebook with full pipeline
│   │   ├── run_inference_and_eval_test.py  # Inference + metrics for test set
│   │   ├── run_inference_and_eval_val.py   # Inference + metrics for val set
│   │   └── train.py                        # YOLOv12 training script
│   │
│   └── Classification/
│       ├── assets/                         # Visual output images
│       ├── README.md
│       ├── TrainValTestSplit.py           # Script to split dataset
│       ├── image_labels_with_majority.csv # Labeling breakdown CSV
│       ├── train.py                        # DINOv2 training script
│       └── train_dino_with_outputs.ipynb   # Notebook with training + visualizations
├── README.md  ← (this file)
```

## Technologies Used

- YOLOv12 (Ultralytics)
- DINOv2 Vision Transformer (Meta AI)
- Hugging Face Datasets
- PyTorch, torchvision
- sklearn, matplotlib, tqdm

---

## Sample Visualizations
- Classification model
<p align="center"> <img src="https://raw.githubusercontent.com/sampath-balaji/Electrical-Lines-Defect-Detection/refs/heads/main/Pole_LeanedStraight_Defect/Classification/assets/val_viz_ep40.png" width="600"/> </p>

- Object Detection model
  <p align="center">
  <img src="https://raw.githubusercontent.com/sampath-balaji/Electrical-Lines-Defect-Detection/refs/heads/main/Pole_LeanedStraight_Defect/ObjectDetection/assets/output.jpeg" width="600"/>
</p>

## License

- **Code:** MIT License  
- **Datasets:** CC BY 4.0
