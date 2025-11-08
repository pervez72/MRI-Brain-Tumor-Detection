# Brain Tumor Detection & Classification using YOLOv8

This repository contains a **Deep Learning–based Brain Tumor Detection system** using **YOLOv8**, capable of identifying and classifying four types of brain tumors from MRI scans:
> **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

The model is trained and evaluated on a curated MRI dataset using **Ultralytics YOLOv8**, achieving **high accuracy**, **robust generalization**, and **real-time detection speed**.

## Project Structure
```
tumor-detection/
│
├── README.md                 # Project overview, setup, usage, and results
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignore unnecessary files
│
├── data/
│   ├── dataset/  
│   
│
├── notebooks/
│   └── training.ipynb        # Google Colab notebook used for training
│
├── models/
│   └── yolov8n.pt            # Trained YOLOv8 model weights
│
├── src/                      # Source code & utility scripts
│   ├── train.py              # Model training pipeline
│   ├── detect.py             # Inference script for testing
│   └── utils.py              # Helper functions and utilities
│
└── outputs/
├── predictions/          # Predicted output images
└── figures/              # Confusion matrix, performance plots, etc.

```

---
## 🚀 Model Performance

| Metric | Value | Description |
| :--- | :---: | :--- |
| **mAP50 (B)** | **> 0.90** | Mean Average Precision @ 0.5 IoU |
| **mAP50–95 (B)** | **≈ 0.60** | Averaged precision over IoU thresholds |
| **Precision (B)** | **> 0.90** | Accuracy of tumor detections |
| **Recall (B)** | **> 0.90** | Proportion of correctly identified tumors |

All losses (`box_loss`, `cls_loss`, `dfl_loss`) decreased steadily, and metrics (`mAP50`, `recall`) improved smoothly — confirming **strong convergence** and **no overfitting**.

---

## Classification Accuracy (Confusion Matrix)

| Class | Recall | Precision |
| :--- | :---: | :---: |
| **Pituitary** | 0.97 | 0.95 |
| **No Tumor** | 0.94 | 0.93 |
| **Glioma** | 0.93 | 0.92 |
| **Meningioma** | 0.92 | 0.91 |

The model shows minimal confusion across tumor types.  
The main challenge is balancing **false positives** vs **false negatives** — common in medical imaging tasks.

---

---
## Dataset
| Class      | Count |
|------------|-------|
| Glioma     | 1,749 |
| Pituitary  | 1,320 |
| Meningioma | 1,077 |
| No Tumor   | 1,005 |

---

---
## Sample Output
![Sample Tumor Detection](output/prediction/output.png)

*Bounding box + class label + confidence score (e.g., Glioma, 0.89)*

---



## Quick Start

```bash
git clone https://github.com/your-username/tumor-detection.git
cd tumor-detection
pip install -r requirements.txt
```
---
## Future Improvements

* **Enhance Bounding Box Precision:** Apply stronger augmentations (rotation, scaling).
* **Reduce False Positives:** Add more background/no-tumor examples.
* **Hyperparameter Tuning:** Experiment with learning rate, confidence threshold, and NMS.
* **Explainability:** Integrate Grad-CAM for visual interpretability.

---

## 🧑‍💻 Author
**Md Pervez Hasan**
---
## License
This project is licensed under the **MIT License**.
Feel free to use, modify, and distribute with proper attribution.
---























