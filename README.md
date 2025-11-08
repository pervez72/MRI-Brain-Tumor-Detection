
---

```markdown
# Brain Tumor Detection & Classification using YOLOv8

This repository contains a **Deep Learning–based Brain Tumor Detection system** using **YOLOv8**, capable of identifying and classifying four types of brain tumors from MRI scans:
> **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

The model is trained and evaluated on a curated MRI dataset using **Ultralytics YOLOv8**, achieving **high accuracy**, **robust generalization**, and **real-time detection speed**.

---

## Project Structure

```

tumor-detection/
│
├── README.md                 # Project overview, setup, usage, and results
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignore unnecessary files
│
├── data/
│   ├── raw/                  # Original dataset (unmodified)
│   ├── processed/            # Preprocessed train/test data
│   └── sample_output/        # Example detection results
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

🔹 The model shows minimal confusion across tumor types.  
🔹 The main challenge is balancing **false positives** vs **false negatives** — common in medical imaging tasks.

---

## Qualitative Results (Sample Predictions)

Predicted images are available in:
```

data/sample_output/
outputs/predictions/

````

Each image includes:
- Bounding box around tumor
- Class label (Glioma, Meningioma, Pituitary, No Tumor)
- Confidence score (0.7–0.9 typical range)

---

## Dataset Overview

The dataset contains MRI images labeled into four categories.  
It has slight imbalance — **Glioma** being most frequent.

| Class | Instances |
| :--- | :---: |
| Glioma | 1749 |
| Pituitary | 1320 |
| Meningioma | 1077 |
| No Tumor | 1005 |

Bounding boxes are generally **small and centered**.

---

## ⚙️ How to Use

### 🔧 Prerequisites

- Python **3.8+**
- **PyTorch**
- **Ultralytics YOLOv8**
- GPU recommended for training/inference

### 🧩 Installation

```bash
git clone https://github.com/your-username/tumor-detection.git
cd tumor-detection
pip install -r requirements.txt
````

### 🧠 Train Model

You can retrain the model using your dataset:

```bash
python src/train.py
```

Or open the Colab notebook:

```
notebooks/training.ipynb
```

### 🔍 Run Inference

To run the trained YOLOv8 model on any MRI image:

```bash
python src/detect.py --weights models/yolov8n.pt --source path/to/mri_image.jpg
```

Outputs are saved in:

```
outputs/predictions/
```

---

## 📈 Future Improvements

* **Enhance Bounding Box Precision:** Apply stronger augmentations (rotation, scaling).
* **Reduce False Positives:** Add more background/no-tumor examples.
* **Hyperparameter Tuning:** Experiment with learning rate, confidence threshold, and NMS.
* **Explainability:** Integrate Grad-CAM for visual interpretability.

---

## 🧑‍💻 Author

**Md Pervez Hasan**
Founder — [AsuX AI](https://github.com/AsuX-AI)
📍Bangladesh
   Focused on AI, ML, and solving real-world problems.

---

## License

This project is licensed under the **MIT License**.
Feel free to use, modify, and distribute with proper attribution.

---

> *“AI won’t replace people — but people using AI will replace those who don’t.”*

```

---

Would you like me to now create the **`requirements.txt`** file (with exact YOLOv8 + PyTorch dependencies) for your repo next?  
That will make your GitHub project **runnable directly** on any system or Colab.
```

