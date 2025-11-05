# 🧠 Brain Tumor Detection using YOLOv11

This project uses a YOLOv11 model for detecting three types of brain tumors (Meningioma, Pituitary, Glioma) from MRI images.

## 🔍 Dataset
- Source: Custom labeled MRI Brain Tumor dataset (from Roboflow)
- Format: train/val/test splits

## ⚙️ Tech Stack
- Python
- Ultralytics (YOLOv11)
- Google Colab
- OpenCV

## 🚀 Steps
1. Data preprocessing (done on Google Drive)
2. Model training using YOLOv11
3. Validation and mAP evaluation
4. Prediction on unseen MRI scans

## 📦 Output
- Trained model: `yolo11n.pt`
- Predictions stored in `runs/detect/predict/`

## 📸 Example Results
![Sample](runs/detect/predict/sample_output.jpg)

---

**Author:** Md Pervez Hasan  
**Agency:** AsuX AI

