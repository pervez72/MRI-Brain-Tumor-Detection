

import cv2
import matplotlib.pyplot as plt
import glob
from PIL import Image
import os

def display_image(image_path):
    """
    Reads and displays a single image using matplotlib.
    (From notebook cell 16)
    
    Args:
        image_path (str): The file path to the image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path not found: {image_path}")
        return
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.title(f"Displaying: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()

def plot_training_results(train_dir='runs/detect/train'):
 
    print(f"Attempting to plot results from: {train_dir}")
    
    # Path to the results chart
    results_png = os.path.join(train_dir, 'results.png')
    # Path to a validation batch prediction
    val_pred_jpg = os.path.join(train_dir, 'val_batch0_pred.jpg')
    
    if os.path.exists(results_png):
        print("Displaying training results chart...")
        img_results = Image.open(results_png)
        plt.figure(figsize=(15, 10))
        plt.imshow(img_results)
        plt.title('Training Results (results.png)')
        plt.axis('off')
        plt.show()
    else:
        print(f"Warning: Could not find results chart at: {results_png}")

    if os.path.exists(val_pred_jpg):
        print("Displaying validation batch prediction...")
        img_val = Image.open(val_pred_jpg)
        plt.figure(figsize=(15, 10))
        plt.imshow(img_val)
        plt.title('Validation Batch Prediction (val_batch0_pred.jpg)')
        plt.axis('off')
        plt.show()
    else:
        print(f"Warning: Could not find validation batch at: {val_pred_jpg}")

def display_detection_results(predict_dir='runs/detect/predict'):
    """
    Finds and displays all .jpg images from a YOLOv8 prediction directory.
    (From notebook cell 15)
    
    Args:
        predict_dir (str): Path to the prediction run directory
                           (e.g., 'runs/detect/predict').
    """
    print(f"Displaying detection results from: {predict_dir}")
    
    # Find all JPG images in the prediction directory
    predicted_images = glob.glob(os.path.join(predict_dir, '*.jpg'))
    
    if not predicted_images:
        print(f"No .jpg images found in {predict_dir}")
        return

    print(f"Found {len(predicted_images)} predicted images.")
    
    for img_path in predicted_images:
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {img_path}: could not be read by cv2.")
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(img_rgb)
            plt.title(f"Prediction: {os.path.basename(img_path)}")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error displaying image {img_path}: {e}")
