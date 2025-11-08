

from ultralytics import YOLO
import argparse
import os
import utils  # Import the utility functions

def main(args):
    """
    Runs the main detection pipeline.
    """
    
    # Check if model weights exist
    if not os.path.exists(args.model_weights):
        print(f"Error: Model weights not found at: {args.model_weights}")
        return
        
    # Check if source directory/image exists
    if not os.path.exists(args.source):
        print(f"Error: Source path not found at: {args.source}")
        return

    # Load the trained model
    try:
        model = YOLO(args.model_weights)
        print(f"Successfully loaded trained model: {args.model_weights}")
    except Exception as e:
        print(f"Error loading model {args.model_weights}: {e}")
        return

    # Run prediction
    # (From notebook cell 13, converted to Python)
    try:
        print(f"Running prediction on source: {args.source}...")
        model.predict(
            source=args.source,
            save=True,  # Save prediction results (images with bboxes)
            name=args.name  # Save to runs/detect/[name]
        )
        print("Prediction complete.")
        
        # Display the results
        # (From notebook cell 15, now in utils)
        predict_run_dir = os.path.join('runs/detect', args.name)
        if os.path.exists(predict_run_dir):
            utils.display_detection_results(predict_run_dir)
        else:
            print(f"Could not find prediction run directory: {predict_run_dir}")
            
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detection with a trained YOLOv8 model")
    
    parser.add_argument('--model_weights', type=str, 
                        default='runs/detect/train/weights/best.pt', 
                        help="Path to the trained model weights (e.g., best.pt).")
    
    parser.add_argument('--source', type=str, 
                        default='/content/dataset/test/images', 
                        help="Path to the source directory or image for prediction.")
    
    parser.add_argument('--name', type=str, 
                        default='predict', 
                        help="Name for the prediction run (logs saved to runs/detect/[name]).")
    
    args = parser.parse_args()
    main(args)
