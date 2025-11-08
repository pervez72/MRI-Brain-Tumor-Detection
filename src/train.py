

import yaml
from ultralytics import YOLO
import argparse
import os
import utils  # Import the utility functions

def create_data_yaml(yaml_path='data.yaml', train_path='/content/dataset/train', val_path='/content/dataset/valid'):
    """
    Creates the data.yaml file required for YOLOv8 training.
    (From notebook cell 7)
    
    NOTE: Paths are hardcoded based on the notebook. You may want to
    change these paths or pass them as arguments.
    """
    data_yaml = {
        'train': train_path,
        'val': val_path,
        'nc': 1,  # Number of classes (e.g., 1 for 'brain_tumor')
        'names': ['brain_tumor']  # Class names
    }
    
    try:
        with open(yaml_path, 'w') as file:
            yaml.dump(data_yaml, file, default_flow_style=False)
        print(f"Successfully created '{yaml_path}'")
    except Exception as e:
        print(f"Error creating '{yaml_path}': {e}")

def main(args):
    """
    Runs the main training pipeline.
    """

    if not os.path.exists(args.data_config):
        print(f"'{args.data_config}' not found, creating a default one.")
        create_data_yaml(
            yaml_path=args.data_config,
            train_path=args.train_path,
            val_path=args.val_path
        )
    else:
        print(f"Using existing data config: '{args.data_config}'")

    # Load a base model
    # (From notebook cell 8)
    try:
        model = YOLO(args.base_model)
        print(f"Successfully loaded base model: {args.base_model}")
    except Exception as e:
        print(f"Error loading base model {args.base_model}: {e}")
        return

    # Train the model
    # (From notebook cell 9)
    try:
        print("Starting model training...")
        model.train(
            data=args.data_config,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name
        )
        print("Training complete.")
        
        # Plot results after training
        # (From notebook cell 11, now in utils)
        train_run_dir = os.path.join('runs/detect', args.name)
        if os.path.exists(train_run_dir):
            utils.plot_training_results(train_run_dir)
        else:
            print(f"Could not find training run directory: {train_run_dir}")
            
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model")
    
    # Arguments for paths
    parser.add_argument('--data_config', type=str, default='data.yaml', help="Path to data.yaml file.")
    parser.add_argument('--train_path', type=str, default='/content/dataset/train', help="Path to training images/labels.")
    parser.add_argument('--val_path', type=str, default='/content/dataset/valid', help="Path to validation images/labels.")
    
    # Arguments for model and training
    parser.add_argument('--base_model', type=str, default='yolov8n.pt', help="Base model checkpoint to start training from.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training.")
    parser.add_argument('--batch', type=int, default=8, help="Training batch size.")
    parser.add_argument('--name', type=str, default='yolov8n_brain_tumor', help="Name for the training run (logs saved to runs/detect/[name]).")
    
    args = parser.parse_args()
    main(args)
