#!/usr/bin/env python3
"""
Improved YOLOv8 Car Damage Detection Trainer

This script trains an improved YOLOv8 model with better parameters for car damage detection.
"""

import os
import yaml
import shutil
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("Installing required packages...")
    os.system("pip install ultralytics torch torchvision")
    from ultralytics import YOLO
    import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedCarDamageTrainer:
    """Improved trainer for car damage detection"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_path = self.project_root / "data/yolov8_data/21apr1000dataInsCorr-20220504T101259Z-001/21apr1000dataInsCorr/data"
        self.output_dir = self.project_root / "improved_dataset"
        self.classes = ["scratch", "dent", "glass-shatter", "smash"]
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

    def prepare_dataset(self):
        """Prepare improved dataset with proper splits"""
        logger.info("Preparing improved dataset...")
        
        # Create dataset structure
        dataset_dir = self.output_dir / "car_damage_yolo"
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                (dataset_dir / split / subdir).mkdir(parents=True, exist_ok=True)
        
        # Get all valid image-label pairs
        obj_dir = self.data_path / "obj"
        if not obj_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {obj_dir}")
        
        # Find all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(obj_dir.glob(ext)))
        
        # Filter valid pairs (images with corresponding labels)
        valid_pairs = []
        for img_file in image_files:
            label_file = obj_dir / f"{img_file.stem}.txt"
            if label_file.exists() and label_file.stat().st_size > 0:
                valid_pairs.append((img_file, label_file))
        
        logger.info(f"Found {len(valid_pairs)} valid image-label pairs")
        
        if len(valid_pairs) < 50:
            logger.warning("Very few valid pairs found. Training may not be effective.")
        
        # Split dataset: 70% train, 20% val, 10% test
        train_pairs, temp_pairs = train_test_split(valid_pairs, test_size=0.3, random_state=42)
        val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.33, random_state=42)
        
        splits = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
        
        logger.info(f"Dataset splits: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
        
        # Copy files to new structure
        for split_name, pairs in splits.items():
            for img_file, label_file in pairs:
                # Copy image
                dst_img = dataset_dir / split_name / 'images' / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Copy label
                dst_label = dataset_dir / split_name / 'labels' / label_file.name
                shutil.copy2(label_file, dst_label)
        
        # Create data.yaml
        data_yaml = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = dataset_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"Dataset prepared at: {dataset_dir}")
        return dataset_dir

    def train_model(self, model_size='yolov8n', epochs=150):
        """Train improved YOLOv8 model"""
        logger.info(f"Training {model_size} model for {epochs} epochs...")
        
        # Prepare dataset
        dataset_dir = self.prepare_dataset()
        
        # Initialize model
        model = YOLO(f'{model_size}.pt')
        
        # Train with improved parameters
        model.train(
            data=str(dataset_dir / 'data.yaml'),
            epochs=epochs,
            batch=16,
            imgsz=640,
            patience=30,
            save_period=10,
            device='0' if torch.cuda.is_available() else 'cpu',
            workers=8,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            label_smoothing=0.0,
            nbs=64,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=True,
            save_json=True,
            amp=True,
            fraction=1.0,
            multi_scale=True,
            close_mosaic=10,
            cos_lr=True,
            deterministic=True,
            seed=42,
            verbose=True,
            project='runs/train',
            name='yolov8_damage_detection_improved',
            exist_ok=True,
            # Enhanced augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15.0,      # Increased rotation
            translate=0.2,     # Increased translation
            scale=0.9,         # Scale variation
            shear=5.0,         # Shear augmentation
            perspective=0.0001,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,         # MixUp augmentation
            copy_paste=0.1,    # Copy-paste augmentation
            auto_augment='randaugment',
            erasing=0.4
        )
        
        # Find best model
        model_dir = Path('runs/train/yolov8_damage_detection_improved')
        best_model = model_dir / 'weights' / 'best.pt'
        
        if best_model.exists():
            logger.info(f"Training completed! Best model: {best_model}")
            
            # Validate model
            model_best = YOLO(str(best_model))
            val_results = model_best.val(
                data=str(dataset_dir / 'data.yaml'),
                split='test',
                imgsz=640,
                batch=16,
                conf=0.001,
                iou=0.6,
                max_det=300,
                plots=True,
                save_json=True
            )
            
            logger.info("=== MODEL PERFORMANCE ===")
            logger.info(f"mAP50: {val_results.box.map50:.3f}")
            logger.info(f"mAP50-95: {val_results.box.map:.3f}")
            logger.info(f"Precision: {val_results.box.mp:.3f}")
            logger.info(f"Recall: {val_results.box.mr:.3f}")
            
            return str(best_model)
        else:
            logger.error("Training failed - no model found")
            return None

def main():
    """Main training function"""
    trainer = ImprovedCarDamageTrainer()
    
    # Train model
    model_path = trainer.train_model('yolov8n', 150)
    
    if model_path:
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Best model saved at: {model_path}")
        print(f"🚀 Ready to use in the Streamlit app!")
    else:
        print("❌ Training failed")

if __name__ == "__main__":
    main()
