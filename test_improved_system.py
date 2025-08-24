#!/usr/bin/env python3
"""
Test Improved Car Damage Detection System

This script tests the current model and provides recommendations.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True

def find_test_images():
    """Find test images in the dataset"""
    test_dirs = [
        "data/yolov8_data/21apr1000dataInsCorr-20220504T101259Z-001/21apr1000dataInsCorr/data/test",
        "data/yolov8_data/21apr1000dataInsCorr-20220504T101259Z-001/21apr1000dataInsCorr/data/obj"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            test_path = Path(test_dir)
            images = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
            if images:
                return images[:10]  # Return first 10 images
    
    return []

def test_model_performance(model_path, test_images):
    """Test model on sample images"""
    print(f"Testing model: {os.path.basename(model_path)}")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    detections_found = 0
    total_confidence = 0
    images_with_detections = 0
    
    print(f"Testing on {len(test_images)} images...")
    
    for i, img_path in enumerate(test_images):
        try:
            # Test with different confidence thresholds
            for conf in [0.25, 0.15, 0.05]:
                results = model.predict(
                    source=str(img_path),
                    conf=conf,
                    iou=0.4,
                    save=False,
                    verbose=False
                )
                
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        detections_found += len(boxes)
                        images_with_detections += 1
                        
                        for box in boxes:
                            total_confidence += float(box.conf[0])
                        
                        print(f"  Image {i+1}: {len(boxes)} detections (conf={conf})")
                        break
                else:
                    if conf == 0.05:
                        print(f"  Image {i+1}: No detections")
        
        except Exception as e:
            print(f"  Error processing image {i+1}: {e}")
    
    # Calculate metrics
    detection_rate = images_with_detections / len(test_images) if test_images else 0
    avg_confidence = total_confidence / detections_found if detections_found > 0 else 0
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"  Detection Rate: {detection_rate:.1%} ({images_with_detections}/{len(test_images)})")
    print(f"  Total Detections: {detections_found}")
    print(f"  Average Confidence: {avg_confidence:.3f}")
    
    # Performance assessment
    if detection_rate < 0.3:
        print("❌ Poor performance - model needs improvement")
        return False
    elif detection_rate < 0.6:
        print("⚠️ Moderate performance - could be improved")
        return True
    else:
        print("✅ Good performance")
        return True

def main():
    """Main test function"""
    print("🚗 Testing Car Damage Detection System")
    print("=" * 50)
    
    # Find available models
    model_paths = [
        "runs/train/yolov8_damage_detection_improved/weights/best.pt",
        "runs/train/yolov8_damage_detection11/weights/best.pt",
        "runs/train/yolov8_damage_detection11/weights/last.pt",
        "yolov8n.pt"
    ]
    
    available_models = [path for path in model_paths if os.path.exists(path)]
    
    if not available_models:
        print("❌ No models found!")
        print("\n🚀 RECOMMENDATION: Train a new model")
        print("Run: python train_improved_model.py")
        return
    
    # Find test images
    test_images = find_test_images()
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"Found {len(test_images)} test images")
    print(f"Found {len(available_models)} models")
    
    # Test each model
    best_model = None
    best_performance = False
    
    for model_path in available_models:
        print(f"\n{'='*30}")
        performance = test_model_performance(model_path, test_images)
        
        if performance and best_model is None:
            best_model = model_path
            best_performance = True
    
    # Recommendations
    print(f"\n{'='*50}")
    print("🎯 RECOMMENDATIONS:")
    
    if best_model:
        print(f"✅ Best model found: {os.path.basename(best_model)}")
        print("🚀 Ready to use in Streamlit app!")
        print("\nTo run the app:")
        print("  streamlit run improved_app.py")
    else:
        print("❌ No good models found")
        print("🚀 RECOMMENDED: Train improved model")
        print("\nTo train:")
        print("  python train_improved_model.py")
    
    print(f"\n📱 Available commands:")
    print(f"  python train_improved_model.py  # Train new model")
    print(f"  streamlit run improved_app.py   # Run web app")
    print(f"  python test_improved_system.py  # Run this test")

if __name__ == "__main__":
    main()
