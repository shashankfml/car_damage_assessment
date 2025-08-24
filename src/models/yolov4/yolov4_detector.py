"""
YOLOv4 Object Detection for Car Damage Detection

This module implements YOLOv4 object detection for identifying car damage.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class YOLOv4Detector:
    """YOLOv4 object detector for car damage detection"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YOLOv4 detector

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.yolo_config = config['models']['yolov4']

        # Model parameters
        self.net = None
        self.classes = []
        self.layer_names = []
        self.output_layers = []

        # Load model
        self._load_model()

    def _load_model(self):
        """Load YOLOv4 model and configuration"""
        try:
            # Check if model files exist
            config_path = self.yolo_config['config_path']
            weights_path = self.yolo_config['weights_path']
            names_path = self.yolo_config['names_path']

            if not os.path.exists(config_path):
                logger.warning(f"YOLOv4 config file not found: {config_path}")
                return

            if not os.path.exists(weights_path):
                logger.warning(f"YOLOv4 weights file not found: {weights_path}")
                return

            # Load class names
            if os.path.exists(names_path):
                with open(names_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
            else:
                logger.warning(f"Class names file not found: {names_path}")
                self.classes = ['damage']  # Default class

            # Load the network
            self.net = cv2.dnn.readNet(weights_path, config_path)

            # Get layer names
            self.layer_names = self.net.getLayerNames()

            # Get output layers
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

            logger.info("YOLOv4 model loaded successfully")
            logger.info(f"Classes: {self.classes}")

        except Exception as e:
            logger.error(f"Error loading YOLOv4 model: {e}")
            raise

    def detect(self, image: np.ndarray, confidence_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Detect objects in the image

        Args:
            image: Input image
            confidence_threshold: Confidence threshold for detection

        Returns:
            List[Dict[str, Any]]: List of detected objects
        """
        if self.net is None:
            logger.error("YOLOv4 model not loaded")
            return []

        try:
            if confidence_threshold is None:
                confidence_threshold = self.yolo_config.get('confidence_threshold', 0.5)

            # Prepare image for detection
            blob = self._prepare_image(image)

            # Set input
            self.net.setInput(blob)

            # Forward pass
            outputs = self.net.forward(self.output_layers)

            # Process detections
            detections = self._process_outputs(outputs, image.shape, confidence_threshold)

            return detections

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []

    def _prepare_image(self, image: np.ndarray) -> cv2.dnn.Net:
        """Prepare image for YOLOv4 input"""
        try:
            input_size = self.yolo_config.get('input_size', 416)

            # Resize image
            resized = cv2.resize(image, (input_size, input_size))

            # Create blob
            blob = cv2.dnn.blobFromImage(
                resized,
                scalefactor=1/255.0,
                size=(input_size, input_size),
                swapRB=True,
                crop=False
            )

            return blob

        except Exception as e:
            logger.error(f"Error preparing image: {e}")
            raise

    def _process_outputs(self, outputs: List[np.ndarray], image_shape: Tuple[int, int, int],
                        confidence_threshold: float) -> List[Dict[str, Any]]:
        """Process YOLOv4 outputs"""
        try:
            height, width = image_shape[:2]
            boxes = []
            confidences = []
            class_ids = []

            # Process each output
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > confidence_threshold:
                        # Scale bounding box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression
            nms_threshold = self.yolo_config.get('nms_threshold', 0.4)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

            detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    confidence = confidences[i]
                    class_id = class_ids[i]

                    detection = {
                        'class': self.classes[class_id] if class_id < len(self.classes) else f'class_{class_id}',
                        'confidence': confidence,
                        'bbox': {
                            'x': max(0, x),
                            'y': max(0, y),
                            'width': w,
                            'height': h
                        },
                        'class_id': class_id
                    }

                    detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error processing outputs: {e}")
            return []

    def draw_boxes(self, image: np.ndarray, detections: List[Dict[str, Any]],
                   color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes on image

        Args:
            image: Input image
            detections: List of detections
            color: Box color (BGR)
            thickness: Box thickness

        Returns:
            np.ndarray: Image with bounding boxes
        """
        try:
            image_copy = image.copy()

            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class']

                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

                # Draw bounding box
                cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)

                # Draw label
                label = ".2f"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                # Draw label background
                cv2.rectangle(
                    image_copy,
                    (x, y - label_height - baseline),
                    (x + label_width, y),
                    color,
                    cv2.FILLED
                )

                # Draw label text
                cv2.putText(
                    image_copy,
                    label,
                    (x, y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )

            return image_copy

        except Exception as e:
            logger.error(f"Error drawing boxes: {e}")
            return image

    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.classes.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'YOLOv4',
            'num_classes': len(self.classes),
            'classes': self.classes,
            'input_size': self.yolo_config.get('input_size', 416),
            'confidence_threshold': self.yolo_config.get('confidence_threshold', 0.5),
            'nms_threshold': self.yolo_config.get('nms_threshold', 0.4)
        }

    def calculate_metrics(self, detections: List[Dict[str, Any]],
                         ground_truth: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate detection metrics

        Args:
            detections: List of detections
            ground_truth: List of ground truth annotations

        Returns:
            Dict[str, float]: Performance metrics
        """
        try:
            if ground_truth is None:
                # Calculate basic statistics
                return {
                    'num_detections': len(detections),
                    'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0,
                    'max_confidence': max([d['confidence'] for d in detections]) if detections else 0.0,
                    'min_confidence': min([d['confidence'] for d in detections]) if detections else 0.0
                }

            # Calculate precision, recall, mAP if ground truth is available
            # This is a simplified implementation
            tp = 0
            fp = 0
            fn = len(ground_truth)

            for detection in detections:
                best_iou = 0
                best_gt_idx = -1

                for i, gt in enumerate(ground_truth):
                    iou = self._calculate_iou(detection['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                if best_iou > 0.5:  # IoU threshold
                    tp += 1
                    fn -= 1
                else:
                    fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def _calculate_iou(self, bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
        """Calculate Intersection over Union (IoU)"""
        try:
            x1, y1 = bbox1['x'], bbox1['y']
            w1, h1 = bbox1['width'], bbox1['height']
            x2, y2 = bbox2['x'], bbox2['y']
            w2, h2 = bbox2['width'], bbox2['height']

            # Calculate intersection coordinates
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)

            if x_right < x_left or y_bottom < y_top:
                return 0.0

            # Calculate intersection area
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # Calculate union area
            bbox1_area = w1 * h1
            bbox2_area = w2 * h2
            union_area = bbox1_area + bbox2_area - intersection_area

            # Calculate IoU
            iou = intersection_area / union_area if union_area > 0 else 0.0

            return iou

        except Exception as e:
            logger.error(f"Error calculating IoU: {e}")
            return 0.0