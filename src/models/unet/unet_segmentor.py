"""
U-Net Medical Image Segmentation

This module implements U-Net architecture for medical image segmentation.
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import logging

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.metrics import MeanIoU
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logging.warning("TensorFlow not available. U-Net functionality will be limited.")

logger = logging.getLogger(__name__)

class UNetSegmentor:
    """U-Net model for medical image segmentation"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize U-Net segmentor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.unet_config = config['models']['unet']

        # Model parameters
        self.model = None
        self.input_size = self.unet_config.get('input_size', [256, 256, 3])
        self.num_classes = self.unet_config.get('num_classes', 2)

        # Load or create model
        self._load_or_create_model()

    def _load_or_create_model(self):
        """Load existing model or create new one"""
        if not HAS_TENSORFLOW:
            logger.error("TensorFlow not available")
            return

        try:
            model_path = self.unet_config.get('model_path', 'models/unet/unet_model.h5')

            if os.path.exists(model_path):
                # Load existing model
                self.model = load_model(model_path)
                logger.info(f"U-Net model loaded from {model_path}")
            else:
                # Create new model
                self.model = self._build_unet_model()
                logger.info("New U-Net model created")

        except Exception as e:
            logger.error(f"Error loading/creating U-Net model: {e}")
            self.model = self._build_unet_model()

    def _build_unet_model(self) -> Model:
        """Build U-Net model architecture"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required to build U-Net model")

        try:
            # Input layer
            inputs = Input(self.input_size)

            # Encoder (Contracting Path)
            conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
            conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
            conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
            conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
            conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

            # Bridge
            conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
            conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
            drop5 = Dropout(0.5)(conv5)

            # Decoder (Expanding Path)
            up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
            merge6 = concatenate([drop4, up6], axis=3)
            conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
            conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

            up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
            merge7 = concatenate([conv3, up7], axis=3)
            conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
            conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

            up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
            merge8 = concatenate([conv2, up8], axis=3)
            conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
            conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

            up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
            merge9 = concatenate([conv1, up9], axis=3)
            conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
            conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

            # Output layer
            if self.num_classes == 2:
                # Binary segmentation
                outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
            else:
                # Multi-class segmentation
                outputs = Conv2D(self.num_classes, 1, activation='softmax')(conv9)

            model = Model(inputs=inputs, outputs=outputs)

            return model

        except Exception as e:
            logger.error(f"Error building U-Net model: {e}")
            raise

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Perform segmentation on input image

        Args:
            image: Input image

        Returns:
            np.ndarray: Segmentation mask
        """
        if self.model is None:
            logger.error("U-Net model not available")
            return np.zeros_like(image)

        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Add batch dimension
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, axis=0)

            # Perform prediction
            prediction = self.model.predict(processed_image, verbose=0)

            # Postprocess prediction
            mask = self._postprocess_prediction(prediction[0])

            return mask

        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            return np.zeros_like(image)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for U-Net input"""
        try:
            # Resize to model input size
            target_height, target_width = self.input_size[:2]
            resized = tf.image.resize(image, [target_height, target_width])

            # Normalize to [0, 1]
            normalized = resized / 255.0

            return normalized.numpy()

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def _postprocess_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Postprocess model prediction"""
        try:
            if self.num_classes == 2:
                # Binary segmentation
                mask = (prediction > 0.5).astype(np.uint8) * 255
            else:
                # Multi-class segmentation
                mask = np.argmax(prediction, axis=-1).astype(np.uint8)

            # Convert to RGB if needed
            if len(mask.shape) == 2:
                mask = np.stack([mask] * 3, axis=-1)

            return mask

        except Exception as e:
            logger.error(f"Error postprocessing prediction: {e}")
            raise

    def train(self, train_data: Tuple[np.ndarray, np.ndarray],
              val_data: Tuple[np.ndarray, np.ndarray] = None,
              epochs: int = None, batch_size: int = None) -> Dict[str, Any]:
        """
        Train the U-Net model

        Args:
            train_data: Tuple of (images, masks) for training
            val_data: Tuple of (images, masks) for validation
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Dict[str, Any]: Training history
        """
        if self.model is None:
            logger.error("U-Net model not available")
            return {}

        try:
            if epochs is None:
                epochs = self.unet_config.get('epochs', 100)

            if batch_size is None:
                batch_size = self.unet_config.get('batch_size', 16)

            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=self.unet_config.get('learning_rate', 0.001)),
                loss=self._get_loss_function(),
                metrics=self._get_metrics()
            )

            # Prepare data
            X_train, y_train = train_data

            # Setup callbacks
            callbacks = self._get_callbacks()

            # Training
            if val_data is not None:
                X_val, y_val = val_data
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                history = self.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1
                )

            return history.history

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {}

    def _get_loss_function(self):
        """Get appropriate loss function"""
        if self.num_classes == 2:
            return 'binary_crossentropy'
        else:
            return 'categorical_crossentropy'

    def _get_metrics(self) -> List:
        """Get evaluation metrics"""
        metrics = ['accuracy']
        if HAS_TENSORFLOW:
            metrics.append(MeanIoU(num_classes=self.num_classes))
        return metrics

    def _get_callbacks(self) -> List:
        """Get training callbacks"""
        callbacks = []

        if HAS_TENSORFLOW:
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss' if self.model else 'loss',
                patience=self.config.get('training', {}).get('unet', {}).get('early_stopping_patience', 10),
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

            # Model checkpoint
            checkpoint_path = "models/unet/checkpoints/model_{epoch:02d}_{val_loss:.4f}.h5"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            model_checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss' if self.model else 'loss',
                save_best_only=True,
                save_weights_only=False
            )
            callbacks.append(model_checkpoint)

            # Learning rate reduction
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss' if self.model else 'loss',
                factor=self.config.get('training', {}).get('unet', {}).get('reduce_lr_factor', 0.5),
                patience=self.config.get('training', {}).get('unet', {}).get('reduce_lr_patience', 5),
                min_lr=1e-7
            )
            callbacks.append(reduce_lr)

        return callbacks

    def save_model(self, save_path: str = None) -> bool:
        """
        Save the model

        Args:
            save_path: Path to save the model

        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No model to save")
            return False

        try:
            if save_path is None:
                save_path = self.unet_config.get('model_path', 'models/unet/unet_model.h5')

            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            self.model.save(save_path)
            logger.info(f"Model saved successfully: {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def calculate_metrics(self, image: np.ndarray, mask: np.ndarray,
                         ground_truth: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate segmentation metrics

        Args:
            image: Original image
            mask: Predicted mask
            ground_truth: Ground truth mask

        Returns:
            Dict[str, float]: Performance metrics
        """
        try:
            if ground_truth is None:
                # Basic statistics
                return {
                    'mask_mean': float(mask.mean()),
                    'mask_std': float(mask.std()),
                    'mask_min': float(mask.min()),
                    'mask_max': float(mask.max())
                }

            # Calculate segmentation metrics
            if self.num_classes == 2:
                # Binary segmentation
                pred_binary = (mask > 127).astype(np.uint8)
                gt_binary = (ground_truth > 127).astype(np.uint8)

                # Dice coefficient
                intersection = np.logical_and(pred_binary, gt_binary).sum()
                dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum())

                # IoU
                union = np.logical_or(pred_binary, gt_binary).sum()
                iou = intersection / union if union > 0 else 0

                # Pixel accuracy
                accuracy = np.mean(pred_binary == gt_binary)

                return {
                    'dice_coefficient': float(dice),
                    'iou': float(iou),
                    'accuracy': float(accuracy)
                }
            else:
                # Multi-class segmentation - simplified
                return {
                    'mean_accuracy': float(np.mean(mask == ground_truth)),
                    'num_classes': self.num_classes
                }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'U-Net',
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'architecture': 'U-Net',
            'framework': 'TensorFlow/Keras' if HAS_TENSORFLOW else 'Not Available'
        }