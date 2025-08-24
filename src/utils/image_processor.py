"""
Image processing utilities for the Medical Image Analysis & Car Damage Detection System

This module provides image loading, preprocessing, and utility functions for both
medical images and regular images.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Union, Tuple, Optional, Any
import logging
from PIL import Image
import io

try:
    import SimpleITK as sitk
    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False
    logging.warning("SimpleITK not available. Medical image loading will be limited.")

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    logging.warning("Nibabel not available. NRRD file loading will be limited.")

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processing utility class"""

    # Supported image formats
    SUPPORTED_FORMATS = {
        'regular': ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
        'medical': ['nrrd', 'dicom', 'dcm', 'nii', 'nii.gz']
    }

    def __init__(self):
        """Initialize the image processor"""
        self.logger = logging.getLogger(__name__)

    def load_image(self, image_path: Union[str, Path, bytes], format: str = 'auto') -> np.ndarray:
        """
        Load an image from file path or bytes

        Args:
            image_path: Path to image file or image bytes
            format: Image format ('auto', 'regular', 'medical')

        Returns:
            np.ndarray: Loaded image as numpy array
        """
        try:
            if isinstance(image_path, bytes):
                return self._load_from_bytes(image_path)
            elif isinstance(image_path, (str, Path)):
                return self._load_from_path(str(image_path), format)
            else:
                raise ValueError("Invalid image path type")

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise

    def _load_from_path(self, image_path: str, format: str = 'auto') -> np.ndarray:
        """Load image from file path"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Determine format
        if format == 'auto':
            format = self._detect_format(image_path)

        if format == 'medical':
            return self._load_medical_image(image_path)
        else:
            return self._load_regular_image(image_path)

    def _load_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes"""
        try:
            # Try to decode as regular image first
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # If that fails, try PIL
            image = Image.open(io.BytesIO(image_bytes))
            return np.array(image)

        except Exception as e:
            logger.error(f"Error loading image from bytes: {e}")
            raise

    def _load_regular_image(self, image_path: str) -> np.ndarray:
        """Load regular image formats (JPG, PNG, etc.)"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Convert BGR to RGB
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        except Exception as e:
            logger.error(f"Error loading regular image: {e}")
            raise

    def _load_medical_image(self, image_path: str) -> np.ndarray:
        """Load medical image formats (NRRD, DICOM, NII)"""
        try:
            file_extension = Path(image_path).suffix.lower()

            if file_extension in ['.nrrd']:
                return self._load_nrrd_image(image_path)
            elif file_extension in ['.dicom', '.dcm']:
                return self._load_dicom_image(image_path)
            elif file_extension in ['.nii', '.nii.gz']:
                return self._load_nii_image(image_path)
            else:
                raise ValueError(f"Unsupported medical image format: {file_extension}")

        except Exception as e:
            logger.error(f"Error loading medical image: {e}")
            raise

    def _load_nrrd_image(self, image_path: str) -> np.ndarray:
        """Load NRRD image format"""
        if not HAS_NIBABEL:
            raise ImportError("Nibabel is required for NRRD image loading")

        try:
            img = nib.load(image_path)
            data = img.get_fdata()

            # Handle different dimensions
            if data.ndim == 2:
                # Single slice
                return self._normalize_medical_image(data)
            elif data.ndim == 3:
                # Multi-slice, take middle slice
                middle_slice = data.shape[2] // 2
                return self._normalize_medical_image(data[:, :, middle_slice])
            elif data.ndim == 4:
                # Multi-volume, take first volume and middle slice
                middle_slice = data.shape[2] // 2
                return self._normalize_medical_image(data[:, :, middle_slice, 0])
            else:
                raise ValueError(f"Unsupported NRRD dimensions: {data.ndim}")

        except Exception as e:
            logger.error(f"Error loading NRRD image: {e}")
            raise

    def _load_dicom_image(self, image_path: str) -> np.ndarray:
        """Load DICOM image format"""
        if not HAS_SIMPLEITK:
            raise ImportError("SimpleITK is required for DICOM image loading")

        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(image_path)
            image = reader.Execute()

            # Convert to numpy array
            array = sitk.GetArrayFromImage(image)

            # Handle different dimensions
            if array.ndim == 2:
                return self._normalize_medical_image(array)
            elif array.ndim == 3:
                # Multi-slice, take middle slice
                middle_slice = array.shape[0] // 2
                return self._normalize_medical_image(array[middle_slice])
            else:
                raise ValueError(f"Unsupported DICOM dimensions: {array.ndim}")

        except Exception as e:
            logger.error(f"Error loading DICOM image: {e}")
            raise

    def _load_nii_image(self, image_path: str) -> np.ndarray:
        """Load NII image format"""
        if not HAS_NIBABEL:
            raise ImportError("Nibabel is required for NII image loading")

        try:
            img = nib.load(image_path)
            data = img.get_fdata()

            # Handle different dimensions
            if data.ndim == 3:
                # Take middle slice
                middle_slice = data.shape[2] // 2
                return self._normalize_medical_image(data[:, :, middle_slice])
            elif data.ndim == 4:
                # Multi-volume, take first volume and middle slice
                middle_slice = data.shape[2] // 2
                return self._normalize_medical_image(data[:, :, middle_slice, 0])
            else:
                raise ValueError(f"Unsupported NII dimensions: {data.ndim}")

        except Exception as e:
            logger.error(f"Error loading NII image: {e}")
            raise

    def _normalize_medical_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize medical image for display"""
        try:
            # Convert to float
            image = image.astype(np.float32)

            # Normalize to 0-255 range
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min()) * 255
            else:
                image = np.zeros_like(image)

            # Convert to uint8
            image = image.astype(np.uint8)

            # If grayscale, convert to RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            return image

        except Exception as e:
            logger.error(f"Error normalizing medical image: {e}")
            raise

    def _detect_format(self, image_path: str) -> str:
        """Detect image format based on file extension"""
        extension = Path(image_path).suffix.lower().lstrip('.')

        if extension in self.SUPPORTED_FORMATS['medical']:
            return 'medical'
        elif extension in self.SUPPORTED_FORMATS['regular']:
            return 'regular'
        else:
            return 'regular'  # Default to regular

    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = None,
                        normalize: bool = True) -> np.ndarray:
        """
        Preprocess image for model input

        Args:
            image: Input image
            target_size: Target size (width, height)
            normalize: Whether to normalize pixel values

        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            # Resize if target size specified
            if target_size is not None:
                image = cv2.resize(image, target_size)

            # Normalize pixel values to [0, 1]
            if normalize:
                image = image.astype(np.float32) / 255.0

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def save_image(self, image: np.ndarray, save_path: Union[str, Path],
                   format: str = 'png') -> bool:
        """
        Save image to file

        Args:
            image: Image to save
            save_path: Path to save the image
            format: Image format

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)

            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            # Save image
            cv2.imwrite(str(save_path), image_bgr)
            logger.info(f"Image saved successfully: {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False

    def get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Get image information

        Args:
            image: Input image

        Returns:
            Dict[str, Any]: Image information
        """
        return {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min_value': float(image.min()),
            'max_value': float(image.max()),
            'mean_value': float(image.mean()),
            'channels': image.shape[2] if len(image.shape) == 3 else 1
        }

    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int],
                    keep_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize image

        Args:
            image: Input image
            target_size: Target size (width, height)
            keep_aspect_ratio: Whether to keep aspect ratio

        Returns:
            np.ndarray: Resized image
        """
        try:
            if keep_aspect_ratio:
                # Calculate aspect ratio
                height, width = image.shape[:2]
                target_width, target_height = target_size

                aspect_ratio = width / height
                target_aspect_ratio = target_width / target_height

                if aspect_ratio > target_aspect_ratio:
                    # Image is wider, fit to width
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:
                    # Image is taller, fit to height
                    new_height = target_height
                    new_width = int(target_height * aspect_ratio)

                resized = cv2.resize(image, (new_width, new_height))

                # Pad to target size
                pad_top = (target_height - new_height) // 2
                pad_bottom = target_height - new_height - pad_top
                pad_left = (target_width - new_width) // 2
                pad_right = target_width - new_width - pad_left

                padded = cv2.copyMakeBorder(
                    resized, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )

                return padded
            else:
                return cv2.resize(image, target_size)

        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            raise

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale

        Args:
            image: Input image

        Returns:
            np.ndarray: Grayscale image
        """
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                return image

        except Exception as e:
            logger.error(f"Error converting to grayscale: {e}")
            raise