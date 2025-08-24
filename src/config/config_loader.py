"""
Configuration loader for the Medical Image Analysis & Car Damage Detection System

This module handles loading and managing configuration settings from YAML files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Configuration loader class"""

    @staticmethod
    def load_config(config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Args:
            config_path (str): Path to the configuration file

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent / "config.yaml"

        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            logger.info(f"Configuration loaded successfully from {config_path}")
            return config

        except FileNotFoundError:
            logger.warning(f"Configuration file not found at {config_path}, using default config")
            return ConfigLoader._get_default_config()

        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return ConfigLoader._get_default_config()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration settings"""
        return {
            "project": {
                "name": "Medical Image Analysis & Car Damage Detection System",
                "version": "1.0.0",
                "description": "Capstone project combining medical imaging and vehicle damage detection"
            },
            "models": {
                "yolov4": {
                    "config_path": "models/yolov4/yolov4-custom.cfg",
                    "weights_path": "models/weights/yolov4-custom.weights",
                    "data_path": "models/yolov4/obj.data",
                    "names_path": "models/yolov4/obj.names",
                    "input_size": 416,
                    "confidence_threshold": 0.5,
                    "nms_threshold": 0.4
                },
                "unet": {
                    "input_size": [256, 256, 3],
                    "num_classes": 2,
                    "learning_rate": 0.001,
                    "batch_size": 16,
                    "epochs": 100,
                    "model_path": "models/unet/unet_model.h5"
                }
            },
            "evaluation": {
                "thresholds": {
                    "iou_threshold": 0.5,
                    "confidence_threshold": 0.5
                }
            }
        }

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> bool:
        """
        Save configuration to YAML file

        Args:
            config (Dict[str, Any]): Configuration dictionary
            config_path (str): Path to save the configuration

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved successfully to {config_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    @staticmethod
    def update_config(updates: Dict[str, Any], config_path: str = None) -> Dict[str, Any]:
        """
        Update existing configuration with new values

        Args:
            updates (Dict[str, Any]): Updates to apply
            config_path (str): Path to the configuration file

        Returns:
            Dict[str, Any]: Updated configuration
        """
        config = ConfigLoader.load_config(config_path)

        def deep_update(original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
            """Deep update dictionary"""
            for key, value in updates.items():
                if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                    deep_update(original[key], value)
                else:
                    original[key] = value
            return original

        updated_config = deep_update(config, updates)

        if config_path:
            ConfigLoader.save_config(updated_config, config_path)

        return updated_config

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure

        Args:
            config (Dict[str, Any]): Configuration to validate

        Returns:
            bool: True if valid, False otherwise
        """
        required_keys = ["project", "models"]

        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                return False

        # Validate model configurations
        if "yolov4" not in config.get("models", {}):
            logger.warning("YOLOv4 configuration not found")
        else:
            yolov4_config = config["models"]["yolov4"]
            required_yolo_keys = ["config_path", "weights_path", "input_size"]
            for key in required_yolo_keys:
                if key not in yolov4_config:
                    logger.error(f"Missing YOLOv4 configuration key: {key}")
                    return False

        if "unet" not in config.get("models", {}):
            logger.warning("U-Net configuration not found")
        else:
            unet_config = config["models"]["unet"]
            required_unet_keys = ["input_size", "num_classes"]
            for key in required_unet_keys:
                if key not in unet_config:
                    logger.error(f"Missing U-Net configuration key: {key}")
                    return False

        return True