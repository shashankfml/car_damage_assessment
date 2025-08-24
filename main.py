#!/usr/bin/env python3
"""
Medical Image Analysis & Car Damage Detection System
Main Entry Point

This script serves as the main entry point for the application,
providing command-line interface for different modes of operation.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.config_loader import ConfigLoader
from src.utils.logger import setup_logger

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Medical Image Analysis & Car Damage Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode streamlit    # Run Streamlit web app
  python main.py --mode train-yolo   # Train YOLOv4 model
  python main.py --mode train-unet   # Train U-Net model
  python main.py --mode preprocess   # Run data preprocessing
        """
    )

    parser.add_argument(
        "--mode",
        choices=["streamlit", "train-yolo", "train-unet", "preprocess", "evaluate"],
        default="streamlit",
        help="Operation mode (default: streamlit)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger("main", log_level)

    # Load configuration
    try:
        config = ConfigLoader.load_config(args.config)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Execute based on mode
    if args.mode == "streamlit":
        run_streamlit_app(config)
    elif args.mode == "train-yolo":
        train_yolo_model(config)
    elif args.mode == "train-unet":
        train_unet_model(config)
    elif args.mode == "preprocess":
        run_preprocessing(config)
    elif args.mode == "evaluate":
        run_evaluation(config)

def run_streamlit_app(config):
    """Run Streamlit web application"""
    try:
        import streamlit.web.cli as st_cli
        import sys

        logger = setup_logger("streamlit", "INFO")
        logger.info("Starting Streamlit application...")

        # Set streamlit arguments
        sys.argv = [
            "streamlit", "run",
            "deployment/streamlit_app.py",
            "--server.port", str(config.get("deployment", {}).get("streamlit", {}).get("port", 8501)),
            "--server.address", config.get("deployment", {}).get("streamlit", {}).get("host", "0.0.0.0")
        ]

        # Run streamlit
        st_cli.main()

    except ImportError:
        print("Streamlit not installed. Please install it: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)

def train_yolo_model(config):
    """Train YOLOv4 model"""
    try:
        from src.models.yolov4.yolov4_trainer import YOLOv4Trainer

        logger = setup_logger("yolo_trainer", "INFO")
        logger.info("Starting YOLOv4 training...")

        trainer = YOLOv4Trainer(config)
        trainer.train()

    except ImportError as e:
        print(f"Error importing YOLOv4 trainer: {e}")
        print("Please ensure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error training YOLOv4 model: {e}")
        sys.exit(1)

def train_unet_model(config):
    """Train U-Net model"""
    try:
        from src.models.unet.unet_trainer import UNetTrainer

        logger = setup_logger("unet_trainer", "INFO")
        logger.info("Starting U-Net training...")

        trainer = UNetTrainer(config)
        trainer.train()

    except ImportError as e:
        print(f"Error importing U-Net trainer: {e}")
        print("Please ensure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error training U-Net model: {e}")
        sys.exit(1)

def run_preprocessing(config):
    """Run data preprocessing"""
    try:
        from src.data.data_preprocessor import DataPreprocessor

        logger = setup_logger("preprocessor", "INFO")
        logger.info("Starting data preprocessing...")

        preprocessor = DataPreprocessor(config)
        preprocessor.run()

    except ImportError as e:
        print(f"Error importing data preprocessor: {e}")
        print("Please ensure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        sys.exit(1)

def run_evaluation(config):
    """Run model evaluation"""
    try:
        from src.evaluation.model_evaluator import ModelEvaluator

        logger = setup_logger("evaluator", "INFO")
        logger.info("Starting model evaluation...")

        evaluator = ModelEvaluator(config)
        evaluator.evaluate_all()

    except ImportError as e:
        print(f"Error importing model evaluator: {e}")
        print("Please ensure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()