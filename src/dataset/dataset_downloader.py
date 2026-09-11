import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import subprocess
import urllib.request
from typing import Optional, Dict
import config


class DatasetDownloader:
    """
    Automated Dataset & Model Asset Downloader.
    Downloads datasets to `./data/<dataset_name>` and stores cached models in `./models/<model_type>` to prevent redundant runs.
    """

    @classmethod
    def check_and_download_all(cls):
        """Checks and downloads dataset repositories if credentials/links are available."""
        print("[Dataset Downloader] Checking dataset & model directories...")

        # 1. Download/Verify YOLO Model Weights
        cls.ensure_yolo_weights()

        # 2. Check Kaggle Datasets
        kaggle_user = os.getenv("KAGGLE_USERNAME")
        kaggle_key = os.getenv("KAGGLE_KEY")

        if kaggle_user and kaggle_key and kaggle_user != "your_kaggle_username":
            print(f"[Dataset Downloader] Kaggle API credentials found for '{kaggle_user}'. Setting environment variables...")
            os.environ["KAGGLE_USERNAME"] = kaggle_user
            os.environ["KAGGLE_KEY"] = kaggle_key
            print("[Dataset Downloader] Ready for dataset downloads into `./data/`.")
        else:
            print("[Dataset Downloader Note] Kaggle API credentials not set in .env.")
            print("  Datasets can be manually placed into `./data/` subdirectories.")

    @classmethod
    def ensure_yolo_weights(cls) -> str:
        """Ensures YOLO model weights are cached in `./models/yolo/yolov8n.pt`."""
        yolo_path = config.YOLO_MODEL_PATH
        if not os.path.exists(yolo_path):
            print(f"[Model Manager] Downloading YOLO weights to cached directory: {yolo_path}")
            try:
                from ultralytics import YOLO
                model = YOLO("yolov8n.pt")
                if os.path.exists("yolov8n.pt"):
                    os.rename("yolov8n.pt", yolo_path)
            except Exception as e:
                print(f"[Model Manager Warning] Automated YOLO download failed: {e}")
        else:
            print(f"[Model Manager] Cached YOLO model found at: {yolo_path}")
        return yolo_path

    @classmethod
    def download_kaggle_dataset(cls, dataset_name: str, target_dir: str):
        """Downloads and unzips Kaggle dataset into target_dir."""
        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
            print(f"[Dataset Manager] Dataset '{dataset_name}' already exists in: {target_dir}")
            return

        try:
            print(f"[Dataset Manager] Downloading Kaggle dataset '{dataset_name}' to: {target_dir}")
            cmd = f"kaggle datasets download -d {dataset_name} -p {target_dir} --unzip"
            subprocess.run(cmd, shell=True, check=True)
            print(f"[Dataset Manager Success] Downloaded {dataset_name} to {target_dir}")
        except Exception as e:
            print(f"[Dataset Manager Warning] Failed to download {dataset_name}: {e}")

    @classmethod
    def get_trained_model_dir(cls, model_type: str = "qlora_pose") -> str:
        """Returns target directory for saving/loading trained model checkpoints."""
        target_dir = config.MODEL_SUBDIRS.get(model_type, config.MODELS_DIR)
        os.makedirs(target_dir, exist_ok=True)
        return target_dir


if __name__ == "__main__":
    DatasetDownloader.check_and_download_all()
