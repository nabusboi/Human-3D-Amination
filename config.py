import os
from dotenv import load_dotenv

# Load Environment Variables from .env
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data & Output Directories
DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(BASE_DIR, "data"))
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(BASE_DIR, "models"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "output"))

# Model Storage Subdirectories
MODEL_SUBDIRS = {
    "yolo":       os.path.join(MODELS_DIR, "yolo"),
    "mediapipe":  os.path.join(MODELS_DIR, "mediapipe"),
    "qlora_pose": os.path.join(MODELS_DIR, "qlora_pose"),
}

# Dataset Subdirectories — corrected to actual nested paths on disk
DATASET_SUBDIRS = {
    # data/coco2017/coco2017/{annotations, train2017, val2017, test2017}
    "coco2017":        os.path.join(DATASET_DIR, "coco2017", "coco2017"),
    # data/hdtf/HDTF_dataset/
    "hdtf":            os.path.join(DATASET_DIR, "hdtf", "HDTF_dataset"),
    # data/cmu_mocap/ (root contains allc3d_*, allasfamc etc.)
    "cmu_mocap":       os.path.join(DATASET_DIR, "cmu_mocap"),
    # amass not yet downloaded
    "amass":           os.path.join(DATASET_DIR, "amass"),
    # data/synthetic_faces/eg3d_synthetic_faces/eg3d_synthetic_faces
    "synthetic_faces": os.path.join(DATASET_DIR, "synthetic_faces", "eg3d_synthetic_faces", "eg3d_synthetic_faces"),
    # data/mvp_human/ (root of the cloned MVPHuman repo)
    "mvp_human":       os.path.join(DATASET_DIR, "mvp_human"),
}

# Create output & model dirs automatically (do NOT auto-create dataset dirs — they already exist)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
for path in MODEL_SUBDIRS.values():
    os.makedirs(path, exist_ok=True)

# Alias used by coco_dataset.py as the default COCO 2017 data directory
KAGGLE_DATASET_DIR = DATASET_SUBDIRS["coco2017"]

# Dataset Repository & Download Links
DATASET_URLS = {
    "coco2017":           "https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset",
    "cmu_mocap":          "https://www.kaggle.com/datasets/zara2099/cmu-motion-performance-dataset",
    "hdtf":               "https://github.com/MRzzm/HDTF",
    "synthetic_faces":    "https://www.kaggle.com/datasets/cantonioupao/synthetic-human-faces-for-3d-reconstruction",
    "audio_talking_head": "https://www.kaggle.com/competitions/audio-driven-talking-head-generation",
    "amass":              "https://amass.is.tue.mpg.de/",
    "mvp_human":          "https://github.com/TingtingLiao/MVPHuman.git",
}

# Camera & Video Stream Settings
WEBCAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
MIRROR_WEBCAM = True

# Native desktop avatar (GLB + skeletal animation).
AVATAR_MODEL_PATH = os.path.join(MODELS_DIR, "avatars", "rpm_fullbody.glb")
AVATAR_IDLE_PATH = os.path.join(MODELS_DIR, "avatars", "idle_standing.glb")
AVATAR_VIEW = "auto"  # V cycles automatic body framing, portrait, full body.
AVATAR_AUTO_FRAME = True  # Follow face size and visible hips/knees/feet.

# YOLO Human Detection
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_MODEL_PATH = os.path.join(MODEL_SUBDIRS["yolo"], YOLO_MODEL_NAME)
YOLO_CONFIDENCE_THRESHOLD = 0.50
PERSON_CLASS_ID = 0

# MediaPipe Pose Settings (33 Keypoints)
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.65
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.65

# Pre-recording readiness.
REQUIRED_FACE_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Model Fine-Tuning & QLoRA Hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10

# OpenCV status colors (BGR).
COLOR_TEXT_OK = (80, 220, 100)         # Emerald Green
COLOR_TEXT_WARN = (0, 165, 255)        # Orange
