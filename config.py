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
    "hdtf_facial":os.path.join(MODELS_DIR, "hdtf_facial"),
    "mocap_3d":   os.path.join(MODELS_DIR, "mocap_3d"),
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

# Dataset Repository Links (for reference)
DATASET_URLS = {
    "audio_talking_head": "https://www.kaggle.com/competitions/audio-driven-talking-head-generation",
    "hdtf":               "https://github.com/MRzzm/HDTF",
    "cmu_mocap":          "https://www.kaggle.com/datasets/zara2099/cmu-motion-performance-dataset",
    "amass":              "https://amass.is.tue.mpg.de/",
    "synthetic_faces":    "https://www.kaggle.com/datasets/cantonioupao/synthetic-human-faces-for-3d-reconstruction",
    "mvp_human":          "https://github.com/TingtingLiao/MVPHuman.git",
    "coco2017":           "https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset"
}

# Camera & Video Stream Settings
WEBCAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
MIRROR_WEBCAM = True

# YOLO Human Detection
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_MODEL_PATH = os.path.join(MODEL_SUBDIRS["yolo"], YOLO_MODEL_NAME)
YOLO_CONFIDENCE_THRESHOLD = 0.50
PERSON_CLASS_ID = 0

# MediaPipe Pose Settings (33 Keypoints)
MEDIAPIPE_MODEL_COMPLEXITY = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.65
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.65

# Pre-Recording Readiness & Live Animation Gate
REQUIRE_LIVE_ANIMATION = os.getenv("REQUIRE_LIVE_ANIMATION_FOR_RECORDING", "True").lower() == "true"
PRECHECK_FACE_MIN_CONFIDENCE = 0.70
PRECHECK_BODY_MIN_CONFIDENCE = 0.70
REQUIRED_FACE_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
REQUIRED_BODY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Model Fine-Tuning & QLoRA Hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10

# Color Scheme for 3D Human Avatar Renderer (BGR)
COLOR_BACKGROUND = (18, 22, 28)
COLOR_AVATAR_TORSO = (220, 150, 40)    # Shaded Metallic Blue
COLOR_AVATAR_LIMB = (240, 180, 60)     # Shaded Cyan
COLOR_JOINT_NODE = (0, 240, 255)       # Glowing Yellow
COLOR_FACE_MESH = (180, 255, 100)      # Lime Green
COLOR_TEXT_OK = (80, 220, 100)         # Emerald Green
COLOR_TEXT_WARN = (0, 165, 255)        # Orange
