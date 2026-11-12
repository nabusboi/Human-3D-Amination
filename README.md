# Human 3D Animation & Desktop Avatar (Digital Visualizer)

A real-time, monocular webcam-driven 3D character animation system. The application captures your face, hands, and full-body movement through a standard RGB webcam and translates your motion in real time onto a rigged 3D avatar (`rpm_fullbody.glb`) rendered natively via **Panda3D** and **OpenCV** on the desktop.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Pre-Trained Models (Included in Repository)](#pre-trained-models-included-in-repository)
- [Step-by-Step Basic Guide (Getting Started)](#step-by-step-basic-guide-getting-started)
- [Interactive Controls & Hotkeys](#interactive-controls--hotkeys)
- [Dataset Download Guide & Links](#dataset-download-guide--links)
- [Multi-Task Model Training](#multi-task-model-training)
- [Runtime Tracking & Animation Architecture](#runtime-tracking--animation-architecture)

---

## Project Structure

```text
DV/
├── .env.example                            # Template for environment variables and Kaggle API credentials
├── .gitignore                              # Git ignore rules (cache, venv, large dataset directories)
├── config.py                               # Central configuration: paths, camera, thresholds, dataset URLs
├── main.py                                 # Primary desktop runtime entry point (OpenCV + Panda3D GUI)
├── train_all.py                            # Multi-task sequential QLoRA training script
├── requirements.txt                        # Python dependencies
├── README.md                               # Project quickstart, setup guide, and documentation
├── PROJECT_DOCUMENTATION.txt               # Master project specification & slide blueprint
├── SYSTEM_ARCHITECTURE.md                  # Comprehensive architectural overview and pipeline stages
├── SYSTEM_ARCHITECTURE.txt                 # ASCII architecture schematics
│
├── models/                                 # Pre-trained models and 3D assets (bundled & tracked)
│   ├── avatars/
│   │   ├── rpm_fullbody.glb                # Skinned 3D avatar character (ReadyPlayerMe format)
│   │   ├── idle_standing.glb               # Neutral standing idle reference animation
│   │   └── README.txt                      # Avatar source attribution & asset notes
│   ├── mediapipe/
│   │   ├── face_landmarker.task            # MediaPipe 478-point 3D face mesh & blendshapes model
│   │   ├── hand_landmarker.task            # MediaPipe 21-point 3D hand & finger landmark model
│   │   └── pose_landmarker_full.task       # MediaPipe 33-point 3D full-body pose landmark model
│   ├── qlora_pose/
│   │   └── unified_multitask_qlora.pt      # Fine-tuned Multi-Task QLoRA Transformer checkpoint
│   └── yolo/
│       └── yolov8n.pt                      # Ultralytics YOLOv8 Nano person detector
│
├── src/                                    # Core application source code
│   ├── __init__.py                         # Package initializer
│   ├── pipeline.py                         # 10-stage unified motion capture & inference pipeline
│   ├── animator/
│   │   ├── avatar_3d_animator.py           # Legacy 2.5D math renderer / compatibility adapter
│   │   ├── gltf_asset.py                   # In-memory GLTF skin weight & skeleton-root corrector
│   │   └── native_avatar_renderer.py       # Panda3D 3D skeletal renderer & offscreen buffer
│   ├── dataset/
│   │   ├── cmu_mocap_dataset.py            # CMU 3D MoCap (ASF/AMC) parser & kinematic sequences
│   │   ├── coco_dataset.py                 # Microsoft COCO 2017 2D keypoint parser
│   │   ├── dataset_downloader.py           # Kaggle API automated dataset downloader
│   │   ├── hdtf_facial_dataset.py          # HDTF facial talking head dataset parser
│   │   └── synthetic_faces_dataset.py      # EG3D synthetic faces & camera matrix dataset parser
│   ├── detector/
│   │   └── yolo_detector.py                # YOLOv8 person bounding-box locator & crop extractor
│   ├── models/
│   │   └── unified_multitask_transformer.py# Transformer backbone with switchable task-specific LoRA heads
│   ├── pose/
│   │   ├── body_posture.py                 # Sitting vs. standing classifier & lower-body inference
│   │   ├── facial_expression_engine.py     # Blendshape extraction (blink, smile, mouth open, yaw/pitch/roll)
│   │   └── mediapipe_pose.py               # MediaPipe holistic tracker (pose, hands, face)
│   ├── recorder/
│   │   └── motion_recorder.py              # Synchronized MP4 video writer & JSON telemetry exporter
│   └── utils/
│       ├── camera_capture.py               # OpenCV camera device manager & aspect ratio handler
│       ├── mocap_rig.py                    # 31-joint kinematic chain definition & forward kinematics
│       └── skeleton_adapter.py             # Coordinate normalization, EMA filter & joint remapping
│
├── data/                                   # Local datasets directory (ignored in git)
│   ├── coco2017/                           # Microsoft COCO 2017 dataset
│   ├── cmu_mocap/                          # CMU 3D Motion Capture dataset
│   ├── hdtf/                               # HDTF Talking Head video dataset
│   └── synthetic_faces/                    # EG3D Synthetic Human Faces dataset
│
└── output/                                 # Session export directory (created at runtime)
    ├── *.mp4                               # Recorded side-by-side session video
    └── *.json                              # Recorded frame-by-frame 3D telemetry
```

---

## Pre-Trained Models (Included in Repository)

All required pre-trained weights and 3D assets are **already committed and pushed** to this repository. You do not need to download model weights separately to run the desktop application:

| Model / Asset | File Path | File Size | Description |
| --- | --- | --- | --- |
| **ReadyPlayerMe Avatar** | `models/avatars/rpm_fullbody.glb` | ~13.8 MB | Skinned humanoid 3D mesh with skeletal bones and blendshapes. |
| **Idle Animation Clip** | `models/avatars/idle_standing.glb` | ~168 KB | Reference standing idle clip for joints with low tracking confidence. |
| **Multi-Task QLoRA** | `models/qlora_pose/unified_multitask_qlora.pt` | ~17.2 MB | Multi-task transformer backbone fine-tuned for COCO and HDTF signals. |
| **YOLOv8 Nano** | `models/yolo/yolov8n.pt` | ~6.5 MB | Fast real-time person bounding box detector. |
| **MediaPipe Pose** | `models/mediapipe/pose_landmarker_full.task` | ~9.4 MB | 33-point 3D full-body metric landmark estimator. |
| **MediaPipe Hands** | `models/mediapipe/hand_landmarker.task` | ~7.8 MB | 21-point 3D hand/finger landmarks (dual hand tracking). |
| **MediaPipe Face** | `models/mediapipe/face_landmarker.task` | ~3.8 MB | 478-point 3D face mesh and 52 facial blendshapes. |

---

## Step-by-Step Basic Guide (Getting Started)

### Step 1: Clone the Repository

```bash
git clone https://github.com/nabusboi/Human-3D-Amination.git
cd Human-3D-Amination
```

### Step 2: Set Up a Python Virtual Environment

Python **3.10** or **3.11** is recommended.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies

Install the packages from `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Note on PyTorch:** By default, `pip install -r requirements.txt` installs the CPU/CUDA wheel specified by your pip index. If you have an NVIDIA GPU and want CUDA acceleration, install PyTorch matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

### Step 4: Configure Environment Variables (Optional)

Create a `.env` file in the root directory if you want to customize settings or use Kaggle API downloads:

```ini
# Camera Index (default: 0)
WEBCAM_INDEX=0

# Hardware Acceleration: "cuda" or "cpu"
AVATAR_DEVICE=cuda

# Kaggle API Credentials (required only if downloading datasets via script)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

### Step 5: Launch the Application

Run the application:

```powershell
python main.py
```

To force CPU inference on machines without an NVIDIA GPU:

```powershell
$env:AVATAR_DEVICE = "cpu"
python main.py
```

*When started, look for `[Avatar] Native GLB active` and `[Checkpoint] Successfully loaded unified_multitask_qlora.pt` in the console.*

---

## Interactive Controls & Hotkeys

While the desktop window is active, use the following hotkeys:

| Key | Function | Description |
| :---: | :---: | --- |
| **V** | **Cycle Framing** | Switches between **Auto Body Framing**, **Portrait**, and **Full Body** view. |
| **C** | **Recalibrate** | Recalibrates the neutral head pose to your current position. Face the camera when pressing this. |
| **Space** | **Readiness Gate** | Validates tracking confidence and posture readiness before recording. |
| **R** | **Record / Stop** | Toggles synchronized recording of MP4 video and frame-by-frame JSON telemetry into `./output/`. |
| **Q** or **Esc** | **Quit** | Safely shuts down the camera, closes Panda3D buffers, and releases system resources. |

---

## Dataset Download Guide & Links

If you want to train or fine-tune the multi-task transformer using `train_all.py`, download the datasets below and place them in the corresponding `./data/` subdirectories.

> **Dataset Fallback Note:** If a dataset directory is empty or not downloaded, the dataset loader automatically generates synthetic training samples so that pipeline tests and code validation can run without error.

### 1. Microsoft COCO 2017 Dataset (Person Keypoints)
- **Used for:** 2D human pose refinement (17 keypoints).
- **Download Links:**
  - Kaggle: [awsaf49/coco-2017-dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
  - Official COCO: [cocodataset.org/#download](https://cocodataset.org/#download)
- **Target Folder:** `data/coco2017/coco2017/`
- **Expected Subfolders:**
  ```text
  data/coco2017/coco2017/
  ├── annotations/
  │   └── person_keypoints_val2017.json
  ├── train2017/
  └── val2017/
  ```
- **Download via Kaggle CLI:**
  ```powershell
  kaggle datasets download -d awsaf49/coco-2017-dataset -p data/coco2017 --unzip
  ```

### 2. CMU Motion Capture (MoCap) Dataset
- **Used for:** 3D motion forecasting and joint kinematics (31 3D joints in ASF/AMC format).
- **Download Links:**
  - Kaggle: [zara2099/cmu-motion-performance-dataset](https://www.kaggle.com/datasets/zara2099/cmu-motion-performance-dataset)
  - Official Site: [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu/)
- **Target Folder:** `data/cmu_mocap/`
- **Expected Contents:** Unzipped `.asf` skeleton files and `.amc` motion recording files.
- **Download via Kaggle CLI:**
  ```powershell
  kaggle datasets download -d zara2099/cmu-motion-performance-dataset -p data/cmu_mocap --unzip
  ```

### 3. HDTF (High-Definition Talking Face) Dataset
- **Used for:** Facial dynamics, mouth opening, lip stretch, eye blink, and head roll parameters.
- **Download Links:**
  - GitHub Repository: [MRzzm/HDTF](https://github.com/MRzzm/HDTF)
  - Kaggle Competition Reference: [Audio-Driven Talking Head Generation](https://www.kaggle.com/competitions/audio-driven-talking-head-generation)
- **Target Folder:** `data/hdtf/HDTF_dataset/`
- **Expected Contents:** Video frames and facial parameter files (`.json` or `.npy`).

### 4. Synthetic Human Faces for 3D Reconstruction (EG3D)
- **Used for:** Camera pose regression and 3D face orientation matrices.
- **Download Links:**
  - Kaggle: [cantonioupao/synthetic-human-faces-for-3d-reconstruction](https://www.kaggle.com/datasets/cantonioupao/synthetic-human-faces-for-3d-reconstruction)
- **Target Folder:** `data/synthetic_faces/eg3d_synthetic_faces/eg3d_synthetic_faces/`
- **Expected Contents:** PNG face images (`seed0000.png` ...) paired with camera pose files (`seed0000.npy` ...).
- **Download via Kaggle CLI:**
  ```powershell
  kaggle datasets download -d cantonioupao/synthetic-human-faces-for-3d-reconstruction -p data/synthetic_faces --unzip
  ```

### 5. Additional Extended Datasets (Optional)
- **AMASS (Archive of Motion Capture as Surface Shapes):** [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) — Placed in `data/amass/`
- **MVP-Human:** [github.com/TingtingLiao/MVPHuman](https://github.com/TingtingLiao/MVPHuman.git) — Placed in `data/mvp_human/`

### Automated Dataset Verification & Download Script

You can verify and automatically download datasets using `src/dataset/dataset_downloader.py`:

```powershell
python src/dataset/dataset_downloader.py
```

*(Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` configured in `.env` or system environment).*

---

## Multi-Task Model Training

The project includes a single multi-task transformer backbone trained across four heterogeneous tasks using Parameter-Efficient Fine-Tuning (QLoRA):

1. **`coco`**: 2D human pose refinement (Wing Loss on facial landmarks + Smooth L1 on body joints).
2. **`mocap`**: 3D motion forecasting (MSE Loss on 31-joint displacement vectors).
3. **`hdtf`**: Talking head facial coefficients (Smooth L1 Loss).
4. **`faces`**: 3D camera pose matrix regression (MSE Loss).

To train or fine-tune the model:

```powershell
# Default training run
python train_all.py

# Custom epochs, batch size, and learning rate
python train_all.py --epochs 15 --batch_size 32 --lr 0.0001
```

During training, the transformer backbone weights are frozen while only the task-specific LoRA adapters (`<5%` of parameters) and projection heads are trained. The final weights are saved into `models/qlora_pose/unified_multitask_qlora.pt`.

---

## Runtime Tracking & Animation Architecture

### Camera & Auto-Framing
- The application opens the camera using OpenCV and streams frames into an offscreen Panda3D graphics buffer.
- `AVATAR_VIEW` in `config.py` sets the default framing. When `AVATAR_AUTO_FRAME = True`, camera framing dynamically scales based on face bounding box size and automatically widens as hips, knees, or feet enter the camera view. Pressing **V** cycles between auto framing, portrait framing, and full-body view.

### Hand & Arm IK Tracking
- Wrist targets are matched directly to MediaPipe 3D body wrist landmarks.
- Individual finger joint rotations and palm orientation are resolved using MediaPipe's 3D hand landmarker.
- **Dropout Hold:** Brief hand tracking dropouts (up to 0.35s) are smoothed and held.
- **Palms Overlapping Resolution:** When hands come together or cross, ambiguous finger updates are held at the last reliable position to avoid flipping. Tracking automatically resumes once the palms separate.
- **Torso Clearance:** Wrist positions are constrained by a torso clearance plane before solving inverse kinematics, preventing limbs from clipping through the avatar's body.

### Facial Blendshapes & Head Pose
- Head orientation (yaw, pitch, roll) is estimated from facial landmarks and stabilized using Exponential Moving Average (EMA) filtering.
- 52 blendshape coefficients drive facial expressions (mouth open, smile, eye blinks, eyebrow raises).
- Press **C** at any time while facing the camera to recalibrate the neutral head orientation.

### Biomechanical Posture Estimation
- Sitting vs. standing posture is inferred when both legs are visible.
- If legs leave the camera frame, the last known reliable posture is held, and upper-body tracking continues seamlessly.
