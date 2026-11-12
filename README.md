# Desktop avatar

Run from this project folder:

```powershell
python main.py
```

The Python application opens the webcam with OpenCV. Panda3D renders the local
`models/avatars/rpm_fullbody.glb` character into the same desktop preview.
No browser, HTML page, or server is used.

Install dependencies in the Python environment used to run the application:

```powershell
python -m pip install -r requirements.txt
```

The first face detected sets the neutral head position. Face the camera when
starting, or press **C** to recalibrate.

| Key | Action |
| --- | --- |
| V | Cycle automatic body / portrait / full body framing |
| C | Recalibrate neutral head position |
| Space | Check recording readiness |
| R | Start / stop recording |
| Q or Esc | Close the application |

`AVATAR_VIEW` in `config.py` sets the initial framing. Automatic view follows
your face size and widens as hips, knees, or feet enter the camera image. Set
`AVATAR_AUTO_FRAME = False` to keep a fixed crop.

Hands are matched to the body wrists. Palm rotation and individual finger
segments use MediaPipe's 3D hand landmarks, with a brief hold during detection
dropouts (up to 0.35 seconds). Fresh hand detection is retried periodically when
fewer than two palms are tracked. Metric body landmarks drive the arms, with
a fixed-length wrist positioning solver when hands are detected, including
when elbows are cropped. Forward reach is bounded: a monocular webcam does
not measure absolute hand distance. A completely hidden second palm cannot
supply fresh finger motion. Camera input and the desktop preview preserve
aspect ratio when resized.

When palms overlap, ambiguous hand assignments and finger depth updates are
held at the last independently observed pair. Tracking resumes after sustained
separation. Brief detection gaps are held for 0.35 seconds; start with both
hands visible separately so a reliable pair is available before joining them.
During contact, palm orientation is held in avatar space to avoid wrist flips.
Wrist targets use a conservative torso clearance plane and smooth before arm
solving. This is an approximate clearance constraint, not mesh collision detection.

Hand landmarks have a palm-size-relative deadband and adaptive smoothing.
Sudden large changes require 0.15 seconds of consistent observations; depth-only
changes are held while image landmarks remain still. Wrist targets and elbow
bend directions are stabilized separately. After a tracking dropout, fingers
return to a fixed neutral pose instead of playing the idle finger animation.
Very small intentional gestures may need to exceed the deadband to register.

The existing `unified_multitask_qlora.pt` checkpoint remains in the live pipeline:
its COCO branch assists body joints, and its HDTF branch contributes mouth,
lip, and blink metrics. MediaPipe supplies body and finger landmarks, head pose,
and facial blendshapes. These signals drive the GLB skeleton and facial targets.
The checkpoint predicts motion; the GLB supplies the character's appearance.

Joints with low tracking confidence use the standing animation. Full body
tracking requires the corresponding limbs to be visible. Sitting and standing
are inferred from both visible legs; the last reliable posture is held when the
legs leave the image. The checkpoint is
loaded as-is; this renderer change does not retrain it.

For CPU inference:

```powershell
$env:AVATAR_DEVICE = "cpu"
python main.py
```

Native 3D rendering still uses the graphics driver even when model inference
uses the CPU. Check the terminal for `[Avatar] Native GLB active` and the
checkpoint's successful-load message.
