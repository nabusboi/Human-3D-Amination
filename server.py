import os
import cv2
import json
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
import config
from src.pipeline import MovementDetectorPipeline

app = FastAPI(
    title="Human-to-Animation Movement Detector API",
    description="0-Cent Cost Backend API providing live pose estimation, stickman animation streaming, and motion recording endpoints.",
    version="1.0.0"
)

pipeline = MovementDetectorPipeline()


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Human-to-Animation Movement Detector API",
        "endpoints": {
            "webcam_status": "/api/status",
            "websocket_stream": "/ws/stream"
        }
    }


@app.get("/api/status")
def get_status():
    return {
        "camera_index": config.WEBCAM_INDEX,
        "resolution": [config.FRAME_WIDTH, config.FRAME_HEIGHT],
        "fps": config.FPS,
        "yolo_model": config.YOLO_MODEL_NAME,
        "mediapipe_landmarks": 33,
        "current_state": pipeline.recorder.state.value
    }


@app.websocket("/ws/stream")
async def websocket_motion_stream(websocket: WebSocket):
    """
    WebSocket endpoint streaming 33 keypoints and telemetry frames at 30 FPS.
    """
    await websocket.accept()
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.03)
                continue

            _, _, telemetry = pipeline.process_frame(frame)
            await websocket.send_json(telemetry)
            await asyncio.sleep(1.0 / config.FPS)
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    finally:
        cap.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
