import platform
from typing import Optional

import cv2


def open_camera(preferred_index: int = 0, index_count: int = 4) -> Optional[cv2.VideoCapture]:
    """Try common platform backends and nearby indices, returning the first usable camera."""
    if platform.system() == "Windows":
        backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
    else:
        backends = (cv2.CAP_ANY,)

    for index in range(max(0, preferred_index), max(0, preferred_index) + max(1, index_count)):
        for backend in backends:
            capture = cv2.VideoCapture()
            try:
                if not capture.open(index, backend):
                    capture.release()
                    continue
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, frame = capture.read()
                if ok and frame is not None and frame.size:
                    backend_name = capture.getBackendName()
                    print(f"[Camera] Opened index {index} with {backend_name} ({frame.shape[1]}x{frame.shape[0]}).")
                    return capture
            except Exception as exc:
                print(f"[Camera] Index {index}, backend {backend}: {exc}")
            capture.release()
    return None
