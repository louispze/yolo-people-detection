"""
backend_pi.py — Backend LÉGER pour Raspberry Pi 3A+ (512 Mo)
============================================================
Webcam réelle + comptage de personnes MobileNet-SSD (OpenCV DNN, SANS torch).
Expose l'API attendue par l'app mobile P2F (cf. mobile/src/lib/api.ts) :
  GET /api/health, /api/state, /api/config (GET/PUT),
  GET /api/stream/{source} (MJPEG annoté), /api/snapshot/{source},
  WS  /ws (push de l'état toutes les 0.5 s)

Tient dans ~150-200 Mo de RAM. Le modèle se télécharge tout seul au 1er lancement.
"""
import json
import os
import threading
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
import uvicorn

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(exist_ok=True)
PROTO = MODEL_DIR / "MobileNetSSD_deploy.prototxt"
WEIGHTS = MODEL_DIR / "MobileNetSSD_deploy.caffemodel"
CONFIG_FILE = HERE / "config.json"

CAM_SOURCE = int(os.environ.get("P2F_CAM", "0"))
ROOM_ID = os.environ.get("P2F_ROOM", "salon")
CONF = float(os.environ.get("P2F_CONF", "0.45"))
PERSON_CLASS = 15  # MobileNet-SSD (VOC) : 15 = personne


def ensure_model():
    files = {
        PROTO: "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt",
        WEIGHTS: "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel",
    }
    for dst, url in files.items():
        if not dst.exists() or dst.stat().st_size == 0:
            print(f"[model] téléchargement {dst.name}…", flush=True)
            urllib.request.urlretrieve(url, dst)
    print("[model] prêt", flush=True)


class CameraWorker(threading.Thread):
    """Capture la webcam, détecte les personnes, garde la dernière image annotée."""

    def __init__(self):
        super().__init__(daemon=True)
        self.net = cv2.dnn.readNetFromCaffe(str(PROTO), str(WEIGHTS))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.lock = threading.Lock()
        self.jpeg = self._placeholder("Démarrage caméra…")
        self.count = 0
        self.active = 0
        self.updated = 0.0
        self.running = True
        self._hist = deque(maxlen=7)      # stabilisation du comptage
        self._prev_gray = None

    @staticmethod
    def _placeholder(text):
        img = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(img, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 200), 2)
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def _detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        det = self.net.forward()
        boxes = []
        for i in range(det.shape[2]):
            if int(det[0, 0, i, 1]) != PERSON_CLASS:
                continue
            if float(det[0, 0, i, 2]) < CONF:
                continue
            x1 = int(det[0, 0, i, 3] * w); y1 = int(det[0, 0, i, 4] * h)
            x2 = int(det[0, 0, i, 5] * w); y2 = int(det[0, 0, i, 6] * h)
            boxes.append((x1, y1, x2, y2))
        return boxes

    def run(self):
        cap = cv2.VideoCapture(CAM_SOURCE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            with self.lock:
                self.jpeg = self._placeholder("Webcam introuvable")
            print("[cam] impossible d'ouvrir la webcam", flush=True)
            return
        print(f"[cam] webcam ouverte (source={CAM_SOURCE})", flush=True)

        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            boxes = self._detect(frame)
            raw = len(boxes)
            self._hist.append(raw)
            # comptage stabilisé : valeur la plus fréquente sur la fenêtre
            stable = Counter(self._hist).most_common(1)[0][0]

            # mouvement global -> "actif"
            gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
            motion = 0.0
            if self._prev_gray is not None:
                motion = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
            self._prev_gray = gray
            active = stable if (motion > 6.0 and stable > 0) else 0

            # annotation
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.rectangle(frame, (0, 0), (300, 38), (0, 0, 0), -1)
            cv2.putText(frame, f"Personnes : {stable}", (8, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self.lock:
                if ok:
                    self.jpeg = buf.tobytes()
                self.count = stable
                self.active = active
                self.updated = time.time()

        cap.release()

    def snapshot(self):
        with self.lock:
            return self.jpeg

    def state(self):
        with self.lock:
            return self.count, self.active, self.updated


# ---------------------------------------------------------------------------
ensure_model()
worker = CameraWorker()
worker.start()

app = FastAPI(title="P2F Pi Backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def build_state():
    count, active, updated = worker.state()
    return {
        "type": "state",
        "rooms": {ROOM_ID: {"people": count, "active": active, "workout": 0,
                             "exercises": {}, "updatedAt": int(updated * 1000)}},
        "temps": {ROOM_ID: 21.5},
    }


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/state")
def state():
    s = build_state()
    return {"rooms": s["rooms"], "temps": s["temps"]}


@app.get("/api/config")
def get_config():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text("utf-8"))
        except Exception:
            return {}
    return {}


@app.put("/api/config")
async def put_config(payload: dict):
    try:
        CONFIG_FILE.write_text(json.dumps(payload), "utf-8")
    except Exception:
        pass
    return {"ok": True}


def mjpeg_generator():
    boundary = b"--frame"
    while True:
        frame = worker.snapshot()
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.07)  # ~14 fps de diffusion


@app.get("/api/stream/{source}")
def stream(source: str):
    return StreamingResponse(mjpeg_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/snapshot/{source}")
def snapshot(source: str):
    return Response(content=worker.snapshot(), media_type="image/jpeg")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(build_state())
            await _sleep(0.5)
    except (WebSocketDisconnect, Exception):
        return


import asyncio
async def _sleep(s):
    await asyncio.sleep(s)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
