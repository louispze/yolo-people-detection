"""
server.py  –  Pont YOLO → Interface Smart Home
================================================
Lance YOLO sur chaque caméra (une par pièce) et diffuse les comptages
via WebSocket sur ws://localhost:8765

Dépendances :
    pip install ultralytics opencv-python websockets deep-sort-realtime

Usage :
    python server.py
"""

import asyncio
import json
import threading
import time
import cv2
import websockets
from ultralytics import YOLO

# ------------------------------------------------------------------
# CONFIG – adapte les sources caméra à ton setup
# ------------------------------------------------------------------
# Chaque entrée : { id, name, source }
#   source = 0,1,2,...  → index webcam
#   source = "rtsp://..." → flux IP
#   source = "video.mp4" → fichier (pour tests)
ROOMS = [
    {"id": "salon",    "name": "Salon",         "source": 0},  # ← seule vraie webcam
    {"id": "chambre1", "name": "Chambre 1",      "source": None},
    {"id": "chambre2", "name": "Chambre 2",      "source": None},
    {"id": "cuisine",  "name": "Cuisine",        "source": None},
    {"id": "bureau",   "name": "Bureau",         "source": None},
    {"id": "sdb",      "name": "Salle de bain",  "source": None},
]
# source = 0,1,2...  → index webcam
# source = "video.mp4" → fichier de test
# source = None      → pas de caméra, compte restera à 0

MODEL_PATH = "yolov8n.pt"   # téléchargé automatiquement si absent
CONFIDENCE = 0.4
WS_PORT    = 8765
FPS_CAP    = 5              # frames analysées / seconde par caméra

# ------------------------------------------------------------------
# État partagé (thread-safe via lock)
# ------------------------------------------------------------------
state_lock = threading.Lock()
people_state = {r["id"]: 0 for r in ROOMS}   # { room_id: count }

# ------------------------------------------------------------------
# Thread de détection par pièce
# ------------------------------------------------------------------
def detection_thread(room: dict, model: YOLO):
    """Tourne en continu pour une pièce, met à jour people_state."""
    room_id = room["id"]
    source  = room["source"]

    if source is None:
        print(f"[SKIP] {room['name']} : pas de caméra configurée.")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARN] {room['name']} : impossible d'ouvrir source={source}. Compte = 0.")
        return

    print(f"[OK] {room['name']} : caméra ouverte (source={source})")
    interval = 1.0 / FPS_CAP

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            # Fin de fichier → rebobine (utile pour tests vidéo)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, classes=[0], conf=CONFIDENCE, verbose=False)[0]
        count   = len(results.boxes)

        with state_lock:
            people_state[room_id] = count

        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))

# ------------------------------------------------------------------
# WebSocket – envoie l'état complet à chaque client connecté
# ------------------------------------------------------------------
connected_clients = set()

async def ws_handler(websocket):
    connected_clients.add(websocket)
    print(f"[WS] Client connecté ({len(connected_clients)} total)")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Client déconnecté ({len(connected_clients)} restants)")

async def broadcast_loop():
    """Diffuse l'état toutes les 500 ms à tous les clients."""
    global connected_clients
    while True:
        await asyncio.sleep(0.5)
        if not connected_clients:
            continue
        with state_lock:
            payload = json.dumps({"type": "people_update", "data": dict(people_state)})
        dead = set()
        for ws in connected_clients:
            try:
                await ws.send(payload)
            except Exception:
                dead.add(ws)
        connected_clients -= dead

# ------------------------------------------------------------------
# Lancement
# ------------------------------------------------------------------
async def main():
    print("Chargement du modèle YOLO…")
    model = YOLO(MODEL_PATH)
    print("Modèle chargé ✅")

    # Démarre un thread de détection par pièce
    for room in ROOMS:
        t = threading.Thread(target=detection_thread, args=(room, model), daemon=True)
        t.start()

    # Serveur WebSocket
    print(f"WebSocket démarré sur ws://localhost:{WS_PORT}")
    async with websockets.serve(ws_handler, "localhost", WS_PORT):
        await broadcast_loop()   # tourne indéfiniment

if __name__ == "__main__":
    asyncio.run(main())
