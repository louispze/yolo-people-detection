"""
server_app.py — Backend FastAPI de l'app P2F Smart Home (Raspberry Pi)
=====================================================================

Ce serveur expose EXACTEMENT le contrat attendu par l'app mobile
(voir app/mobile/src/lib/api.ts) :

  GET  /api/health                 -> {"ok": true}
  GET  /api/config                 -> config persistée (rooms/people/lights/climate...)
  PUT  /api/config       (body)    -> écrit backend/config.json, {"ok": true}
  GET  /api/state                  -> {"rooms": {...}, "temps": {...}}
  WS   /ws                         -> pousse toutes les 0.5 s {"type":"state",...}
  GET  /api/stream/{source}        -> flux MJPEG annoté (multipart/x-mixed-replace)
  GET  /api/snapshot/{source}      -> une image JPEG annotée

Philosophie : LE SERVEUR DÉMARRE ET SERT TOUJOURS, même sans caméra ni modèle
YOLO. Le détecteur est PLUGGABLE avec repli gracieux :

  1. On tente d'importer PeopleAnalyzer (YOLO pose) depuis le dossier parent (app/).
  2. On tente d'ouvrir la caméra réelle.
  3. Si l'un des deux échoue (ultralytics absent, modèle manquant, caméra absente),
     on bascule sur un GÉNÉRATEUR D'IMAGES DE SUBSTITUTION qui fabrique des frames
     synthétiques avec un compteur simulé, de sorte que le flux et le state ne
     sont JAMAIS vides. Le mode choisi est clairement logué.

Lancement :
    uvicorn server_app:app --host 0.0.0.0 --port 8000

Dépendances : fastapi, uvicorn[standard], opencv-python, numpy
              (ultralytics est optionnel — utilisé seulement si présent)
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# --- FastAPI / Starlette ---
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

# ---------------------------------------------------------------------------
# Chemins & logging
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../app/backend
PARENT = HERE.parent                            # .../app  (où vit analyzer.py)
CONFIG_PATH = HERE / "config.json"              # config persistée (envoyée par l'app)
CAMERAS_PATH = HERE / "cameras.json"            # mapping {roomId: source}

# On ajoute le dossier parent au path pour pouvoir importer analyzer.py
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("p2f.backend")

# ---------------------------------------------------------------------------
# Paramètres réglables (variables d'environnement)
# ---------------------------------------------------------------------------
# Modèle YOLO pose (cherché d'abord dans app/, sinon nom court résolu par ultralytics)
MODEL_PATH = os.environ.get("P2F_MODEL", "yolo11n-pose.pt")
IMG_SIZE = int(os.environ.get("P2F_IMGSZ", "640"))      # 480/320 sur Raspberry Pi
FPS_CAP = float(os.environ.get("P2F_FPS", "8"))         # images analysées / s / caméra
JPEG_QUALITY = int(os.environ.get("P2F_JPEG_QUALITY", "75"))
STREAM_FPS = float(os.environ.get("P2F_STREAM_FPS", "15"))  # cadence d'envoi MJPEG
# Forcer le mode substitution (utile pour tester sans matériel) : P2F_FAKE=1
FORCE_FAKE = os.environ.get("P2F_FAKE", "").strip() in ("1", "true", "True", "yes")

FRAME_W, FRAME_H = 640, 480     # taille des frames de substitution

# Couleurs BGR (OpenCV)
COL_BOX = (0, 200, 0)
COL_BOX_ACTIVE = (0, 165, 255)   # orange : personne active
COL_BOX_WORKOUT = (0, 0, 255)    # rouge : personne en exercice
COL_BANNER_BG = (0, 0, 0)
COL_TEXT = (255, 255, 255)
COL_ACCENT = (29, 158, 117)      # vert P2F


# ---------------------------------------------------------------------------
# Chargement PLUGGABLE du détecteur PeopleAnalyzer (repli gracieux)
# ---------------------------------------------------------------------------
def try_load_analyzer_class():
    """Essaie d'importer la classe PeopleAnalyzer.

    Retourne la classe si l'import réussit (ultralytics présent), sinon None.
    L'import seul ne charge PAS le modèle (cela se fait à l'instanciation),
    donc un import OK ne garantit pas qu'une caméra/un modèle marchera.
    """
    if FORCE_FAKE:
        log.warning("P2F_FAKE actif : détecteur réel désactivé, mode substitution forcé.")
        return None
    try:
        from analyzer import PeopleAnalyzer  # noqa: WPS433 (import local volontaire)
        log.info("PeopleAnalyzer importé depuis %s", PARENT)
        return PeopleAnalyzer
    except Exception as exc:  # ultralytics absent, torch manquant, etc.
        log.warning(
            "PeopleAnalyzer indisponible (%s). Repli sur le mode substitution.",
            exc.__class__.__name__,
        )
        return None


PeopleAnalyzerClass = try_load_analyzer_class()


# ---------------------------------------------------------------------------
# Helpers d'annotation (boîtes + bandeau compteur)
# ---------------------------------------------------------------------------
def draw_banner(frame: np.ndarray, people: int, active: int, workout: int,
                source: str, mode: str) -> None:
    """Dessine le bandeau d'information en haut de l'image (modifie frame en place)."""
    h, w = frame.shape[:2]
    bar_h = 34
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), COL_BANNER_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    text = f"Personnes: {people} | Actives: {active} | Sport: {workout}"
    cv2.putText(frame, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_TEXT, 2,
                cv2.LINE_AA)

    # Étiquette source + mode (à droite)
    tag = f"{source} [{mode}]"
    (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, tag, (max(10, w - tw - 10), 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_ACCENT, 1, cv2.LINE_AA)


def draw_persons(frame: np.ndarray, persons: List[dict]) -> None:
    """Dessine la boîte de chaque personne détectée + ses infos (modifie en place)."""
    for p in persons:
        bbox = p.get("bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]

        if p.get("is_workout"):
            color = COL_BOX_WORKOUT
        elif p.get("active"):
            color = COL_BOX_ACTIVE
        else:
            color = COL_BOX
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Libellé : ID + exercice/repetitions si pertinent
        label = f"ID {p.get('id', '?')}"
        if p.get("is_workout"):
            label += f" - {p.get('exercise', '')} x{p.get('reps', 0)}"
        elif p.get("active"):
            label += " - actif"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1, th + 6)
        cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 6, ly), color, -1)
        cv2.putText(frame, label, (x1 + 3, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def encode_jpeg(frame: np.ndarray) -> Optional[bytes]:
    """Encode une frame BGR en JPEG. Retourne les octets ou None si échec."""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return None
    return buf.tobytes()


def aggregate_exercises(persons: List[dict]) -> Dict[str, int]:
    """Compte les personnes confirmées par type d'exercice : {squat: 1, pompes: 2}."""
    counts: Dict[str, int] = {}
    for p in persons:
        if p.get("confirmed") and p.get("is_workout"):
            ex = p.get("exercise", "indetermine")
            counts[ex] = counts.get(ex, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Détecteur de SUBSTITUTION (aucun matériel requis)
# ---------------------------------------------------------------------------
class FakeDetector:
    """Génère des frames synthétiques + un comptage simulé.

    Utilisé quand il n'y a ni caméra ni modèle. Simule des "personnes" qui se
    déplacent dans l'image (marche aléatoire douce) avec, de temps en temps,
    une personne active/en exercice — pour que l'app ait toujours des données
    plausibles à afficher et que le flux soit visible.
    """

    EXOS = ["squat", "pompes", "jumping_jack"]

    def __init__(self, source: str, seed: int = 0):
        self.source = source
        self.rng = random.Random(hash((source, seed)) & 0xFFFFFFFF)
        self.t0 = time.time()
        # 0 à 3 "personnes" virtuelles, chacune avec une phase aléatoire
        n = self.rng.randint(0, 3)
        self.agents = []
        for i in range(n):
            self.agents.append({
                "id": i + 1,
                "phase_x": self.rng.uniform(0, math.tau),
                "phase_y": self.rng.uniform(0, math.tau),
                "speed": self.rng.uniform(0.2, 0.6),
                "size": self.rng.uniform(0.18, 0.30),   # fraction de la hauteur
                "active": self.rng.random() < 0.4,
                "workout": self.rng.random() < 0.2,
                "exercise": self.rng.choice(self.EXOS),
                "reps": self.rng.randint(0, 12),
            })

    def process(self, frame: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """Retourne (frame_synthetique, result) au même format que PeopleAnalyzer."""
        t = time.time() - self.t0
        img = self._background()

        persons = []
        active = 0
        workout = 0
        for a in self.agents:
            cx = (0.5 + 0.4 * math.sin(a["speed"] * t + a["phase_x"]))
            cy = (0.55 + 0.18 * math.sin(0.7 * a["speed"] * t + a["phase_y"]))
            bw = a["size"] * 0.55
            bh = a["size"]
            x1 = int((cx - bw / 2) * FRAME_W)
            y1 = int((cy - bh / 2) * FRAME_H)
            x2 = int((cx + bw / 2) * FRAME_W)
            y2 = int((cy + bh / 2) * FRAME_H)
            if a["active"]:
                active += 1
            if a["workout"]:
                workout += 1
            persons.append({
                "id": a["id"],
                "bbox": np.array([x1, y1, x2, y2], dtype=float),
                "keypoints": None,
                "confirmed": True,
                "active": a["active"],
                "activity_score": 0.1 if a["active"] else 0.0,
                "exercise": a["exercise"] if a["workout"] else "debout",
                "reps": a["reps"] if a["workout"] else 0,
                "is_workout": a["workout"],
            })

        result = {
            "count": len(self.agents),
            "active_count": active,
            "workout_count": workout,
            "persons": persons,
        }
        return img, result

    def _background(self) -> np.ndarray:
        """Fond dégradé + filigrane 'SIMULATION' pour bien indiquer le mode."""
        img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        # dégradé vertical sombre
        for y in range(FRAME_H):
            v = int(20 + 30 * (y / FRAME_H))
            img[y, :] = (v, max(0, v - 8), max(0, v - 12))
        cv2.putText(img, "SIMULATION", (FRAME_W // 2 - 150, FRAME_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 60, 60), 3, cv2.LINE_AA)
        cv2.putText(img, self.source, (FRAME_W // 2 - 40, FRAME_H // 2 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2, cv2.LINE_AA)
        return img


# ---------------------------------------------------------------------------
# Worker caméra : un thread par source
# ---------------------------------------------------------------------------
class CameraWorker:
    """Gère UNE source caméra dans son propre thread.

    - Tente d'ouvrir la caméra + d'instancier PeopleAnalyzer (mode "real").
    - À défaut, bascule sur FakeDetector (mode "fake"/substitution).
    - Conserve toujours la dernière frame annotée (pour le MJPEG/snapshot) et
      le dernier comptage (pour /api/state et /ws).

    Le mode peut basculer "real" -> "fake" à chaud si la caméra meurt en cours
    de route (après plusieurs échecs de lecture), pour ne jamais bloquer l'app.
    """

    def __init__(self, room_id: str, source: Any):
        self.room_id = room_id
        self.source = source
        self.mode = "init"                       # "real" | "fake"
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # dernière frame JPEG annotée + dernier état
        self._last_jpeg: Optional[bytes] = None
        self._last_result: dict = {
            "count": 0, "active_count": 0, "workout_count": 0, "persons": [],
        }
        self._updated_at: float = 0.0
        # condition pour réveiller les générateurs MJPEG quand une frame arrive
        self._frame_event = threading.Event()

        self.cap: Optional[cv2.VideoCapture] = None
        self.analyzer = None
        self.fake = FakeDetector(str(room_id))

        self._thread = threading.Thread(
            target=self._run, name=f"cam-{room_id}", daemon=True
        )

    # ---- cycle de vie ----
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ---- accès thread-safe ----
    def snapshot_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._last_jpeg

    def wait_jpeg(self, timeout: float = 1.0) -> Optional[bytes]:
        """Attend la prochaine frame (pour le MJPEG), renvoie la dernière connue."""
        self._frame_event.wait(timeout)
        self._frame_event.clear()
        return self.snapshot_jpeg()

    def live_state(self) -> dict:
        """Retourne l'état au format LiveRoomState attendu par l'app mobile."""
        with self._lock:
            r = self._last_result
            return {
                "people": int(r["count"]),
                "active": int(r["active_count"]),
                "workout": int(r["workout_count"]),
                "exercises": aggregate_exercises(r["persons"]),
                "updatedAt": int(self._updated_at * 1000),
            }

    # ---- internes ----
    def _publish(self, jpeg: Optional[bytes], result: dict):
        with self._lock:
            if jpeg is not None:
                self._last_jpeg = jpeg
            self._last_result = result
            self._updated_at = time.time()
        self._frame_event.set()

    def _open_real(self) -> bool:
        """Tente d'activer le mode réel (caméra + analyzer). True si OK."""
        if PeopleAnalyzerClass is None:
            return False
        # source : index int si "0", sinon chemin/url
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        try:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                log.warning("[%s] caméra source=%r non ouverte.", self.room_id, self.source)
                cap.release()
                return False
            # On instancie l'analyzer ICI (charge le modèle YOLO -> peut échouer)
            analyzer = PeopleAnalyzerClass(model_path=MODEL_PATH, imgsz=IMG_SIZE)
            self.cap = cap
            self.analyzer = analyzer
            return True
        except Exception as exc:
            log.warning("[%s] échec init mode réel (%s: %s).",
                        self.room_id, exc.__class__.__name__, exc)
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.analyzer = None
            return False

    def _run(self):
        # 1) On essaie le mode réel, sinon substitution.
        if self._open_real():
            self.mode = "real"
            log.info("[%s] MODE REEL : caméra source=%r + YOLO pose.",
                     self.room_id, self.source)
        else:
            self.mode = "fake"
            log.info("[%s] MODE SUBSTITUTION : frames synthétiques + comptage simulé.",
                     self.room_id)

        interval = 1.0 / FPS_CAP if FPS_CAP > 0 else 0.0
        read_fails = 0

        while not self._stop.is_set():
            t0 = time.time()

            if self.mode == "real":
                ok = self._tick_real()
                if not ok:
                    read_fails += 1
                    # Après trop d'échecs : on bascule définitivement en substitution
                    if read_fails > 50:
                        log.warning("[%s] caméra perdue (50 échecs) -> bascule substitution.",
                                    self.room_id)
                        self._teardown_real()
                        self.mode = "fake"
                        read_fails = 0
                    time.sleep(0.1)
                    continue
                read_fails = 0
            else:
                self._tick_fake()

            # Cadence : on limite la charge CPU
            elapsed = time.time() - t0
            sleep = interval - elapsed
            if sleep > 0:
                self._stop.wait(sleep)

        self._teardown_real()

    def _tick_real(self) -> bool:
        """Lit une frame réelle, l'analyse, l'annote et la publie. False si échec."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Fichier vidéo terminé -> rebobine ; sinon échec caméra
            if isinstance(self.source, str) and not str(self.source).startswith("rtsp"):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return False
        try:
            result = self.analyzer.process(frame)
        except Exception as exc:
            log.warning("[%s] erreur analyzer.process (%s).", self.room_id,
                        exc.__class__.__name__)
            return False

        draw_persons(frame, result["persons"])
        draw_banner(frame, result["count"], result["active_count"],
                    result["workout_count"], str(self.room_id), "live")
        self._publish(encode_jpeg(frame), result)
        return True

    def _tick_fake(self):
        """Génère une frame synthétique annotée + comptage simulé."""
        frame, result = self.fake.process()
        draw_persons(frame, result["persons"])
        draw_banner(frame, result["count"], result["active_count"],
                    result["workout_count"], str(self.room_id), "simu")
        self._publish(encode_jpeg(frame), result)

    def _teardown_real(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.analyzer = None


# ---------------------------------------------------------------------------
# Gestionnaire de caméras
# ---------------------------------------------------------------------------
class CameraManager:
    """Crée/retient un CameraWorker par source et agrège l'état global."""

    def __init__(self, cameras: Dict[str, Any]):
        self.workers: Dict[str, CameraWorker] = {}
        for room_id, source in cameras.items():
            self.workers[str(room_id)] = CameraWorker(str(room_id), source)

    def start(self):
        for w in self.workers.values():
            w.start()

    def stop(self):
        for w in self.workers.values():
            w.stop()

    def get(self, source: str) -> Optional[CameraWorker]:
        return self.workers.get(source)

    def state(self) -> dict:
        """Agrège l'état de toutes les caméras au format /api/state."""
        rooms: Dict[str, dict] = {}
        temps: Dict[str, float] = {}
        for room_id, w in self.workers.items():
            rooms[room_id] = w.live_state()
            temps[room_id] = simulated_temp(room_id, rooms[room_id])
        return {"rooms": rooms, "temps": temps}


# ---------------------------------------------------------------------------
# Température (placeholder : pas de capteur ici)
# ---------------------------------------------------------------------------
_TEMP_BASE = {}     # mémorise une base par pièce pour une dérive lente plausible


def simulated_temp(room_id: str, live: dict) -> float:
    """Renvoie une température plausible pour la pièce.

    Il n'y a pas de capteur physique branché sur ce backend : on fournit une
    valeur stable et légèrement variable (sinusoïde lente) afin que l'app ait
    un champ `temps` exploitable. À remplacer par une vraie lecture capteur
    (1-wire DS18B20, MQTT, etc.) le moment venu.
    """
    if room_id not in _TEMP_BASE:
        # base déterministe par pièce (entre 19 et 23 °C)
        _TEMP_BASE[room_id] = 19.0 + (hash(room_id) % 40) / 10.0
    base = _TEMP_BASE[room_id]
    drift = 0.4 * math.sin(time.time() / 120.0 + (hash(room_id) % 100))
    # Plus il y a de monde / d'activité, plus ça monte légèrement
    occ = 0.15 * live.get("people", 0) + 0.1 * live.get("active", 0)
    return round(base + drift + occ, 1)


# ---------------------------------------------------------------------------
# Chargement de la configuration des caméras
# ---------------------------------------------------------------------------
def load_cameras() -> Dict[str, Any]:
    """Charge le mapping {roomId: source}.

    Priorité :
      1. variable d'env P2F_CAMERAS (JSON inline, ex: '{"salon":0,"cuisine":1}')
      2. fichier backend/cameras.json
      3. défaut : {"salon": 0}
    """
    env = os.environ.get("P2F_CAMERAS")
    if env:
        try:
            data = json.loads(env)
            if isinstance(data, dict) and data:
                log.info("Caméras chargées depuis P2F_CAMERAS : %s", data)
                return data
        except json.JSONDecodeError:
            log.warning("P2F_CAMERAS n'est pas un JSON valide, ignoré.")

    if CAMERAS_PATH.exists():
        try:
            data = json.loads(CAMERAS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                log.info("Caméras chargées depuis %s : %s", CAMERAS_PATH.name, data)
                return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Lecture %s impossible (%s).", CAMERAS_PATH.name, exc)

    log.info("Aucune config caméra : défaut {'salon': 0}.")
    return {"salon": 0}


# ---------------------------------------------------------------------------
# Persistance de la config applicative (envoyée par l'app mobile)
# ---------------------------------------------------------------------------
def read_config() -> Any:
    """Lit backend/config.json. Renvoie {} si absent/illisible."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("config.json illisible (%s), renvoi {}.", exc)
        return {}


def write_config(data: Any) -> None:
    """Écrit backend/config.json de façon atomique."""
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_PATH)


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="P2F Smart Home Backend", version="1.0.0")

# CORS grand ouvert : l'app mobile (origine variable) doit pouvoir appeler le Pi.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,    # pas de cookies -> compatible avec allow_origins="*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manager global (initialisé au démarrage)
manager: Optional[CameraManager] = None


@app.on_event("startup")
def on_startup():
    global manager
    cameras = load_cameras()
    manager = CameraManager(cameras)
    manager.start()
    mode_real = "disponible" if PeopleAnalyzerClass is not None else "INDISPONIBLE"
    log.info("Backend P2F démarré. Caméras=%s | PeopleAnalyzer=%s",
             list(cameras.keys()), mode_real)


@app.on_event("shutdown")
def on_shutdown():
    if manager is not None:
        manager.stop()
    log.info("Backend P2F arrêté.")


# ---- Santé ----
@app.get("/api/health")
def health():
    return {"ok": True}


# ---- Config persistée ----
@app.get("/api/config")
def get_config():
    return JSONResponse(content=read_config())


@app.put("/api/config")
async def put_config(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "JSON invalide"})
    write_config(body)
    return {"ok": True}


# ---- State agrégé ----
@app.get("/api/state")
def get_state():
    if manager is None:
        return {"rooms": {}, "temps": {}}
    return manager.state()


# ---- WebSocket : pousse l'état toutes les 0.5 s ----
@app.websocket("/ws")
async def ws_state(websocket: WebSocket):
    import asyncio

    await websocket.accept()
    log.info("[WS] client connecté.")
    try:
        while True:
            if manager is not None:
                payload = manager.state()
            else:
                payload = {"rooms": {}, "temps": {}}
            await websocket.send_json({
                "type": "state",
                "rooms": payload["rooms"],
                "temps": payload["temps"],
            })
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        log.info("[WS] client déconnecté.")
    except Exception as exc:
        log.info("[WS] fermeture (%s).", exc.__class__.__name__)


# ---- Flux MJPEG annoté ----
def _placeholder_jpeg(text: str) -> bytes:
    """Image JPEG 'source inconnue' (quand la source demandée n'existe pas)."""
    img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    img[:] = (30, 20, 20)
    cv2.putText(img, text, (30, FRAME_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    return encode_jpeg(img) or b""


@app.get("/api/stream/{source}")
def stream(source: str):
    """Flux MJPEG (multipart/x-mixed-replace) annoté de la source demandée."""
    worker = manager.get(source) if manager is not None else None

    def gen():
        boundary = b"--frame\r\n"
        interval = 1.0 / STREAM_FPS if STREAM_FPS > 0 else 0.0
        if worker is None:
            # Source inconnue : on diffuse quand même une image fixe explicative
            jpeg = _placeholder_jpeg(f"Source '{source}' inconnue")
            while True:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.5)
        while True:
            jpeg = worker.wait_jpeg(timeout=1.0)
            if jpeg is None:
                jpeg = _placeholder_jpeg(f"{source} : initialisation...")
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            if interval:
                time.sleep(interval)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


# ---- Snapshot JPEG ----
@app.get("/api/snapshot/{source}")
def snapshot(source: str):
    """Une seule image JPEG annotée de la source demandée."""
    worker = manager.get(source) if manager is not None else None
    if worker is None:
        jpeg = _placeholder_jpeg(f"Source '{source}' inconnue")
    else:
        jpeg = worker.snapshot_jpeg() or _placeholder_jpeg(f"{source} : initialisation...")
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


# ---------------------------------------------------------------------------
# Lancement direct : python server_app.py  (équivaut à uvicorn)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("P2F_HOST", "0.0.0.0")
    port = int(os.environ.get("P2F_PORT", "8000"))
    log.info("Démarrage uvicorn sur %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
