"""
analyzer.py — Cœur de détection : comptage stable + activité physique
=====================================================================

UN SEUL modèle (YOLO pose) en mode tracking fait tout :
  - détection des personnes        → boîtes + IDs persistants  (comptage)
  - 17 keypoints du squelette       → mouvement                (activité)

Pourquoi c'est mieux que l'ancien `len(results.boxes)` :
  1. On TRACKE (ByteTrack) au lieu de re-compter chaque image → l'ID survit
     quand la personne bouge, lève la main ou est brièvement occultée.
  2. DÉBOUNCE : une personne n'est comptée qu'après `min_hits` images, et
     reste comptée jusqu'à `max_age` images sans détection → plus de
     clignotement du compteur quand ça bouge.
  3. ACTIVITÉ : on mesure le déplacement des keypoints d'une image à l'autre,
     normalisé par la taille du torse (donc indépendant de la distance à la
     caméra), lissé sur une fenêtre → "actif" / "immobile" stable.

Usage :
    from analyzer import PeopleAnalyzer
    analyzer = PeopleAnalyzer()
    result = analyzer.process(frame)
    # result = {
    #   "count": 3,                # nb de personnes (stable)
    #   "active_count": 1,         # parmi elles, combien sont actives
    #   "persons": [ {id, bbox, keypoints, confirmed, active, activity_score}, ... ]
    # }
"""

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from exercises import ExerciseTracker

# Indices des keypoints COCO utilisés pour la normalisation (taille du torse)
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

_HERE = Path(__file__).resolve().parent


class PeopleAnalyzer:
    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        tracker: str = "bytetrack_stable.yaml",
        conf: float = 0.25,        # seuil bas : on récupère les corps partiels en intérieur
        iou: float = 0.7,
        imgsz: int = 640,          # baisser à 480/320 sur Raspberry Pi
        # --- stabilisation du comptage ---
        min_hits: int = 3,         # images consécutives avant de compter une personne
        max_age: int = 30,         # images sans détection avant de la décompter
        # --- détection d'activité ---
        activity_window: int = 12,    # fenêtre (images) d'historique des keypoints
        activity_threshold: float = 0.06,  # mouvement normalisé au-dessus duquel = actif
        activity_smooth: int = 8,     # lissage du score d'activité (anti-clignotement)
        kpt_conf: float = 0.5,        # confiance mini d'un keypoint pour être utilisé
        detect_exercises: bool = True,  # reconnaître squat/pompes/jumping jack + compter les reps
    ):
        # tracker : chemin absolu pour fonctionner quel que soit le cwd
        tracker_path = _HERE / tracker
        self.tracker = str(tracker_path) if tracker_path.exists() else tracker

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.min_hits = min_hits
        self.max_age = max_age
        self.activity_window = activity_window
        self.activity_threshold = activity_threshold
        self.activity_smooth = activity_smooth
        self.kpt_conf = kpt_conf

        # État par ID de track
        self.hits = defaultdict(int)       # images consécutives vues
        self.misses = defaultdict(int)     # images consécutives manquées
        self.confirmed = set()             # IDs actuellement comptés
        self.kpt_hist = defaultdict(lambda: deque(maxlen=activity_window))
        self.score_hist = defaultdict(lambda: deque(maxlen=activity_smooth))

        self.detect_exercises = detect_exercises
        self.exercises = ExerciseTracker() if detect_exercises else None

    # ------------------------------------------------------------------
    @staticmethod
    def _torso_len(kxy: np.ndarray):
        """Distance épaules→hanches, sert d'échelle (indépendance à la distance)."""
        sh = (kxy[L_SHOULDER] + kxy[R_SHOULDER]) / 2.0
        hp = (kxy[L_HIP] + kxy[R_HIP]) / 2.0
        d = float(np.linalg.norm(sh - hp))
        return d if d > 1.0 else None

    def _activity(self, tid, kxy: np.ndarray, kconf: np.ndarray):
        """Score de mouvement normalisé + décision actif/immobile (lissée)."""
        self.kpt_hist[tid].append((kxy.copy(), kconf.copy()))
        if len(self.kpt_hist[tid]) < 2:
            return False, 0.0

        cur_xy, cur_c = self.kpt_hist[tid][-1]
        prev_xy, prev_c = self.kpt_hist[tid][-2]

        torso = self._torso_len(cur_xy)
        valid = (cur_c > self.kpt_conf) & (prev_c > self.kpt_conf)

        if torso is None or not valid.any():
            score = 0.0
        else:
            disp = np.linalg.norm(cur_xy - prev_xy, axis=1)[valid]
            score = float(disp.mean() / torso)

        self.score_hist[tid].append(score)
        smooth = float(np.mean(self.score_hist[tid]))
        return smooth > self.activity_threshold, smooth

    def _forget(self, tid):
        self.confirmed.discard(tid)
        self.hits.pop(tid, None)
        self.misses.pop(tid, None)
        self.kpt_hist.pop(tid, None)
        self.score_hist.pop(tid, None)
        if self.exercises is not None:
            self.exercises.forget(tid)

    # ------------------------------------------------------------------
    def process(self, frame) -> dict:
        r = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            classes=[0],            # 0 = personne (COCO)
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        persons = []
        ids_this_frame = set()

        has_ids = r.boxes is not None and r.boxes.id is not None
        if has_ids:
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().numpy()
            kdata = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None

            for i, tid in enumerate(ids):
                ids_this_frame.add(tid)
                self.hits[tid] += 1
                self.misses[tid] = 0
                if self.hits[tid] >= self.min_hits:
                    self.confirmed.add(tid)

                active, score, kxy = False, 0.0, None
                exercise, reps, is_workout = "indetermine", 0, False
                if kdata is not None:
                    kxy = kdata[i][:, :2]
                    kconf = kdata[i][:, 2]
                    active, score = self._activity(tid, kxy, kconf)
                    if self.exercises is not None:
                        ex = self.exercises.update(tid, kxy, kconf)
                        exercise, reps, is_workout = ex["exercise"], ex["reps"], ex["is_workout"]

                persons.append({
                    "id": tid,
                    "bbox": boxes[i],
                    "keypoints": kxy,
                    "confirmed": tid in self.confirmed,
                    "active": active,
                    "activity_score": score,
                    "exercise": exercise,
                    "reps": reps,
                    "is_workout": is_workout,
                })

        # Vieillissement des IDs confirmés non revus cette image (débounce de sortie)
        for tid in list(self.confirmed):
            if tid not in ids_this_frame:
                self.misses[tid] += 1
                if self.misses[tid] > self.max_age:
                    self._forget(tid)

        # Nettoyage des IDs jamais confirmés et disparus (évite la fuite mémoire)
        for tid in list(self.hits.keys()):
            if tid not in self.confirmed and tid not in ids_this_frame:
                self.misses[tid] += 1
                if self.misses[tid] > self.max_age:
                    self._forget(tid)

        count = len(self.confirmed)
        active_count = sum(1 for p in persons if p["confirmed"] and p["active"])
        workout_count = sum(1 for p in persons if p["confirmed"] and p["is_workout"])

        return {
            "count": count,
            "active_count": active_count,
            "workout_count": workout_count,
            "persons": persons,
        }
