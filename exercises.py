"""
exercises.py — Reconnaissance d'exercices + comptage de répétitions
===================================================================

Approche (la même que la solution `AIGym` d'Ultralytics, généralisée) :
  1. À partir des 17 keypoints COCO, on calcule des ANGLES ARTICULAIRES
     (coude, genou, hanche) et l'ORIENTATION DU TORSE (debout vs au sol).
  2. On en déduit l'EXERCICE par règles géométriques, lissé par vote majoritaire
     sur une fenêtre (anti-clignotement).
  3. On COMPTE LES RÉPÉTITIONS avec une machine d'état "haut/bas" sur l'angle
     pertinent de l'exercice (ex. coude pour les pompes, genou pour les squats).

Exercices reconnus : squat, pompes, jumping_jack, (debout / indetermine sinon).

⚠️ Honnêteté technique : la classification par règles marche bien pour un petit
nombre d'exercices BIEN distincts vus de profil/face. Pour beaucoup d'exercices
ou une grande robustesse, il faut entraîner un petit classifieur sur séquences de
keypoints (voir note en bas de ce fichier). Ici on couvre tes exemples (pompes,
squats) + jumping jacks, avec comptage de reps.
"""

import math
from collections import defaultdict, deque

import numpy as np

# Indices keypoints COCO
NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

KPT_CONF = 0.5   # confiance mini pour utiliser un keypoint


def _angle(a, b, c) -> float:
    """Angle (deg) au point b formé par a-b-c. Formule identique à Ultralytics AIGym."""
    rad = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    deg = abs(rad * 180.0 / math.pi)
    return deg if deg <= 180.0 else 360.0 - deg


def _pt(kxy, kconf, i):
    """Renvoie (x,y) si le keypoint est assez fiable, sinon None."""
    return kxy[i] if kconf[i] > KPT_CONF else None


def _joint_angle(kxy, kconf, a, b, c):
    """Angle articulaire moyen-robuste : None si un des 3 points manque."""
    pa, pb, pc = _pt(kxy, kconf, a), _pt(kxy, kconf, b), _pt(kxy, kconf, c)
    if pa is None or pb is None or pc is None:
        return None
    return _angle(pa, pb, pc)


def _mean_angle(*vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Définition des exercices : keypoint d'angle "compteur" + seuils haut/bas
# ---------------------------------------------------------------------------
EXERCISES = {
    "pompes":       {"down": 95.0,  "up": 150.0},  # angle coude
    "squat":        {"down": 100.0, "up": 165.0},  # angle genou
    "jumping_jack": {"down": None,  "up": None},    # compté différemment (bras haut/bas)
}


class _PersonExercise:
    """État de classification + comptage pour UNE personne (un track ID)."""

    def __init__(self, vote_window=12):
        self.votes = deque(maxlen=vote_window)
        self.exercise = "indetermine"
        self.stage = None       # "up" / "down" pour la machine d'état des reps
        self.reps = 0
        self.jj_arms_up = False  # état précédent des bras (jumping jack)

    # ---- géométrie d'une image -> exercice candidat ----
    def _classify_frame(self, kxy, kconf, angles):
        elbow, knee, hip = angles["elbow"], angles["knee"], angles["hip"]

        # Orientation du torse : 0° = debout (vertical), 90° = au sol (horizontal)
        sh = _midpoint(kxy, kconf, L_SH, R_SH)
        hp = _midpoint(kxy, kconf, L_HIP, R_HIP)
        torso_horizontal = False
        upright = False
        if sh is not None and hp is not None:
            dx, dy = hp[0] - sh[0], hp[1] - sh[1]
            from_vertical = abs(math.degrees(math.atan2(dx, dy)))  # 0 vertical, 90 horizontal
            torso_horizontal = from_vertical > 55.0
            upright = from_vertical < 40.0

        # Poignets au-dessus de la tête ? (jumping jack bras levés)
        wrists_up = _wrists_above_head(kxy, kconf)

        # --- Règles ---
        if torso_horizontal and elbow is not None:
            # corps gainé à l'horizontale + flexion possible des coudes = pompes
            return "pompes"
        if upright:
            if wrists_up:
                return "jumping_jack"
            if knee is not None and knee < 150.0:
                # genoux fléchis en position debout = squat (descente)
                return "squat"
            return "debout"
        return "indetermine"

    # ---- comptage de répétitions selon l'exercice courant ----
    def _count_reps(self, kxy, kconf, angles):
        ex = self.exercise

        if ex == "pompes" and angles["elbow"] is not None:
            self._angle_rep(angles["elbow"], EXERCISES["pompes"])
        elif ex == "squat" and angles["knee"] is not None:
            self._angle_rep(angles["knee"], EXERCISES["squat"])
        elif ex == "jumping_jack":
            self._jumping_jack_rep(kxy, kconf)
        else:
            self.stage = None  # exercice non compté → on réinitialise la phase

    def _angle_rep(self, ang, thr):
        """Machine d'état : descend sous 'down' puis remonte au-dessus de 'up' = 1 rep."""
        if ang < thr["down"]:
            self.stage = "down"
        elif ang > thr["up"]:
            if self.stage == "down":
                self.reps += 1
            self.stage = "up"

    def _jumping_jack_rep(self, kxy, kconf):
        up = _wrists_above_head(kxy, kconf)
        if up and not self.jj_arms_up:   # transition bas -> haut = 1 rep
            self.reps += 1
        self.jj_arms_up = up

    def update(self, kxy, kconf):
        angles = {
            "elbow": _mean_angle(
                _joint_angle(kxy, kconf, L_SH, L_EL, L_WR),
                _joint_angle(kxy, kconf, R_SH, R_EL, R_WR),
            ),
            "knee": _mean_angle(
                _joint_angle(kxy, kconf, L_HIP, L_KNEE, L_ANK),
                _joint_angle(kxy, kconf, R_HIP, R_KNEE, R_ANK),
            ),
            "hip": _mean_angle(
                _joint_angle(kxy, kconf, L_SH, L_HIP, L_KNEE),
                _joint_angle(kxy, kconf, R_SH, R_HIP, R_KNEE),
            ),
        }

        candidate = self._classify_frame(kxy, kconf, angles)
        self.votes.append(candidate)
        # Exercice = vote majoritaire sur la fenêtre (stable)
        new_ex = max(set(self.votes), key=self.votes.count)
        if new_ex != self.exercise:
            self.exercise = new_ex
            self.stage = None  # on repart proprement quand l'exercice change

        self._count_reps(kxy, kconf, angles)
        return {"exercise": self.exercise, "reps": self.reps,
                "is_workout": self.exercise in EXERCISES}


def _midpoint(kxy, kconf, i, j):
    a, b = _pt(kxy, kconf, i), _pt(kxy, kconf, j)
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _wrists_above_head(kxy, kconf):
    """True si au moins un poignet est au-dessus du nez (y plus petit = plus haut)."""
    nose = _pt(kxy, kconf, NOSE)
    if nose is None:
        # fallback : au-dessus de la ligne des épaules
        ref = _midpoint(kxy, kconf, L_SH, R_SH)
        if ref is None:
            return False
        ref_y = ref[1]
    else:
        ref_y = nose[1]
    for w in (L_WR, R_WR):
        p = _pt(kxy, kconf, w)
        if p is not None and p[1] < ref_y:
            return True
    return False


class ExerciseTracker:
    """Gère l'état d'exercice de toutes les personnes (clé = track ID)."""

    def __init__(self, vote_window=12):
        self.vote_window = vote_window
        self.people = defaultdict(lambda: _PersonExercise(vote_window))

    def update(self, tid, kxy, kconf):
        return self.people[tid].update(np.asarray(kxy), np.asarray(kconf))

    def forget(self, tid):
        self.people.pop(tid, None)


# ---------------------------------------------------------------------------
# NOTE — aller plus loin (plus d'exercices / plus de robustesse) :
#   Entraîner un petit classifieur sur séquences de keypoints normalisés.
#   Repos de référence :
#     - github.com/Alimustoofaa/YoloV8-Pose-Keypoint-Classification
#     - github.com/tringn/2D-Keypoints-based-Pose-Classifier
#   Pipeline : collecter des keypoints étiquetés (squat/pompe/...) -> normaliser
#   par la taille du torse -> entraîner un MLP/LSTM léger -> remplacer
#   _classify_frame() par l'inférence du modèle. Tourne aussi sur Raspberry Pi.
# ---------------------------------------------------------------------------
