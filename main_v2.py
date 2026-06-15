"""
main_v2.py — Démo webcam améliorée (comptage stable + activité physique)
========================================================================

Remplace main.py. Utilise PeopleAnalyzer (analyzer.py) :
  - comptage qui ne saute plus quand ça bouge (tracking + débounce)
  - étiquette ACTIF / IMMOBILE par personne (mouvement des keypoints)
  - squelette dessiné

Touches :  q = quitter   |   k = afficher/masquer le squelette
"""

import time

import cv2

from analyzer import PeopleAnalyzer

SOURCE = 0          # 0 = webcam | "video.mp4" = fichier
IMG_SIZE = 640      # baisser à 480/320 sur Raspberry Pi

# Squelette COCO (paires de keypoints à relier)
COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),         # bras
    (11, 13), (13, 15), (12, 14), (14, 16),  # jambes
    (5, 6), (11, 12), (5, 11), (6, 12),      # torse
    (3, 5), (4, 6), (0, 1), (0, 2), (1, 3), (2, 4),  # tête
]

GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def draw_skeleton(frame, kxy, kconf_ok=None):
    for a, b in COCO_SKELETON:
        xa, ya = kxy[a]
        xb, yb = kxy[b]
        if xa > 0 and ya > 0 and xb > 0 and yb > 0:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (200, 200, 0), 2)
    for x, y in kxy:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)


def main():
    analyzer = PeopleAnalyzer(imgsz=IMG_SIZE)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la source vidéo.")
        return

    print("✅ Démarrage — 'q' pour quitter, 'k' pour le squelette")
    show_skeleton = True
    prev_t = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = analyzer.process(frame)

        for p in result["persons"]:
            if not p["confirmed"]:
                continue  # pas encore assez vue → on ne l'affiche pas (anti-clignotement)

            x1, y1, x2, y2 = map(int, p["bbox"])
            color = RED if p["active"] else GREEN
            etat = "ACTIF" if p["active"] else "IMMOBILE"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Ligne 1 : ID + état mouvement
            label = f"ID {p['id']} - {etat} ({p['activity_score']:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

            # Ligne 2 : exercice reconnu + compteur de répétitions
            if p.get("is_workout"):
                ex_label = f"{p['exercise'].upper()} x{p['reps']}"
                (ew, eh), _ = cv2.getTextSize(ex_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y2), (x1 + ew + 6, y2 + eh + 10), (40, 40, 220), -1)
                cv2.putText(frame, ex_label, (x1 + 3, y2 + eh + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

            if show_skeleton and p["keypoints"] is not None:
                draw_skeleton(frame, p["keypoints"])

        # FPS lissé
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Bandeau compteur
        cv2.rectangle(frame, (0, 0), (520, 70), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"Personnes : {result['count']}   Actives : {result['active_count']}   Sport : {result['workout_count']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS : {fps:.1f}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("P2F - Comptage + Activite", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('k'):
            show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
