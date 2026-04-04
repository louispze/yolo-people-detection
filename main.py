import cv2
import random
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SOURCE     = 0             # 0 = webcam | "video.mp4" = fichier vidéo
MODEL_PATH = "yolov8n.pt"  # nano = rapide | yolov8s.pt = plus précis
CONFIDENCE = 0.4           # seuil de confiance YOLO

# ------------------------------------------------------------------
# INIT
# ------------------------------------------------------------------
model   = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30)   # max_age = frames avant oubli d'un ID

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("❌ Impossible d'ouvrir la source vidéo.")
    exit()

print("✅ Démarrage — appuie sur 'q' pour quitter")

# Couleur fixe par ID (seed = ID → toujours la même couleur pour cette personne)
id_colors = {}

def get_color(track_id):
    if track_id not in id_colors:
        random.seed(track_id)
        id_colors[track_id] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return id_colors[track_id]

# ------------------------------------------------------------------
# BOUCLE PRINCIPALE
# ------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Détection YOLO (classe 0 = personne) ---
    results = model(frame, classes=[0], conf=CONFIDENCE, verbose=False)[0]

    # Format attendu par DeepSORT : liste de ([x1, y1, w, h], confidence, class)
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))

    # --- Mise à jour DeepSORT ---
    tracks = tracker.update_tracks(detections, frame=frame)

    active_count = 0
    for track in tracks:
        if not track.is_confirmed():
            continue

        active_count += 1
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        color = get_color(track_id)

        # Boîte englobante
        cv2.rectangle(frame, (l, t), (r, b), color, 2)

        # Label avec ID
        label = f"ID {track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (l, t - th - 8), (l + tw + 6, t), color, -1)
        cv2.putText(frame, label, (l + 3, t - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Compteur en haut à gauche ---
    cv2.rectangle(frame, (0, 0), (250, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Personnes : {active_count}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2)

    cv2.imshow("Comptage de personnes - DeepSORT", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
