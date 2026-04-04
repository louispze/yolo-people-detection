import cv2
from ultralytics import YOLO
from tracker import Tracker

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SOURCE      = 0          # 0 = webcam | "video.mp4" = fichier vidéo
MODEL_PATH  = "yolov8n.pt"   # nano = rapide | yolov8s.pt = plus précis
CONFIDENCE  = 0.4            # seuil de confiance YOLO (0.0 - 1.0)
MAX_AGE     = 30             # frames avant suppression d'une personne perdue
MAX_DIST    = 80             # distance max (px) pour relier une détection à une personne

# ------------------------------------------------------------------
# INIT
# ------------------------------------------------------------------
model   = YOLO(MODEL_PATH)   # téléchargement auto si absent
tracker = Tracker(max_age=MAX_AGE, max_distance=MAX_DIST)
cap     = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("❌ Impossible d'ouvrir la source vidéo.")
    exit()

print("✅ Démarrage — appuie sur 'q' pour quitter")

# ------------------------------------------------------------------
# BOUCLE PRINCIPALE
# ------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Détection YOLO (classe 0 = personne) ---
    results = model(frame, classes=[0], conf=CONFIDENCE, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cx   = (x1 + x2) // 2
        cy   = (y1 + y2) // 2
        detections.append((cx, cy, x1, y1, x2, y2, conf))

    # --- Mise à jour du tracker ---
    persons = tracker.update(detections)

    # --- Dessin des boîtes et trajectoires ---
    for person in persons:
        r, g, b = person.getRGB()
        color   = (b, g, r)   # OpenCV = BGR

        # Boîte englobante
        if person.bbox:
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {person.getId()}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2)

        # Trajectoire
        tracks = person.getTracks()
        for i in range(1, len(tracks)):
            cv2.line(frame, tracks[i - 1], tracks[i], color, 2)

    # --- Compteur en haut à gauche ---
    count = tracker.count
    cv2.rectangle(frame, (0, 0), (220, 50), (0, 0, 0), -1)   # fond noir
    cv2.putText(frame, f"Personnes : {count}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2)

    cv2.imshow("Comptage de personnes", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print(f"\n📊 Total personnes vues : {tracker.total_seen}")
