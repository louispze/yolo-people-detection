"""
counter_pi.py — Compteur de personnes léger pour Raspberry Pi Zero 2 W
======================================================================

Contraintes du Pi Zero 2 W : 512 Mo de RAM, CPU 1 GHz. PyTorch/Ultralytics
n'y tiennent PAS. Cette version :
  - détecte les personnes avec MobileNet-SSD via OpenCV DNN (≈100-150 Mo RAM,
    pas de torch) ;
  - réutilise TON tracker par centroïdes (tracker.py / person.py) pour des IDs
    stables ;
  - applique un débounce (min_hits / max_age) pour un compteur qui ne saute pas ;
  - publie le compte en MQTT (optionnel) pour Home Assistant.

⚠️ Pas de pose ni d'exercices ici : trop lourd pour le Pi Zero. Objectif =
   COMPTER LES PERSONNES de façon fiable. (Le pose/exercices restent pour la
   version PC/serveur.)

Exemples :
    python3 counter_pi.py --source 0 --display
    python3 counter_pi.py --source 0 --headless --mqtt-host 192.168.1.20 --room salon
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# --- réutilise le tracker existant à la racine du projet (app/) ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracker import Tracker  # noqa: E402

PERSON_CLASS = 15  # MobileNet-SSD (VOC) : 15 = person
HERE = Path(__file__).resolve().parent


def build_net(model_dir: Path):
    proto = model_dir / "MobileNetSSD_deploy.prototxt"
    weights = model_dir / "MobileNetSSD_deploy.caffemodel"
    if not proto.exists() or not weights.exists():
        sys.exit("Modèle introuvable. Lance d'abord :  python3 get_model.py")
    net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def detect_people(net, frame, conf_thresh, size):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (size, size)),
                                 0.007843, (size, size), 127.5)
    net.setInput(blob)
    det = net.forward()  # (1,1,N,7) : [_, classe, conf, x1,y1,x2,y2] normalisés
    out = []
    for i in range(det.shape[2]):
        if int(det[0, 0, i, 1]) != PERSON_CLASS:
            continue
        conf = float(det[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        x1 = int(det[0, 0, i, 3] * w); y1 = int(det[0, 0, i, 4] * h)
        x2 = int(det[0, 0, i, 5] * w); y2 = int(det[0, 0, i, 6] * h)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        out.append((cx, cy, x1, y1, x2, y2, conf))
    return out


def make_mqtt(args):
    """Retourne une fonction publish(count) ou None si MQTT non demandé."""
    if not args.mqtt_host:
        return None
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        sys.exit("paho-mqtt manquant :  pip install paho-mqtt")
    client = mqtt.Client()
    if args.mqtt_user:
        client.username_pw_set(args.mqtt_user, args.mqtt_pass)
    client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
    client.loop_start()
    topic = args.mqtt_topic or f"p2f/{args.room}/people"
    print(f"[MQTT] publication sur {topic} ({args.mqtt_host}:{args.mqtt_port})")

    def publish(count):
        client.publish(topic, payload=str(count), qos=0, retain=True)

    return publish


def main():
    ap = argparse.ArgumentParser(description="Compteur de personnes léger (Pi Zero 2 W)")
    ap.add_argument("--source", default="0", help="0 = webcam | chemin vidéo | url rtsp")
    ap.add_argument("--room", default="salon", help="nom de la pièce (topic MQTT)")
    ap.add_argument("--conf", type=float, default=0.5, help="seuil de confiance détection")
    ap.add_argument("--size", type=int, default=300, help="taille d'entrée réseau (300 standard, 224 plus rapide)")
    ap.add_argument("--min-hits", type=int, default=3, help="images avant de compter une personne")
    ap.add_argument("--max-age", type=int, default=30, help="images sans détection avant de la décompter")
    ap.add_argument("--max-distance", type=int, default=80, help="distance max d'association (px)")
    ap.add_argument("--max-fps", type=float, default=0, help="limite le nb d'images traitées/s (0 = max)")
    ap.add_argument("--display", action="store_true", help="affiche une fenêtre (PC/desktop)")
    ap.add_argument("--headless", action="store_true", help="aucune fenêtre (Pi sans écran)")
    # MQTT (optionnel)
    ap.add_argument("--mqtt-host", default=None)
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--mqtt-user", default=None)
    ap.add_argument("--mqtt-pass", default=None)
    ap.add_argument("--mqtt-topic", default=None, help="défaut : p2f/<room>/people")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    net = build_net(HERE / "models")
    tracker = Tracker(max_age=args.max_age, max_distance=args.max_distance)
    publish = make_mqtt(args)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir la source : {args.source}")

    show = args.display and not args.headless
    hits = {}                # id -> nb d'images vu (pour le débounce d'entrée)
    last_count = -1
    fails = 0                # échecs de lecture consécutifs
    min_interval = (1.0 / args.max_fps) if args.max_fps > 0 else 0.0
    print(f"[OK] Compteur démarré (room={args.room}). Ctrl+C pour arrêter.", flush=True)

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                # Fichier vidéo terminé -> rebobine ; caméra en échec -> on temporise
                fails += 1
                if isinstance(source, str) and not source.startswith("rtsp"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if fails > 50:
                    sys.exit("Source vidéo indisponible (50 échecs de lecture). "
                             "Caméra occupée par un autre programme ou débranchée ?")
                time.sleep(0.1)
                continue
            fails = 0

            dets = detect_people(net, frame, args.conf, args.size)
            persons = tracker.update(dets)

            # Débounce : ne compter que les personnes vues >= min_hits
            present_ids = set()
            for p in persons:
                pid = p.getId()
                present_ids.add(pid)
                if p.age == 0:                       # détectée cette image
                    hits[pid] = hits.get(pid, 0) + 1
            for pid in list(hits):                   # purge des IDs disparus
                if pid not in present_ids:
                    hits.pop(pid, None)

            count = sum(1 for p in persons if hits.get(p.getId(), 0) >= args.min_hits)

            if count != last_count:
                print(f"[{time.strftime('%H:%M:%S')}] {args.room} : {count} personne(s)", flush=True)
                if publish:
                    publish(count)
                last_count = count

            if show:
                for p in persons:
                    if hits.get(p.getId(), 0) < args.min_hits or p.bbox is None:
                        continue
                    x1, y1, x2, y2 = p.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(frame, f"ID {p.getId()}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, (0, 0), (230, 40), (0, 0, 0), -1)
                cv2.putText(frame, f"Personnes : {count}", (8, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("P2F - Compteur (Pi)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if min_interval:
                time.sleep(max(0.0, min_interval - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nArrêt demandé.")
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
