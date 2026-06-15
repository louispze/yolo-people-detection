"""
full_pi.py — Version COMPLÈTE pour Raspberry Pi 3 (1 Go) via NCNN
================================================================

Comptage stable + détection d'activité + reconnaissance d'exercices, sur le Pi,
en réutilisant EXACTEMENT le même cœur que la version PC (analyzer.py /
exercises.py). La seule différence : le modèle pose est chargé au format **NCNN**
(backend ARM optimisé, sans inférence PyTorch) et en basse résolution.

Perf attendue sur Pi 3 : ~1-2 img/s à imgsz=256. Suffisant pour piloter un
chauffage selon l'occupation ET l'activité de la pièce.

Pré-requis sur le Pi :
  - ultralytics installé (sert de wrapper + tracking ; l'inférence passe par NCNN)
  - le dossier 'yolo11n-pose_ncnn_model/' copié à la racine du projet
    (génré par  python export_ncnn.py  sur ton PC)

Exemples :
    python3 full_pi.py --headless
    python3 full_pi.py --headless --room salon --mqtt-host 192.168.1.20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

# réutilise le cœur du projet (app/analyzer.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyzer import PeopleAnalyzer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def make_mqtt(args):
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
    base = f"p2f/{args.room}"
    print(f"[MQTT] publication sous {base}/* ({args.mqtt_host})", flush=True)

    def publish(count, active, workout, exercises):
        client.publish(f"{base}/people", str(count), retain=True)
        client.publish(f"{base}/active", str(active), retain=True)
        client.publish(f"{base}/workout", str(workout), retain=True)
        client.publish(f"{base}/exercises", json.dumps(exercises), retain=True)

    return publish


def main():
    ap = argparse.ArgumentParser(description="Version complète Pi 3 (NCNN) : comptage + activité + exercices")
    ap.add_argument("--source", default="0", help="0 = webcam | chemin vidéo | url rtsp")
    ap.add_argument("--room", default="salon")
    ap.add_argument("--model", default=str(ROOT / "yolo11n-pose_ncnn_model"),
                    help="dossier du modèle NCNN exporté")
    ap.add_argument("--imgsz", type=int, default=256, help="doit correspondre à l'export (256/320)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--max-fps", type=float, default=0, help="limite CPU (0 = max)")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--mqtt-host", default=None)
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--mqtt-user", default=None)
    ap.add_argument("--mqtt-pass", default=None)
    args = ap.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"Modèle NCNN introuvable : {args.model}\n"
                 f"Génère-le avec  python export_ncnn.py  puis copie le dossier sur le Pi.")

    source = int(args.source) if args.source.isdigit() else args.source
    analyzer = PeopleAnalyzer(model_path=args.model, imgsz=args.imgsz,
                              conf=args.conf, detect_exercises=True)
    publish = make_mqtt(args)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir la source : {args.source}")

    show = args.display and not args.headless
    last_key = None
    fails = 0
    min_interval = (1.0 / args.max_fps) if args.max_fps > 0 else 0.0
    print(f"[OK] Version complète démarrée (room={args.room}, NCNN imgsz={args.imgsz}).", flush=True)

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                fails += 1
                if isinstance(source, str) and not source.startswith("rtsp"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if fails > 50:
                    sys.exit("Source vidéo indisponible (50 échecs de lecture).")
                time.sleep(0.1)
                continue
            fails = 0

            res = analyzer.process(frame)

            # Récapitulatif des exercices en cours { "squat": 1, "pompes": 1, ... }
            exercises = {}
            for p in res["persons"]:
                if p["confirmed"] and p["is_workout"]:
                    exercises[p["exercise"]] = exercises.get(p["exercise"], 0) + 1

            key = (res["count"], res["active_count"], res["workout_count"], tuple(sorted(exercises.items())))
            if key != last_key:
                msg = (f"[{time.strftime('%H:%M:%S')}] {args.room} : "
                       f"{res['count']} pers. | {res['active_count']} actives | "
                       f"{res['workout_count']} sport")
                if exercises:
                    msg += " | " + ", ".join(f"{k}:{v}" for k, v in exercises.items())
                print(msg, flush=True)
                if publish:
                    publish(res["count"], res["active_count"], res["workout_count"], exercises)
                last_key = key

            if show:
                for p in res["persons"]:
                    if not p["confirmed"]:
                        continue
                    x1, y1, x2, y2 = map(int, p["bbox"])
                    color = (0, 0, 255) if p["active"] else (0, 200, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    txt = f"ID{p['id']}"
                    if p["is_workout"]:
                        txt += f" {p['exercise']} x{p['reps']}"
                    cv2.putText(frame, txt, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, (0, 0), (430, 40), (0, 0, 0), -1)
                cv2.putText(frame, f"Pers:{res['count']} Actives:{res['active_count']} Sport:{res['workout_count']}",
                            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("P2F - Complet (Pi 3 / NCNN)", frame)
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
