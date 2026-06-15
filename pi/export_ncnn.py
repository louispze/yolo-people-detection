"""
export_ncnn.py — Exporte yolo11n-pose.pt en modèle NCNN (backend ARM rapide).

À lancer UNE fois (sur ton PC de préférence, c'est plus rapide), puis copier le
dossier 'yolo11n-pose_ncnn_model/' à la racine du projet sur le Pi 3.

    python export_ncnn.py            # imgsz 256 par défaut (bon compromis Pi 3)
    python export_ncnn.py --imgsz 320

Nécessite ultralytics installé (pip install ultralytics).
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolo11n-pose.pt")
ap.add_argument("--imgsz", type=int, default=256, help="256 (rapide) ou 320 (plus précis)")
args = ap.parse_args()

YOLO(args.model).export(format="ncnn", imgsz=args.imgsz)
print(f"OK -> dossier '{args.model.replace('.pt', '')}_ncnn_model' (à copier sur le Pi)")
