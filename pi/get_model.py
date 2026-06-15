"""
get_model.py — Télécharge le modèle MobileNet-SSD (OpenCV DNN) dans pi/models/.
À lancer une fois sur le Raspberry Pi :  python3 get_model.py
N'utilise que la lib standard (pas de curl/wget requis).
"""
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
MODELS.mkdir(exist_ok=True)

FILES = {
    "MobileNetSSD_deploy.prototxt":
        "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel":
        "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel",
}

for name, url in FILES.items():
    dst = MODELS / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[OK] déjà présent : {name}")
        continue
    print(f"[..] téléchargement {name} …")
    urllib.request.urlretrieve(url, dst)
    print(f"[OK] {name} ({dst.stat().st_size} octets)")

print("Modèle prêt dans", MODELS)
