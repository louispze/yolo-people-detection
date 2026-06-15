# Détection de personnes — Raspberry Pi 3 (1 Go)

Deux versions au choix selon ce que tu veux faire tourner sur le Pi 3 :

| Version | Fichier | Ce qu'elle fait | RAM | Vitesse | Install |
|---|---|---|---|---|---|
| **Légère** | `counter_pi.py` | Comptage de personnes seul | ~150 Mo | ~3-5 img/s | simple (OpenCV) |
| **Complète** | `full_pi.py` | Comptage + activité + exercices | ~450-500 Mo | ~1-2 img/s | + ultralytics (NCNN) |

Les deux publient en **MQTT** pour Home Assistant. Le Pi 3 (1 Go) fait tourner les
deux ; la complète est plus lourde mais reste dans la RAM (garde du swap en marge).

---

## Version LÉGÈRE — `counter_pi.py` (compteur seul)

Détecteur MobileNet-SSD via OpenCV DNN, **sans PyTorch**. Robuste et rapide.

```bash
sudo apt update && sudo apt install -y python3-opencv python3-pip
pip3 install paho-mqtt                      # optionnel (MQTT)

cd yolo-people-detection/pi
python3 get_model.py                        # télécharge MobileNet-SSD
python3 counter_pi.py --source 0 --headless
# avec Home Assistant :
python3 counter_pi.py --source 0 --headless --room salon --mqtt-host 192.168.1.20
```
Réglages utiles : `--conf`, `--size` (300/224), `--min-hits`, `--max-age`, `--max-fps`.

---

## Version COMPLÈTE — `full_pi.py` (comptage + activité + exercices)

Réutilise le même cœur que la version PC (`analyzer.py` / `exercises.py`), mais le
modèle pose est exécuté en **NCNN** (backend ARM, pas d'inférence PyTorch) à basse
résolution.

### 1. Générer le modèle NCNN (sur ton PC, plus rapide)
```bash
pip install ultralytics
cd pi
python export_ncnn.py                 # crée yolo11n-pose_ncnn_model/ (imgsz 256)
```
Copie ensuite le dossier **`yolo11n-pose_ncnn_model/`** à la racine du projet sur le Pi
(à côté de `analyzer.py`).

### 2. Installer sur le Pi 3
```bash
sudo apt install -y python3-opencv python3-pip
pip3 install ultralytics paho-mqtt    # ultralytics sert de wrapper + tracking
```
> ⚠️ L'install d'ultralytics (qui tire PyTorch) est longue sur Pi 3. C'est normal.
> Augmente le swap d'abord (voir plus bas).

### 3. Lancer
```bash
cd yolo-people-detection/pi
python3 full_pi.py --headless --room salon
# avec Home Assistant :
python3 full_pi.py --headless --room salon --mqtt-host 192.168.1.20
```
Sortie console (et MQTT) typique :
```
[20:39] salon : 2 pers. | 1 actives | 1 sport | squat:1
```

---

## Augmenter le swap (recommandé pour la version complète)
```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo dphys-swapfile swapon
```

## Home Assistant (MQTT)
Topics publiés (version complète) :
`p2f/<pièce>/people`, `/active`, `/workout`, `/exercises` (JSON).
La version légère publie seulement `p2f/<pièce>/people`.

```yaml
mqtt:
  sensor:
    - name: "Personnes Salon"
      state_topic: "p2f/salon/people"
      unit_of_measurement: "pers."
    - name: "Actives Salon"
      state_topic: "p2f/salon/active"
    - name: "Sport Salon"
      state_topic: "p2f/salon/workout"
```
Puis une automatisation HA applique ta logique température (ex. « ≥ X personnes
ou activité sportive → ajuster la consigne »).

## Démarrage automatique (systemd)
`/etc/systemd/system/p2f.service` :
```ini
[Unit]
Description=P2F Detection
After=network-online.target

[Service]
# version légère :
ExecStart=/usr/bin/python3 /home/pi/yolo-people-detection/pi/counter_pi.py --source 0 --headless --room salon --mqtt-host 192.168.1.20
# (ou version complète : remplacer par full_pi.py)
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```
`sudo systemctl enable --now p2f`

## Conseils perf Pi 3
- Version complète lente ? exporte en `--imgsz 256` (déjà par défaut) et lance avec
  `--max-fps 2` pour limiter la chauffe.
- Besoin d'un peu plus de précision ? réexporte en `--imgsz 320` (plus lent).
- Le Pi 3 chauffe en charge continue : prévois un petit dissipateur.
