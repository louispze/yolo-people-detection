# Backend P2F Smart Home (FastAPI) — à exécuter sur le Raspberry Pi

Serveur HTTP + WebSocket qui alimente l'application mobile P2F. Il diffuse les
flux caméra annotés (boîtes + compteur), le comptage de personnes par pièce
(avec activité et exercices), et persiste la configuration envoyée par l'app.

Point clé : **le serveur démarre et sert TOUJOURS**, même sans caméra ni modèle
YOLO. En l'absence de matériel/dépendance, il bascule automatiquement sur un
**mode substitution** (images synthétiques + comptage simulé). On peut donc le
tester sur un PC sans webcam.

---

## 1. Installation

```bash
cd app/backend
python -m venv .venv          # optionnel mais recommandé
# Windows : .venv\Scripts\activate   |   Linux/Pi : source .venv/bin/activate
pip install -r requirements_backend.txt
```

`ultralytics` (détection réelle YOLO pose) est **optionnel** : décommente-le dans
`requirements_backend.txt` pour activer le mode réel. Sans lui, tout fonctionne en
mode substitution.

---

## 2. Lancement

```bash
uvicorn server_app:app --host 0.0.0.0 --port 8000
```

ou directement :

```bash
python server_app.py
```

`--host 0.0.0.0` rend le serveur accessible depuis le téléphone sur le réseau
local. Au démarrage, les logs indiquent clairement le mode de chaque caméra :

```
[salon] MODE REEL : caméra source=0 + YOLO pose.
[cuisine] MODE SUBSTITUTION : frames synthétiques + comptage simulé.
```

### Service au démarrage du Pi (optionnel)

Exemple d'unité systemd (`/etc/systemd/system/p2f-backend.service`) :

```ini
[Unit]
Description=P2F Smart Home Backend
After=network-online.target

[Service]
WorkingDirectory=/home/pi/P2F_FABLE/app/backend
ExecStart=/home/pi/P2F_FABLE/app/backend/.venv/bin/uvicorn server_app:app --host 0.0.0.0 --port 8000
Restart=on-failure
Environment=P2F_IMGSZ=480

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now p2f-backend
```

---

## 3. Configurer les caméras

Le mapping `{roomId: source}` détermine quelles caméras sont ouvertes. **Le
`roomId` est la clé de la source** : il doit correspondre au `cameraSource` de la
pièce côté app (par défaut le `cameraSource` = l'`id` de la pièce, ex. `"salon"`).

Une `source` peut être :
- un **index entier** de webcam : `0`, `1`, ...
- une **URL RTSP** d'une caméra IP : `"rtsp://192.168.1.42:554/stream"`
- un **fichier vidéo** (tests) : `"demo.mp4"` (rejoué en boucle)

Trois façons de configurer (par ordre de priorité) :

1. **Variable d'environnement** `P2F_CAMERAS` (JSON inline) :
   ```bash
   P2F_CAMERAS='{"salon":0,"cuisine":1}' uvicorn server_app:app --host 0.0.0.0 --port 8000
   ```
2. **Fichier `cameras.json`** dans `app/backend/` (voir `cameras.example.json`) :
   ```json
   { "salon": 0, "cuisine": 1 }
   ```
3. **Défaut** si rien n'est fourni : `{ "salon": 0 }`.

### Autres variables d'environnement utiles

| Variable            | Défaut             | Rôle                                                |
|---------------------|--------------------|-----------------------------------------------------|
| `P2F_CAMERAS`       | —                  | mapping caméras JSON inline                          |
| `P2F_MODEL`         | `yolo11n-pose.pt`  | modèle YOLO pose (cherché dans `app/`)              |
| `P2F_IMGSZ`         | `640`              | taille d'inférence (mettre `480`/`320` sur Pi)      |
| `P2F_FPS`           | `8`                | images analysées/s par caméra                       |
| `P2F_STREAM_FPS`    | `15`               | cadence d'envoi du flux MJPEG                        |
| `P2F_JPEG_QUALITY`  | `75`               | qualité JPEG (0-100)                                |
| `P2F_FAKE`          | —                  | `1` force le mode substitution (test sans matériel) |
| `P2F_HOST`/`P2F_PORT` | `0.0.0.0`/`8000` | hôte/port quand lancé via `python server_app.py`    |

---

## 4. Contrat d'API

Tout est sous le préfixe `/api` (sauf le WebSocket `/ws`). CORS est grand ouvert.

| Méthode | Route                     | Réponse                                                            |
|---------|---------------------------|-------------------------------------------------------------------|
| GET     | `/api/health`             | `{"ok": true}`                                                    |
| GET     | `/api/config`             | config persistée (ou `{}` si jamais enregistrée)                  |
| PUT     | `/api/config`             | body JSON quelconque -> écrit `config.json`, renvoie `{"ok":true}`|
| GET     | `/api/state`              | `{"rooms": {...}, "temps": {...}}` (voir ci-dessous)             |
| WS      | `/ws`                     | pousse toutes les 0,5 s `{"type":"state","rooms":{...},"temps":{...}}` |
| GET     | `/api/stream/{source}`    | flux **MJPEG** annoté (`multipart/x-mixed-replace; boundary=frame`) |
| GET     | `/api/snapshot/{source}`  | une **image JPEG** annotée                                       |

### Forme de `/api/state` et des messages `/ws`

```json
{
  "rooms": {
    "salon": {
      "people": 2,
      "active": 1,
      "workout": 0,
      "exercises": { "squat": 1 },
      "updatedAt": 1718480000000
    }
  },
  "temps": { "salon": 22.4 }
}
```

- `rooms[roomId]` correspond au type `LiveRoomState` de l'app
  (`people`, `active`, `workout`, `exercises`, `updatedAt` en ms).
- `temps[roomId]` est une température en °C. ⚠️ Aucune sonde n'est branchée sur
  ce backend : la valeur est **simulée** (dérive lente + influence de
  l'occupation). À remplacer par une vraie lecture capteur (DS18B20 1-wire,
  MQTT, etc.) dans `simulated_temp()`.

### Annotation des flux

Chaque flux/snapshot superpose :
- une **boîte** par personne (verte = présente, orange = active, rouge =
  en exercice) avec son ID et, le cas échéant, l'exercice + le nombre de reps ;
- un **bandeau** `Personnes: N | Actives: A | Sport: W` et une étiquette
  `source [live|simu]` indiquant le mode.

---

## 5. Connexion depuis l'app mobile

Dans l'app, écran **Réglages** :
1. **Hôte** = l'adresse IP du Raspberry Pi sur le réseau local (ex. `192.168.1.50`).
   La trouver sur le Pi : `hostname -I`.
2. **Port** = `8000`.
3. **TLS** = désactivé (le backend sert en HTTP en clair sur le LAN).

L'app teste alors `GET http://<ip>:8000/api/health`, ouvre le WebSocket
`ws://<ip>:8000/ws` pour l'état temps réel, et affiche les caméras via
`http://<ip>:8000/api/stream/<cameraSource>`.

Le `cameraSource` de chaque pièce (réglable dans l'app, par défaut = l'id de la
pièce) doit correspondre à une clé du mapping caméras du backend pour afficher un
vrai flux ; sinon une image « source inconnue » est servie.

### Vérification rapide

```bash
curl http://<ip-du-pi>:8000/api/health        # -> {"ok":true}
curl http://<ip-du-pi>:8000/api/state         # -> rooms + temps
# ouvrir dans un navigateur : http://<ip-du-pi>:8000/api/stream/salon
```

---

## 6. Architecture (résumé)

- **`CameraManager`** : un `CameraWorker` (thread dédié) par source caméra ;
  agrège l'état de toutes les pièces pour `/api/state` et `/ws`.
- **`CameraWorker`** : tente le mode **réel** (OpenCV + `PeopleAnalyzer` importé
  depuis `app/`). En cas d'échec (ultralytics absent, modèle manquant, caméra
  injoignable) ou de perte de caméra en cours de route, bascule sur
  **`FakeDetector`** (substitution). Conserve toujours la dernière frame JPEG
  annotée + le dernier comptage.
- **Détecteur pluggable** : `PeopleAnalyzer.process(frame)` renvoie
  `{count, active_count, workout_count, persons[...]}` ; le backend en dérive le
  `LiveRoomState` (dont l'agrégation des exercices par nom).
