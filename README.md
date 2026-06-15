# P2F Smart Home

Système domotique « maison intelligente » qui pilote le **climat selon l'occupation
réelle** des pièces, détectée par caméra. Une caméra observe une pièce, un modèle
**YOLO pose** compte les personnes, mesure leur activité (immobile / en mouvement /
en train de faire du sport) et reconnaît quelques exercices. Une **app mobile
Android** (style Home Assistant) affiche l'occupation, les caméras en direct et la
**consigne de température adaptative** calculée pièce par pièce.

> En clair : plus il y a de monde — et plus ça bouge — dans une pièce, plus la
> consigne de chauffage baisse automatiquement (la chaleur corporelle chauffe déjà).

---

## Vue d'ensemble de l'architecture

Le projet est en trois parties : la **détection** (vision), le **backend sur le
Raspberry Pi** (caméra + API), et l'**app mobile** (interface). Le flux nominal va
de la caméra jusqu'au téléphone ; Home Assistant / MQTT est une intégration
optionnelle pour piloter de vrais équipements.

```
   ┌──────────┐      images       ┌───────────────────────────────────────┐
   │  Caméra  │ ────────────────▶ │            RASPBERRY PI                │
   │ (USB/IP) │                   │                                       │
   └──────────┘                   │  ┌─────────────────────────────────┐  │
                                  │  │ Détection YOLO pose (NCNN/ARM)  │  │
                                  │  │  analyzer.py + exercises.py     │  │
                                  │  │  → count / active / workout /   │  │
                                  │  │    exercises par pièce          │  │
                                  │  └───────────────┬─────────────────┘  │
                                  │                  │                     │
                                  │     ┌────────────┴───────────┐         │
                                  │     │  Backend FastAPI       │         │
                                  │     │  (app/backend/)        │         │
                                  │     │  REST /api/* + /ws +   │         │
                                  │     │  flux caméra MJPEG     │         │
                                  │     └────────────┬───────────┘         │
                                  └──────────────────┼─────────────────────┘
                                                     │
                  WebSocket (état temps réel)        │   MJPEG (flux vidéo)
                  ws://<ip-pi>:<port>/ws             │   /api/stream/<caméra>
                                                     │
                                       ┌─────────────▼─────────────┐
                                       │     APP MOBILE ANDROID    │
                                       │     (app/mobile/)         │
                                       │  React + Vite + Capacitor │
                                       │  • plan de la maison      │
                                       │  • caméras en direct      │
                                       │  • climat adaptatif       │
                                       │  • préférences/personne   │
                                       └───────────────────────────┘

        ── OPTION ──────────────────────────────────────────────────────────
        Détection (Pi) ──MQTT──▶ Home Assistant ──▶ chauffage / thermostats réels
        Topics : p2f/<pièce>/people | /active | /workout | /exercises
```

Deux manières de consommer la détection, non exclusives :

- **App mobile** (chemin principal) : le backend FastAPI sert l'état via WebSocket
  et les caméras en MJPEG ; l'app affiche tout et calcule la consigne côté client.
- **Home Assistant** (option) : la détection publie en MQTT, et une automatisation
  HA applique la logique de température sur de vrais thermostats.

---

## Les composants

### 1. Détection (PC / serveur) — `app/`

Le cœur vision, qui tourne aussi bien sur PC pour le développement que sur le Pi.

| Fichier | Rôle |
|---|---|
| `analyzer.py` | Cœur de détection : YOLO pose en **tracking** (ByteTrack) → comptage stable (anti-clignotement) + score d'activité par personne. |
| `exercises.py` | Reconnaissance d'exercices (squat, pompes, jumping jack) + comptage de répétitions par règles géométriques sur les keypoints. |
| `main_v2.py` | Démo webcam locale (fenêtre OpenCV) : squelette, état ACTIF/IMMOBILE, compteur de reps. |
| `server.py` | Pont multi-caméras → WebSocket (`ws://localhost:8765`) diffusant `people` / `activity` / `workout` par pièce. |
| `yolo11n-pose.pt` | Modèle pose PyTorch (PC). Sur le Pi, on utilise sa version NCNN (`yolo11n-pose_ncnn_model/`). |

Sortie de `analyzer.process(frame)` : `{ count, active_count, workout_count, persons[] }`.

### 2. Raspberry Pi — `app/pi/` + `app/backend/`

Le Pi est l'appareil « toujours allumé » qui voit la pièce et expose les données.

- **`app/pi/`** — versions optimisées ARM de la détection :
  - `counter_pi.py` : version **légère** (MobileNet-SSD via OpenCV, sans PyTorch) →
    comptage seul, ~3-5 img/s, ~150 Mo de RAM.
  - `full_pi.py` : version **complète** (même cœur `analyzer.py`/`exercises.py`, modèle
    pose en **NCNN**) → comptage + activité + exercices, ~1-2 img/s.
  - `export_ncnn.py` / `get_model.py` : génération/téléchargement des modèles.
  - Les deux publient en **MQTT** pour Home Assistant.
- **`app/backend/`** — serveur **FastAPI** consommé par l'app mobile : expose l'état
  temps réel (REST + WebSocket) et relaie les **flux caméra en MJPEG**.

### 3. App mobile Android — `app/mobile/`

Tableau de bord type Home Assistant, en **React + Vite + Capacitor** (TypeScript
strict, état Zustand). Écrans : tableau de bord, plan de la maison éditable,
personnes, caméras, réglages. Fonctions clés :

- **Plan de la maison éditable** : positionner/dimensionner chaque pièce sur une grille.
- **Préférences de température par personne** : chacun a sa température idéale.
- **Climat adaptatif** (`src/lib/engine.ts`) : la consigne d'une pièce part de la
  préférence des personnes présentes (moyenne ou « la plus fraîche »), puis baisse
  selon l'occupation, le mouvement et l'activité physique détectés en direct — avec
  une explication lisible (« 3 personnes → −1°C », « 1 en exercice → −1°C »).
- **Caméras en direct** : flux MJPEG du backend affichés dans l'app.

---

## Contrat backend ↔ app mobile

L'app attend ces endpoints côté backend FastAPI (voir `mobile/src/lib/api.ts`) :

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/api/health` | GET | sonde de disponibilité → `{ ok: true }` |
| `/api/config` | GET / PUT | lire / enregistrer la configuration (pièces, personnes, lumières, climat) |
| `/api/state` | GET | snapshot `{ rooms: {id: LiveRoomState}, temps: {id} }` |
| `/ws` | WebSocket | flux temps réel, messages `{ type:"state", rooms, temps }` |
| `/api/stream/{caméra}` | GET | flux **MJPEG** (`multipart/x-mixed-replace`) |
| `/api/snapshot/{caméra}` | GET | image fixe |

`LiveRoomState` = `{ people, active, workout, exercises:{}, updatedAt }`.

---

## Parcours type (mise en route)

1. **Brancher la caméra** sur le Raspberry Pi (USB ou flux IP/RTSP).
2. **Installer et lancer le backend sur le Pi** (caméra + API + WebSocket + MJPEG) —
   voir [`backend/README_BACKEND.md`](backend/README_BACKEND.md). Pour la détection
   seule, voir [`pi/README_PI.md`](pi/README_PI.md) (versions légère / complète).
   Notez l'**adresse IP du Pi** et le **port** du backend.
3. **Installer l'app mobile** sur le téléphone Android (génération de l'APK) —
   voir [`mobile/BUILD_APK.md`](mobile/BUILD_APK.md).
4. **Renseigner l'IP du Pi** dans l'écran *Réglages* de l'app (hôte + port). Le statut
   passe à « en ligne » dès que le WebSocket est connecté.
5. **Profiter** : l'app affiche l'**occupation** par pièce, les **caméras en direct**
   et la **consigne de température adaptative** calculée en temps réel selon qui est
   présent et ce qu'il fait.
6. *(Optionnel)* **Home Assistant** : activer la publication MQTT côté Pi pour piloter
   de vrais thermostats à partir des mêmes données (voir `pi/README_PI.md`).

---

## Documentation détaillée

- **Raspberry Pi (détection légère/complète, MQTT, swap, systemd)** →
  [`pi/README_PI.md`](pi/README_PI.md)
- **Backend FastAPI (API, WebSocket, flux caméra)** →
  [`backend/README_BACKEND.md`](backend/README_BACKEND.md)
- **App mobile (build de l'APK, Capacitor/Android)** →
  [`mobile/BUILD_APK.md`](mobile/BUILD_APK.md)
- **Contrat frontend (store, types, composants)** →
  [`mobile/CONTRACT.md`](mobile/CONTRACT.md)

> Note : `backend/` et `mobile/BUILD_APK.md` sont les emplacements prévus pour le
> serveur FastAPI et le guide de build APK. S'ils ne sont pas encore présents, ce
> README décrit l'architecture cible et le contrat d'API à respecter (voir
> `mobile/src/lib/api.ts`).

---

## Stack technique

- **Vision** : Ultralytics YOLO (pose `yolo11n-pose`), ByteTrack, OpenCV, NumPy ;
  NCNN sur ARM ; MobileNet-SSD (OpenCV DNN) pour la version légère.
- **Backend Pi** : Python, FastAPI (REST + WebSocket + MJPEG), paho-mqtt (option).
- **App mobile** : React 18, Vite, TypeScript strict, Zustand, lucide-react,
  Capacitor (Android).
- **Intégration** : MQTT / Home Assistant (optionnel).
