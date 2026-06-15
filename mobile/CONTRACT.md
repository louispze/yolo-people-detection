# CONTRAT FRONTEND — à respecter par tous les écrans

Lis ces fichiers AVANT d'écrire, ils sont la source de vérité :
`src/types.ts`, `src/store.ts`, `src/lib/engine.ts`, `src/lib/api.ts`,
`src/ui/components.tsx`, `src/ui/icons.tsx`, `src/theme.css`.

## Store (Zustand) — `import { useStore } from "../store"`
S'abonner à une tranche : `const rooms = useStore(s => s.rooms)`.
Récupérer une action : `const addRoom = useStore(s => s.addRoom)`.

État : `screen, status('online'|'connecting'|'offline'), rooms: Room[],
people: Person[], lights: Light[], climate: ClimateConfig, connection: Connection,
demoMode: boolean, live: Record<string,LiveRoomState>, selectedRoomId,
selectedCameraRoomId`.

Actions : `setScreen, selectRoom, selectCameraRoom, addRoom(partial?)->id,
updateRoom(id,patch), removeRoom(id), addPerson(partial?)->id, updatePerson(id,patch),
removePerson(id), setPersonRoom(personId,roomId|null), addLight(partial?)->id,
updateLight(id,patch), toggleLight(id), removeLight(id), setClimate(patch),
bumpBaseTarget(roomId,delta), setConnection(patch), connect(), disconnect(),
setDemoMode(on)`.

## Calcul de la consigne (réactif) — importer le moteur, NE PAS utiliser climateFor pour le rendu
```ts
import { computeClimate } from "../lib/engine";
const people = useStore(s => s.people);
const live = useStore(s => s.live);
const climateCfg = useStore(s => s.climate);
// pour une pièce `room` :
const res = computeClimate(room, people.filter(p => p.presentRoomId === room.id), live[room.id], climateCfg);
// res: { target, base, occupancyPenalty, activityPenalty, workoutPenalty, reasons[] }
```

## Types — `import type { Room, Person, Light, LiveRoomState, ClimateConfig, ClimateResult, Connection } from "../types"`
- Room: id, name, area, icon, cameraSource, currentTemp, baseTarget, minTemp, maxTemp, map{x,y,w,h}
- Person: id, name, color, preferredTemp, presentRoomId|null
- Light: id, name, roomId|null, on, brightness(0..100)
- LiveRoomState: people, active, workout, exercises{}, updatedAt

## UI — `import { Toggle, Slider, Stepper, Sheet, Field, StatusPill, Empty } from "../ui/components"`
- `<Toggle on onChange/>`, `<Slider value min max step? onChange/>`,
  `<Stepper value step suffix? onChange={(delta)=>...}/>`,
  `<Sheet title? onClose>...</Sheet>`, `<Field label>...</Field>`, `<Empty>...</Empty>`
- Icônes : `import { RoomIcon, ROOM_ICON_KEYS } from "../ui/icons"` → `<RoomIcon icon={room.icon} size={20}/>`.
  Autres icônes lucide ré-exportées depuis `../ui/icons` (ex: `import { Plus, Trash2, Thermometer } from "../ui/icons"`).

## Caméra — `import { cameraStreamUrl, cameraSnapshotUrl } from "../lib/api"`
`cameraStreamUrl(connection, room.cameraSource)` → URL MJPEG pour `<img className="cam" src=.../>`.

## Classes CSS disponibles (voir theme.css)
card / card.occupied, section-label, grid (.rooms .cols2), stats/stat/.v/.l,
btn (.primary .ghost .danger .block), icon-btn, room-icon(.on), toggle, slider,
nav, sheet/.field/.input, row(.between)/.spacer, muted/.small, tag(.live .work),
cam/.cam-wrap/.cam-overlay/.rec/.dot, empty, fab.

## Règles
- Composant par défaut exporté : `export default function NomEcran() {...}`.
- TypeScript strict. Pas de dépendance npm nouvelle (uniquement react, zustand, lucide-react).
- Tout en français côté UI.
