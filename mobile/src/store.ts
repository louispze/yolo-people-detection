// ============================================================================
// Store global (Zustand) — état, actions, persistance, live & démo
// ============================================================================

import { create } from "zustand";
import type {
  Room,
  Person,
  Light,
  ClimateConfig,
  Connection,
  ConnStatus,
  LiveRoomState,
  ScreenId,
  ClimateResult,
} from "./types";
import { computeClimate, DEFAULT_CLIMATE } from "./lib/engine";
import { api, connectStateSocket, type StatePayload } from "./lib/api";

const LS_KEY = "p2f-smarthome-config-v1";
const uid = () => Math.random().toString(36).slice(2, 10);

// ---------------------------------------------------------------------------
// Données d'amorçage (maison de démonstration, modifiable dans l'app)
// ---------------------------------------------------------------------------
function seedRooms(): Room[] {
  const mk = (
    id: string, name: string, area: string, icon: string,
    currentTemp: number, baseTarget: number,
    map: { x: number; y: number; w: number; h: number }
  ): Room => ({
    id, name, area, icon, cameraSource: id,
    currentTemp, baseTarget, minTemp: 16, maxTemp: 26, map,
  });
  return [
    // Démo : une seule pièce, le Salon (= la webcam du Raspberry Pi)
    mk("salon", "Salon", "Maison", "sofa", 22.0, 22, { x: 8, y: 8, w: 84, h: 80 }),
  ];
}

function seedPeople(): Person[] {
  return [
    { id: uid(), name: "Alex", color: "#1D9E75", preferredTemp: 21, presentRoomId: "salon" },
    { id: uid(), name: "Camille", color: "#3b82f6", preferredTemp: 22.5, presentRoomId: null },
    { id: uid(), name: "Sacha", color: "#f59e0b", preferredTemp: 20, presentRoomId: null },
  ];
}

function seedLights(): Light[] {
  // Pas de gestion de lumières dans cette démo (non pilotables)
  return [];
}

const emptyLive = (): LiveRoomState => ({
  people: 0, active: 0, workout: 0, exercises: {}, updatedAt: 0,
});

// ---------------------------------------------------------------------------
// Persistance (slice config uniquement)
// ---------------------------------------------------------------------------
interface PersistShape {
  rooms: Room[];
  people: Person[];
  lights: Light[];
  climate: ClimateConfig;
  connection: Connection;
  demoMode: boolean;
}

function loadPersisted(): Partial<PersistShape> {
  try {
    const raw = localStorage.getItem(LS_KEY);
    return raw ? (JSON.parse(raw) as Partial<PersistShape>) : {};
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------------------
// État + actions
// ---------------------------------------------------------------------------
export interface AppState extends PersistShape {
  screen: ScreenId;
  status: ConnStatus;
  live: Record<string, LiveRoomState>;
  selectedRoomId: string | null;
  selectedCameraRoomId: string | null;

  // navigation
  setScreen: (s: ScreenId) => void;
  selectRoom: (id: string | null) => void;
  selectCameraRoom: (id: string | null) => void;

  // pièces
  addRoom: (partial?: Partial<Room>) => string;
  updateRoom: (id: string, patch: Partial<Room>) => void;
  removeRoom: (id: string) => void;

  // personnes
  addPerson: (partial?: Partial<Person>) => string;
  updatePerson: (id: string, patch: Partial<Person>) => void;
  removePerson: (id: string) => void;
  setPersonRoom: (personId: string, roomId: string | null) => void;

  // lumières
  addLight: (partial?: Partial<Light>) => string;
  updateLight: (id: string, patch: Partial<Light>) => void;
  toggleLight: (id: string) => void;
  removeLight: (id: string) => void;

  // climat
  setClimate: (patch: Partial<ClimateConfig>) => void;
  bumpBaseTarget: (roomId: string, delta: number) => void;

  // connexion / live
  setConnection: (patch: Partial<Connection>) => void;
  connect: () => void;
  disconnect: () => void;
  setDemoMode: (on: boolean) => void;

  // sélecteurs
  climateFor: (roomId: string) => ClimateResult;
  presentPeople: (roomId: string) => Person[];
  liveFor: (roomId: string) => LiveRoomState;
}

let stopSocket: (() => void) | null = null;
let demoTimer: ReturnType<typeof setInterval> | null = null;

export const useStore = create<AppState>((set, get) => {
  const persisted = loadPersisted();

  const persist = () => {
    const s = get();
    const data: PersistShape = {
      rooms: s.rooms, people: s.people, lights: s.lights,
      climate: s.climate, connection: s.connection, demoMode: s.demoMode,
    };
    try { localStorage.setItem(LS_KEY, JSON.stringify(data)); } catch { /* ignore */ }
    // push best-effort vers le backend si en ligne
    if (s.status === "online") api.putConfig(s.connection, data).catch(() => {});
  };

  const applyState = (payload: StatePayload) => {
    set((s) => {
      const live = { ...s.live };
      for (const [id, st] of Object.entries(payload.rooms || {})) {
        live[id] = { ...emptyLive(), ...st, updatedAt: Date.now() };
      }
      const rooms = s.rooms.map((r) =>
        payload.temps && payload.temps[r.id] != null
          ? { ...r, currentTemp: payload.temps[r.id] }
          : r
      );
      return { live, rooms };
    });
  };

  const startDemo = () => {
    if (demoTimer) return;
    demoTimer = setInterval(() => {
      set((s) => {
        const live = { ...s.live };
        for (const r of s.rooms) {
          const cur = live[r.id] ?? emptyLive();
          // marche aléatoire douce
          const people = Math.max(0, Math.min(6, cur.people + (Math.random() < 0.5 ? -1 : 1) * (Math.random() < 0.3 ? 1 : 0)));
          const active = Math.min(people, Math.random() < 0.3 ? 1 : 0);
          const workout = Math.min(active, Math.random() < 0.15 ? 1 : 0);
          live[r.id] = {
            people, active, workout,
            exercises: workout ? { squat: workout } : {},
            updatedAt: Date.now(),
          };
        }
        return { live };
      });
    }, 2500);
  };
  const stopDemo = () => { if (demoTimer) { clearInterval(demoTimer); demoTimer = null; } };

  return {
    // état initial (persisté ou seed)
    rooms: persisted.rooms ?? seedRooms(),
    people: persisted.people ?? seedPeople(),
    lights: persisted.lights ?? seedLights(),
    climate: persisted.climate ?? DEFAULT_CLIMATE,
    connection: persisted.connection ?? { host: "10.42.0.1", port: 8000, useTls: false },
    demoMode: persisted.demoMode ?? true,

    screen: "dashboard",
    status: "offline",
    live: {},
    selectedRoomId: null,
    selectedCameraRoomId: null,

    setScreen: (screen) => set({ screen }),
    selectRoom: (selectedRoomId) => set({ selectedRoomId }),
    selectCameraRoom: (selectedCameraRoomId) => set({ selectedCameraRoomId }),

    addRoom: (partial) => {
      const id = partial?.id ?? uid();
      const room: Room = {
        id, name: partial?.name ?? "Nouvelle pièce", area: partial?.area ?? "Maison",
        icon: partial?.icon ?? "home", cameraSource: partial?.cameraSource ?? id,
        currentTemp: partial?.currentTemp ?? 21, baseTarget: partial?.baseTarget ?? 21,
        minTemp: partial?.minTemp ?? 16, maxTemp: partial?.maxTemp ?? 26,
        map: partial?.map ?? { x: 30, y: 30, w: 30, h: 24 },
      };
      set((s) => ({ rooms: [...s.rooms, room] })); persist(); return id;
    },
    updateRoom: (id, patch) => { set((s) => ({ rooms: s.rooms.map((r) => (r.id === id ? { ...r, ...patch } : r)) })); persist(); },
    removeRoom: (id) => { set((s) => ({ rooms: s.rooms.filter((r) => r.id !== id), people: s.people.map((p) => (p.presentRoomId === id ? { ...p, presentRoomId: null } : p)) })); persist(); },

    addPerson: (partial) => {
      const id = uid();
      const colors = ["#1D9E75", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6"];
      const person: Person = {
        id, name: partial?.name ?? "Personne", color: partial?.color ?? colors[Math.floor(Math.random() * colors.length)],
        preferredTemp: partial?.preferredTemp ?? 21, presentRoomId: partial?.presentRoomId ?? null,
      };
      set((s) => ({ people: [...s.people, person] })); persist(); return id;
    },
    updatePerson: (id, patch) => { set((s) => ({ people: s.people.map((p) => (p.id === id ? { ...p, ...patch } : p)) })); persist(); },
    removePerson: (id) => { set((s) => ({ people: s.people.filter((p) => p.id !== id) })); persist(); },
    setPersonRoom: (personId, roomId) => { set((s) => ({ people: s.people.map((p) => (p.id === personId ? { ...p, presentRoomId: roomId } : p)) })); persist(); },

    addLight: (partial) => {
      const id = uid();
      const light: Light = { id, name: partial?.name ?? "Lumière", roomId: partial?.roomId ?? null, on: partial?.on ?? false, brightness: partial?.brightness ?? 100 };
      set((s) => ({ lights: [...s.lights, light] })); persist(); return id;
    },
    updateLight: (id, patch) => { set((s) => ({ lights: s.lights.map((l) => (l.id === id ? { ...l, ...patch } : l)) })); persist(); },
    toggleLight: (id) => { set((s) => ({ lights: s.lights.map((l) => (l.id === id ? { ...l, on: !l.on, brightness: !l.on ? (l.brightness || 100) : l.brightness } : l)) })); persist(); },
    removeLight: (id) => { set((s) => ({ lights: s.lights.filter((l) => l.id !== id) })); persist(); },

    setClimate: (patch) => { set((s) => ({ climate: { ...s.climate, ...patch } })); persist(); },
    bumpBaseTarget: (roomId, delta) => {
      set((s) => ({ rooms: s.rooms.map((r) => (r.id === roomId ? { ...r, baseTarget: Math.round((r.baseTarget + delta) * 2) / 2 } : r)) }));
      persist();
    },

    setConnection: (patch) => { set((s) => ({ connection: { ...s.connection, ...patch } })); persist(); },

    connect: () => {
      stopSocket?.();
      const c = get().connection;
      set({ status: "connecting" });
      stopSocket = connectStateSocket(c, {
        onState: applyState,
        onStatus: (status) => {
          set({ status });
          if (status === "online") { stopDemo(); api.getState(c).then(applyState).catch(() => {}); }
          else if (get().demoMode) startDemo();
        },
      });
    },
    disconnect: () => { stopSocket?.(); stopSocket = null; set({ status: "offline" }); if (get().demoMode) startDemo(); },

    setDemoMode: (on) => {
      set({ demoMode: on }); persist();
      if (on && get().status !== "online") startDemo();
      if (!on) stopDemo();
    },

    climateFor: (roomId) => {
      const s = get();
      const room = s.rooms.find((r) => r.id === roomId)!;
      const present = s.people.filter((p) => p.presentRoomId === roomId);
      return computeClimate(room, present, s.live[roomId], s.climate);
    },
    presentPeople: (roomId) => get().people.filter((p) => p.presentRoomId === roomId),
    liveFor: (roomId) => get().live[roomId] ?? emptyLive(),
  };
});

// Démarre le mode démo au lancement si activé et hors-ligne
if (typeof window !== "undefined") {
  const s = useStore.getState();
  if (s.demoMode) s.setDemoMode(true);
}
