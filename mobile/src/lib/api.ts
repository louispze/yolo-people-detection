// ============================================================================
// Client backend (Raspberry Pi) — REST + WebSocket + flux caméra
// ============================================================================
// Contrat attendu côté serveur Python (backend/server_app.py) :
//   GET  /api/health                      -> { ok: true }
//   GET  /api/config                      -> ConfigPayload (rooms/people/lights/climate)
//   PUT  /api/config       (body=Config)  -> { ok: true }
//   GET  /api/state                       -> { rooms: {id: LiveRoomState}, temps: {id: number} }
//   WS   /ws                              -> messages { type:"state", rooms, temps }
//   GET  /api/stream/{cameraSource}       -> MJPEG (multipart/x-mixed-replace)
// ============================================================================

import type { Connection, LiveRoomState } from "../types";

export interface StatePayload {
  rooms: Record<string, LiveRoomState>;
  temps: Record<string, number>;
}

export function baseUrl(c: Connection): string {
  const proto = c.useTls ? "https" : "http";
  return `${proto}://${c.host}:${c.port}`;
}

export function wsUrl(c: Connection): string {
  const proto = c.useTls ? "wss" : "ws";
  return `${proto}://${c.host}:${c.port}/ws`;
}

/** URL d'un flux caméra MJPEG (utilisable directement dans <img src=...>). */
export function cameraStreamUrl(c: Connection, cameraSource: string): string {
  return `${baseUrl(c)}/api/stream/${encodeURIComponent(cameraSource)}`;
}

/** URL d'un snapshot caméra (image fixe). */
export function cameraSnapshotUrl(c: Connection, cameraSource: string): string {
  return `${baseUrl(c)}/api/snapshot/${encodeURIComponent(cameraSource)}`;
}

async function jsonFetch<T>(url: string, init?: RequestInit, timeoutMs = 4000): Promise<T> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as T;
  } finally {
    clearTimeout(t);
  }
}

export const api = {
  health: (c: Connection) => jsonFetch<{ ok: boolean }>(`${baseUrl(c)}/api/health`),
  getConfig: (c: Connection) => jsonFetch<unknown>(`${baseUrl(c)}/api/config`),
  putConfig: (c: Connection, cfg: unknown) =>
    jsonFetch<{ ok: boolean }>(`${baseUrl(c)}/api/config`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    }),
  getState: (c: Connection) => jsonFetch<StatePayload>(`${baseUrl(c)}/api/state`),
};

/**
 * Connexion WebSocket résiliente : reconnexion auto, callbacks d'état.
 * Renvoie une fonction de fermeture.
 */
export function connectStateSocket(
  c: Connection,
  handlers: {
    onState: (s: StatePayload) => void;
    onStatus: (status: "connecting" | "online" | "offline") => void;
  }
): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  let retry: ReturnType<typeof setTimeout> | null = null;

  const open = () => {
    if (closed) return;
    handlers.onStatus("connecting");
    try {
      ws = new WebSocket(wsUrl(c));
    } catch {
      schedule();
      return;
    }
    ws.onopen = () => handlers.onStatus("online");
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg && msg.type === "state") {
          handlers.onState({ rooms: msg.rooms ?? {}, temps: msg.temps ?? {} });
        }
      } catch {
        /* ignore */
      }
    };
    ws.onclose = () => {
      handlers.onStatus("offline");
      schedule();
    };
    ws.onerror = () => ws?.close();
  };

  const schedule = () => {
    if (closed || retry) return;
    retry = setTimeout(() => {
      retry = null;
      open();
    }, 3000);
  };

  open();

  return () => {
    closed = true;
    if (retry) clearTimeout(retry);
    ws?.close();
  };
}
