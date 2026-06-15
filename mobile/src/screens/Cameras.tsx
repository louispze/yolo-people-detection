// ============================================================================
// Écran Caméras — flux webcam de la Raspberry (caméra de surveillance)
// ============================================================================
// Affiche un flux MJPEG par pièce ayant une cameraSource. Gère proprement le
// cas "flux indisponible" (placeholder informatif + bouton réessayer) et
// permet d'ouvrir une caméra en plein écran avec les infos live détaillées.
// ============================================================================

import { useState } from "react";
import { useStore } from "../store";
import type { Room, LiveRoomState } from "../types";
import { cameraStreamUrl } from "../lib/api";
import { Sheet, Empty } from "../ui/components";
import { RoomIcon } from "../ui/icons";
import { CameraOff, RefreshCw, Activity, Users, Dumbbell } from "../ui/icons";

// ---------------------------------------------------------------------------
// Helpers d'affichage des infos live
// ---------------------------------------------------------------------------

/** Construit la liste des exercices détectés ("squat ×2, pompes ×1"). */
function exercisesLabel(live: LiveRoomState): string {
  const entries = Object.entries(live.exercises).filter(([, n]) => n > 0);
  if (entries.length === 0) return "";
  return entries.map(([name, n]) => `${name} ×${n}`).join(", ");
}

// ---------------------------------------------------------------------------
// Tags "live" (personnes / mouvement / exercice) — réutilisés en carte et en plein écran
// ---------------------------------------------------------------------------
function LiveTags({ live }: { live: LiveRoomState }) {
  const tags: { key: string; cls: string; text: string }[] = [];
  tags.push({
    key: "people",
    cls: live.people > 0 ? "tag live" : "tag",
    text: `${live.people} pers.`,
  });
  if (live.active > 0) tags.push({ key: "active", cls: "tag live", text: `${live.active} en mvt` });
  if (live.workout > 0) tags.push({ key: "work", cls: "tag work", text: `${live.workout} en exercice` });
  return (
    <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
      {tags.map((t) => (
        <span key={t.key} className={t.cls}>
          {t.text}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tuile caméra — image MJPEG avec son propre état d'erreur + rechargement
// ---------------------------------------------------------------------------
function CameraTile({ room, onOpen }: { room: Room; onOpen: () => void }) {
  const connection = useStore((s) => s.connection);
  const live = useStore((s) => s.live[room.id]) ?? {
    people: 0,
    active: 0,
    workout: 0,
    exercises: {},
    updatedAt: 0,
  };

  const [errored, setErrored] = useState(false);
  // Incrémenté pour forcer le rechargement du flux (param ?t=...).
  const [bust, setBust] = useState(0);

  const src = bust === 0
    ? cameraStreamUrl(connection, room.cameraSource)
    : `${cameraStreamUrl(connection, room.cameraSource)}?t=${bust}`;

  const retry = (e: React.MouseEvent) => {
    e.stopPropagation();
    setErrored(false);
    setBust(Date.now());
  };

  return (
    <div className="cam-wrap" onClick={onOpen} style={{ cursor: "pointer" }}>
      {errored ? (
        <CameraPlaceholder onRetry={retry} />
      ) : (
        <img
          className="cam"
          src={src}
          alt={`Flux caméra ${room.name}`}
          onError={() => setErrored(true)}
        />
      )}

      {/* Overlay haut : badge REC + nom de la pièce */}
      <div className="cam-overlay">
        <span className="rec">
          <span className="dot" />
          REC
        </span>
        <span className="rec" style={{ background: "rgba(0,0,0,0.45)" }}>
          {room.name}
        </span>
      </div>

      {/* Bandeau bas : tags live */}
      {!errored && (
        <div
          style={{
            position: "absolute",
            left: 8,
            right: 8,
            bottom: 8,
          }}
        >
          <LiveTags live={live} />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Placeholder "flux indisponible" (mode démo/hors-ligne ou Pi injoignable)
// ---------------------------------------------------------------------------
function CameraPlaceholder({ onRetry }: { onRetry: (e: React.MouseEvent) => void }) {
  return (
    <div
      className="cam"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 10,
        padding: 16,
        textAlign: "center",
        color: "var(--text-dim)",
      }}
    >
      <CameraOff size={30} />
      <div className="small" style={{ maxWidth: 220, lineHeight: 1.4 }}>
        Flux indisponible — connectez le Pi dans Réglages
      </div>
      <button className="btn ghost small" onClick={onRetry} style={{ padding: "7px 12px" }}>
        <RefreshCw size={14} />
        Réessayer
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Vue plein écran d'un seul flux + infos live détaillées
// ---------------------------------------------------------------------------
function CameraDetail({ room, onClose }: { room: Room; onClose: () => void }) {
  const connection = useStore((s) => s.connection);
  const live = useStore((s) => s.live[room.id]) ?? {
    people: 0,
    active: 0,
    workout: 0,
    exercises: {},
    updatedAt: 0,
  };

  const [errored, setErrored] = useState(false);
  const [bust, setBust] = useState(0);

  const src = bust === 0
    ? cameraStreamUrl(connection, room.cameraSource)
    : `${cameraStreamUrl(connection, room.cameraSource)}?t=${bust}`;

  const retry = (e: React.MouseEvent) => {
    e.stopPropagation();
    setErrored(false);
    setBust(Date.now());
  };

  const exo = exercisesLabel(live);

  return (
    <Sheet
      title={room.name}
      onClose={onClose}
    >
      <div className="cam-wrap" style={{ marginBottom: 14 }}>
        {errored ? (
          <CameraPlaceholder onRetry={retry} />
        ) : (
          <img
            className="cam"
            src={src}
            alt={`Flux caméra ${room.name}`}
            style={{ aspectRatio: "16/9" }}
            onError={() => setErrored(true)}
          />
        )}
        <div className="cam-overlay">
          <span className="rec">
            <span className="dot" />
            REC
          </span>
        </div>
      </div>

      {/* Infos live détaillées */}
      <div className="section-label" style={{ margin: "4px 4px 10px" }}>
        Détection en direct
      </div>
      <div className="stats">
        <div className="stat">
          <div className="v" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <Users size={18} />
            {live.people}
          </div>
          <div className="l">Personnes</div>
        </div>
        <div className="stat">
          <div className="v" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <Activity size={18} />
            {live.active}
          </div>
          <div className="l">En mouvement</div>
        </div>
        <div className="stat">
          <div className="v" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <Dumbbell size={18} />
            {live.workout}
          </div>
          <div className="l">En exercice</div>
        </div>
      </div>

      <div className="field">
        <label>Exercices détectés</label>
        <div className="input" style={{ minHeight: 46, display: "flex", alignItems: "center" }}>
          {exo ? exo : <span className="muted">Aucun exercice détecté</span>}
        </div>
      </div>

      <button className="btn block" onClick={(e) => retry(e)}>
        <RefreshCw size={16} />
        Recharger le flux
      </button>
    </Sheet>
  );
}

// ---------------------------------------------------------------------------
// Écran principal
// ---------------------------------------------------------------------------
export default function Cameras() {
  const rooms = useStore((s) => s.rooms);
  const selectedCameraRoomId = useStore((s) => s.selectedCameraRoomId);
  const selectCameraRoom = useStore((s) => s.selectCameraRoom);

  // Pièces équipées d'une caméra.
  const camRooms = rooms.filter((r) => r.cameraSource.trim().length > 0);

  const selected = camRooms.find((r) => r.id === selectedCameraRoomId) ?? null;

  if (camRooms.length === 0) {
    return (
      <Empty>
        Aucune caméra configurée. Ajoutez une source caméra à une pièce dans le plan ou les réglages.
      </Empty>
    );
  }

  return (
    <>
      <div className="section-label">Caméras de surveillance</div>
      <div className="grid" style={{ gridTemplateColumns: "1fr" }}>
        {camRooms.map((room) => (
          <CameraTile
            key={room.id}
            room={room}
            onOpen={() => selectCameraRoom(room.id)}
          />
        ))}
      </div>

      {selected && (
        <CameraDetail room={selected} onClose={() => selectCameraRoom(null)} />
      )}
    </>
  );
}
