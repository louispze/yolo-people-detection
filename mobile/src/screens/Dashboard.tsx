// ============================================================================
// Écran d'accueil — tableau de bord (stats, pièces, lumières)
// ============================================================================

import { useState } from "react";
import { useStore } from "../store";
import { computeClimate } from "../lib/engine";
import type { Room, Person, Light, LiveRoomState, ClimateResult, ClimateConfig } from "../types";
import { Toggle, Slider, Stepper, Sheet, Empty } from "../ui/components";
import { RoomIcon, Thermometer, Users, Activity, Lightbulb } from "../ui/icons";

const round1 = (v: number) => Math.round(v * 10) / 10;

const EMPTY_LIVE: LiveRoomState = {
  people: 0, active: 0, workout: 0, exercises: {}, updatedAt: 0,
};

/** Met en forme les exercices live en libellés ("squat x1"). */
function exerciseTags(live: LiveRoomState): string[] {
  return Object.entries(live.exercises)
    .filter(([, n]) => n > 0)
    .map(([name, n]) => `🏋 ${name} x${n}`);
}

// ----------------------------------------------------------------------------
// Carte d'une pièce
// ----------------------------------------------------------------------------
function RoomCard({
  room, present, live, climateCfg, online, onOpen,
}: {
  room: Room;
  present: Person[];
  live: LiveRoomState;
  climateCfg: ClimateConfig;
  online: boolean;
  onOpen: () => void;
}) {
  const bumpBaseTarget = useStore((s) => s.bumpBaseTarget);
  const res: ClimateResult = computeClimate(room, present, live, climateCfg);

  const occupied = live.people > 0;
  const penalty = round1(res.base - res.target);
  const works = exerciseTags(live);

  return (
    <div
      className={`card${occupied ? " occupied" : ""}`}
      onClick={onOpen}
      style={{ display: "flex", flexDirection: "column", gap: 12, cursor: "pointer" }}
    >
      <div className="row between">
        <span className={`room-icon${occupied ? " on" : ""}`}>
          <RoomIcon icon={room.icon} size={20} />
        </span>
        {occupied && online && <span className="tag live">LIVE</span>}
      </div>

      <div>
        <div style={{ fontWeight: 700, fontSize: 15 }}>{room.name}</div>
        <div className="muted small">{room.area}</div>
      </div>

      <div className="row" style={{ alignItems: "baseline", gap: 4 }}>
        <span style={{ fontSize: 30, fontWeight: 700, lineHeight: 1 }}>
          {round1(room.currentTemp)}
        </span>
        <span className="muted" style={{ fontSize: 15, fontWeight: 600 }}>°C</span>
      </div>

      <div className="row small muted" style={{ gap: 6 }}>
        <Users size={14} />
        <span>{live.people} détectée{live.people > 1 ? "s" : ""}</span>
      </div>

      {works.length > 0 && (
        <div className="row" style={{ flexWrap: "wrap", gap: 6 }}>
          {works.map((w, i) => (
            <span key={i} className="tag work">{w}</span>
          ))}
        </div>
      )}

      {/* Consigne calculée + réglage de la base */}
      <div
        className="row between"
        style={{
          marginTop: "auto",
          paddingTop: 12,
          borderTop: "1px solid var(--border)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <div className="row" style={{ gap: 5, color: "var(--accent)", fontWeight: 700 }}>
            <Thermometer size={15} />
            <span style={{ fontSize: 17 }}>{res.target.toFixed(1)}°C</span>
          </div>
          {penalty > 0 && (
            <div className="muted small" style={{ marginTop: 2 }}>
              base {res.base.toFixed(1)}° −{penalty.toFixed(1)}°
            </div>
          )}
        </div>
        <Stepper
          value={room.baseTarget}
          step={0.5}
          suffix="°"
          onChange={(delta) => bumpBaseTarget(room.id, delta)}
        />
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------
// Détail d'une pièce (sheet)
// ----------------------------------------------------------------------------
function RoomSheet({
  room, present, live, climateCfg, onClose,
}: {
  room: Room;
  present: Person[];
  live: LiveRoomState;
  climateCfg: ClimateConfig;
  onClose: () => void;
}) {
  const res = computeClimate(room, present, live, climateCfg);
  const works = exerciseTags(live);

  return (
    <Sheet title={room.name} onClose={onClose}>
      {/* Récapitulatif climat */}
      <div className="row between" style={{ marginBottom: 8 }}>
        <div className="muted small">Mesurée</div>
        <div style={{ fontWeight: 700 }}>{round1(room.currentTemp).toFixed(1)}°C</div>
      </div>
      <div className="row between" style={{ marginBottom: 14 }}>
        <div className="muted small">Consigne</div>
        <div style={{ fontWeight: 700, color: "var(--accent)" }}>
          {res.target.toFixed(1)}°C
        </div>
      </div>

      <div className="section-label" style={{ margin: "4px 0 8px" }}>
        Détail du calcul
      </div>
      <div className="card" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 6 }}>
        {res.reasons.map((r, i) => (
          <div key={i} className="row small" style={{ gap: 8 }}>
            <Activity size={14} className="muted" />
            <span>{r}</span>
          </div>
        ))}
      </div>

      <div className="section-label">Personnes présentes</div>
      {present.length === 0 ? (
        <Empty>Aucune personne marquée présente.</Empty>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {present.map((p) => (
            <div key={p.id} className="card row between" style={{ padding: 12 }}>
              <div className="row" style={{ gap: 10 }}>
                <span
                  style={{
                    width: 28, height: 28, borderRadius: "50%",
                    background: p.color, display: "inline-flex",
                    alignItems: "center", justifyContent: "center",
                    fontSize: 12, fontWeight: 700, color: "#fff",
                  }}
                >
                  {p.name.slice(0, 1).toUpperCase()}
                </span>
                <span style={{ fontWeight: 600 }}>{p.name}</span>
              </div>
              <span className="muted small">préf. {round1(p.preferredTemp)}°C</span>
            </div>
          ))}
        </div>
      )}

      <div className="section-label">Activité détectée</div>
      {live.people === 0 && works.length === 0 ? (
        <Empty>Aucune activité en cours.</Empty>
      ) : (
        <div className="card" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 8 }}>
          <div className="row between small">
            <span className="muted">Personnes détectées</span>
            <span style={{ fontWeight: 600 }}>{live.people}</span>
          </div>
          <div className="row between small">
            <span className="muted">En mouvement</span>
            <span style={{ fontWeight: 600 }}>{live.active}</span>
          </div>
          <div className="row between small">
            <span className="muted">En exercice</span>
            <span style={{ fontWeight: 600 }}>{live.workout}</span>
          </div>
          {works.length > 0 && (
            <div className="row" style={{ flexWrap: "wrap", gap: 6, marginTop: 2 }}>
              {works.map((w, i) => (
                <span key={i} className="tag work">{w}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </Sheet>
  );
}

// ----------------------------------------------------------------------------
// Élément lumière
// ----------------------------------------------------------------------------
function LightRow({ light }: { light: Light }) {
  const toggleLight = useStore((s) => s.toggleLight);
  const updateLight = useStore((s) => s.updateLight);

  return (
    <div className="card" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
      <div className="row between">
        <div className="row" style={{ gap: 12 }}>
          <span className={`room-icon${light.on ? " on" : ""}`}>
            <Lightbulb size={18} />
          </span>
          <div>
            <div style={{ fontWeight: 600 }}>{light.name}</div>
            <div className="muted small">
              {light.on ? `Allumée · ${Math.round(light.brightness)}%` : "Éteinte"}
            </div>
          </div>
        </div>
        <Toggle on={light.on} onChange={() => toggleLight(light.id)} />
      </div>

      {light.on && (
        <Slider
          value={light.brightness}
          min={0}
          max={100}
          step={1}
          onChange={(v) => updateLight(light.id, { brightness: v })}
        />
      )}
    </div>
  );
}

// ----------------------------------------------------------------------------
// Écran principal
// ----------------------------------------------------------------------------
export default function Dashboard() {
  const rooms = useStore((s) => s.rooms);
  const people = useStore((s) => s.people);
  const lights = useStore((s) => s.lights);
  const live = useStore((s) => s.live);
  const climateCfg = useStore((s) => s.climate);
  const status = useStore((s) => s.status);

  const [openRoomId, setOpenRoomId] = useState<string | null>(null);

  const online = status === "online";
  const liveOf = (id: string): LiveRoomState => live[id] ?? EMPTY_LIVE;
  const presentOf = (id: string): Person[] => people.filter((p) => p.presentRoomId === id);

  // Statistiques agrégées (réactives)
  const totalPeople = rooms.reduce((sum, r) => sum + liveOf(r.id).people, 0);
  const avgTemp = rooms.length
    ? round1(rooms.reduce((sum, r) => sum + r.currentTemp, 0) / rooms.length)
    : 0;
  const occupiedCount = rooms.filter((r) => liveOf(r.id).people > 0).length;

  const openRoom = openRoomId ? rooms.find((r) => r.id === openRoomId) ?? null : null;

  return (
    <div>
      {/* 1) Stats */}
      <div className="stats">
        <div className="stat">
          <div className="v">{totalPeople}</div>
          <div className="l">Personnes</div>
        </div>
        <div className="stat">
          <div className="v">{rooms.length ? `${avgTemp.toFixed(1)}°` : "—"}</div>
          <div className="l">Temp. moy.</div>
        </div>
        <div className="stat">
          <div className="v">{occupiedCount}</div>
          <div className="l">Occupées</div>
        </div>
      </div>

      {/* 2) Pièces */}
      <div className="section-label">Pièces</div>
      {rooms.length === 0 ? (
        <Empty>Aucune pièce configurée.</Empty>
      ) : (
        <div className="grid rooms">
          {rooms.map((room) => (
            <RoomCard
              key={room.id}
              room={room}
              present={presentOf(room.id)}
              live={liveOf(room.id)}
              climateCfg={climateCfg}
              online={online}
              onOpen={() => setOpenRoomId(room.id)}
            />
          ))}
        </div>
      )}

      {/* 4) Lumières */}
      <div className="section-label">Lumières</div>
      {lights.length === 0 ? (
        <Empty>Aucune lumière configurée.</Empty>
      ) : (
        <div className="grid" style={{ gap: 12 }}>
          {lights.map((light) => (
            <LightRow key={light.id} light={light} />
          ))}
        </div>
      )}

      {/* 3) Détail pièce */}
      {openRoom && (
        <RoomSheet
          room={openRoom}
          present={presentOf(openRoom.id)}
          live={liveOf(openRoom.id)}
          climateCfg={climateCfg}
          onClose={() => setOpenRoomId(null)}
        />
      )}
    </div>
  );
}
