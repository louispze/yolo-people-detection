// ============================================================================
// Écran « Personnes » — préférences de température & présence
// ============================================================================
// Objectif : éditer la température préférée de chaque personne (le moteur de
// climat s'adapte), et assigner une personne à une pièce (présence).
// ============================================================================

import { useMemo, useState } from "react";
import { useStore } from "../store";
import { computeClimate } from "../lib/engine";
import type { Person } from "../types";
import { Slider, Stepper, Sheet, Field, Empty } from "../ui/components";
import { Plus, Trash2, Thermometer, MapPin, RoomIcon } from "../ui/icons";

// Palette de couleurs d'avatar (alignée sur le thème).
const COLORS = [
  "#1D9E75",
  "#3b82f6",
  "#f59e0b",
  "#ef4444",
  "#a855f7",
  "#14b8a6",
  "#ec4899",
  "#eab308",
];

const round1 = (v: number) => Math.round(v * 10) / 10;
const initial = (name: string) => (name.trim()[0] ?? "?").toUpperCase();

function Avatar({ person, size = 44 }: { person: Person; size?: number }) {
  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        background: person.color,
        color: "#fff",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        fontWeight: 700,
        fontSize: size * 0.42,
        flex: "none",
        boxShadow: `0 0 0 3px ${person.color}26`,
      }}
    >
      {initial(person.name)}
    </div>
  );
}

export default function People() {
  const people = useStore((s) => s.people);
  const rooms = useStore((s) => s.rooms);
  const live = useStore((s) => s.live);
  const climateCfg = useStore((s) => s.climate);

  const addPerson = useStore((s) => s.addPerson);
  const updatePerson = useStore((s) => s.updatePerson);
  const removePerson = useStore((s) => s.removePerson);
  const setPersonRoom = useStore((s) => s.setPersonRoom);

  const [editingId, setEditingId] = useState<string | null>(null);

  const editing = useMemo(
    () => people.find((p) => p.id === editingId) ?? null,
    [people, editingId]
  );

  const roomName = (id: string | null) =>
    id ? rooms.find((r) => r.id === id)?.name ?? "Pièce inconnue" : null;

  const handleAdd = () => {
    const id = addPerson();
    setEditingId(id);
  };

  // Impact de la préférence de la personne éditée sur la pièce où elle est présente.
  const impact = useMemo(() => {
    if (!editing || !editing.presentRoomId) return null;
    const room = rooms.find((r) => r.id === editing.presentRoomId);
    if (!room) return null;
    const present = people.filter((p) => p.presentRoomId === room.id);
    return { room, res: computeClimate(room, present, live[room.id], climateCfg) };
  }, [editing, rooms, people, live, climateCfg]);

  return (
    <>
      <div className="section-label">
        {people.length} personne{people.length > 1 ? "s" : ""}
      </div>

      {people.length === 0 ? (
        <Empty>
          Aucune personne pour l'instant.
          <br />
          Ajoutez les membres du foyer pour adapter la température à leurs
          préférences.
        </Empty>
      ) : (
        <div className="grid" style={{ gap: 12 }}>
          {people.map((p) => {
            const present = p.presentRoomId != null;
            return (
              <button
                key={p.id}
                className={`card ${present ? "occupied" : ""}`}
                onClick={() => setEditingId(p.id)}
                style={{
                  textAlign: "left",
                  width: "100%",
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                }}
              >
                <Avatar person={p} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 700, fontSize: 16 }}>{p.name}</div>
                  <div
                    className="row small muted"
                    style={{ gap: 12, marginTop: 4, flexWrap: "wrap" }}
                  >
                    <span className="row" style={{ gap: 4 }}>
                      <Thermometer size={13} />
                      {round1(p.preferredTemp)}°C
                    </span>
                    <span className="row" style={{ gap: 4 }}>
                      <MapPin size={13} />
                      {present ? roomName(p.presentRoomId) : "Absent"}
                    </span>
                  </div>
                </div>
                {present && <span className="tag live">Présent</span>}
              </button>
            );
          })}
        </div>
      )}

      <button className="fab" onClick={handleAdd} aria-label="Ajouter une personne">
        <Plus size={24} color="#fff" />
      </button>

      {editing && (
        <Sheet title="Modifier la personne" onClose={() => setEditingId(null)}>
          <div
            className="row"
            style={{ gap: 14, marginBottom: 6, alignItems: "center" }}
          >
            <Avatar person={editing} size={52} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 700, fontSize: 17 }}>{editing.name}</div>
              <div className="small muted">
                Préférence {round1(editing.preferredTemp)}°C ·{" "}
                {editing.presentRoomId
                  ? roomName(editing.presentRoomId)
                  : "Absent"}
              </div>
            </div>
          </div>

          {/* Nom */}
          <Field label="Nom">
            <input
              className="input"
              type="text"
              value={editing.name}
              onChange={(e) =>
                updatePerson(editing.id, { name: e.target.value })
              }
              placeholder="Prénom"
            />
          </Field>

          {/* Température préférée */}
          <Field label="Température préférée">
            <div
              className="row between"
              style={{ gap: 12, marginBottom: 10 }}
            >
              <span className="row" style={{ gap: 6, fontWeight: 700, fontSize: 18 }}>
                <Thermometer size={18} />
                {round1(editing.preferredTemp)}°C
              </span>
              <Stepper
                value={editing.preferredTemp}
                step={0.5}
                suffix="°C"
                onChange={(delta) =>
                  updatePerson(editing.id, {
                    preferredTemp: round1(
                      Math.max(16, Math.min(26, editing.preferredTemp + delta))
                    ),
                  })
                }
              />
            </div>
            <Slider
              value={editing.preferredTemp}
              min={16}
              max={26}
              step={0.5}
              onChange={(v) =>
                updatePerson(editing.id, { preferredTemp: round1(v) })
              }
            />
            <div
              className="row between small muted"
              style={{ marginTop: 6 }}
            >
              <span>16°C</span>
              <span>26°C</span>
            </div>
          </Field>

          {/* Couleur d'avatar */}
          <Field label="Couleur">
            <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
              {COLORS.map((c) => {
                const active = editing.color.toLowerCase() === c.toLowerCase();
                return (
                  <button
                    key={c}
                    onClick={() => updatePerson(editing.id, { color: c })}
                    aria-label={`Couleur ${c}`}
                    style={{
                      width: 34,
                      height: 34,
                      borderRadius: "50%",
                      background: c,
                      border: active
                        ? "3px solid #fff"
                        : "3px solid transparent",
                      boxShadow: active ? `0 0 0 2px ${c}` : "none",
                    }}
                  />
                );
              })}
            </div>
          </Field>

          {/* Présence (pièce assignée) */}
          <Field label="Présence">
            <div className="grid" style={{ gap: 8 }}>
              <PresenceOption
                active={editing.presentRoomId === null}
                onClick={() => setPersonRoom(editing.id, null)}
                label="Absent"
                sub="Ne participe à aucune consigne"
                icon={<MapPin size={18} />}
              />
              {rooms.map((r) => (
                <PresenceOption
                  key={r.id}
                  active={editing.presentRoomId === r.id}
                  onClick={() => setPersonRoom(editing.id, r.id)}
                  label={r.name}
                  sub={r.area}
                  icon={<RoomIcon icon={r.icon} size={18} />}
                />
              ))}
            </div>
          </Field>

          {/* Explication de l'impact */}
          <div
            className="card"
            style={{
              padding: 14,
              background: "var(--inner)",
              marginTop: 4,
            }}
          >
            {impact ? (
              <>
                <div className="small" style={{ fontWeight: 600 }}>
                  Impact sur {impact.room.name}
                </div>
                <div className="small muted" style={{ marginTop: 4 }}>
                  La préférence de {editing.name} sert de base à la consigne de
                  cette pièce. Consigne actuelle :{" "}
                  <strong style={{ color: "var(--accent)" }}>
                    {round1(impact.res.target)}°C
                  </strong>{" "}
                  (base {round1(impact.res.base)}°C).
                </div>
                {impact.res.reasons.length > 0 && (
                  <ul
                    className="small muted"
                    style={{
                      margin: "8px 0 0",
                      paddingLeft: 16,
                      lineHeight: 1.6,
                    }}
                  >
                    {impact.res.reasons.map((reason, i) => (
                      <li key={i}>{reason}</li>
                    ))}
                  </ul>
                )}
              </>
            ) : (
              <div className="small muted">
                {editing.name} est absent : sa préférence n'influence aucune
                pièce. Assignez-le à une pièce pour adapter la consigne.
              </div>
            )}
          </div>

          {/* Suppression */}
          <button
            className="btn danger block"
            style={{ marginTop: 16 }}
            onClick={() => {
              removePerson(editing.id);
              setEditingId(null);
            }}
          >
            <Trash2 size={16} />
            Supprimer {editing.name}
          </button>
        </Sheet>
      )}
    </>
  );
}

function PresenceOption({
  active,
  onClick,
  label,
  sub,
  icon,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  sub: string;
  icon: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className="row"
      style={{
        gap: 12,
        width: "100%",
        textAlign: "left",
        padding: "10px 12px",
        borderRadius: "var(--r-md)",
        background: active ? "var(--accent-soft)" : "var(--inner)",
        border: `1px solid ${active ? "var(--accent)" : "var(--border)"}`,
      }}
    >
      <span
        className={`room-icon ${active ? "on" : ""}`}
        style={{ width: 34, height: 34 }}
      >
        {icon}
      </span>
      <span style={{ flex: 1, minWidth: 0 }}>
        <span
          style={{
            display: "block",
            fontWeight: 600,
            color: active ? "var(--accent)" : "var(--text)",
          }}
        >
          {label}
        </span>
        <span className="small muted" style={{ display: "block" }}>
          {sub}
        </span>
      </span>
      {active && <span className="tag live">Sélectionné</span>}
    </button>
  );
}
