// ============================================================================
// Plan éditable de la maison — mapper, renommer et placer chaque pièce
// ============================================================================
import { useMemo } from "react";
import { useStore } from "../store";
import type { Room } from "../types";
import { Sheet, Field, Slider, Stepper, Empty } from "../ui/components";
import {
  RoomIcon,
  ROOM_ICON_KEYS,
  Plus,
  Trash2,
  Thermometer,
  Move,
  Maximize2,
} from "../ui/icons";

const clampPct = (v: number) => Math.max(0, Math.min(100, Math.round(v)));

export default function HouseMap() {
  const rooms = useStore((s) => s.rooms);
  const live = useStore((s) => s.live);
  const selectedRoomId = useStore((s) => s.selectedRoomId);

  const selectRoom = useStore((s) => s.selectRoom);
  const addRoom = useStore((s) => s.addRoom);
  const updateRoom = useStore((s) => s.updateRoom);
  const removeRoom = useStore((s) => s.removeRoom);

  const selected = useMemo(
    () => rooms.find((r) => r.id === selectedRoomId) ?? null,
    [rooms, selectedRoomId]
  );

  const handleAdd = () => {
    const id = addRoom();
    selectRoom(id);
  };

  return (
    <div>
      <div className="section-label">Plan de la maison</div>

      {rooms.length === 0 ? (
        <Empty>
          Aucune pièce pour le moment.
          <br />
          Touchez le bouton + pour en ajouter une.
        </Empty>
      ) : (
        <div
          className="card"
          style={{
            position: "relative",
            aspectRatio: "3 / 4",
            padding: 0,
            overflow: "hidden",
            backgroundImage:
              "radial-gradient(circle at 1px 1px, rgba(255,255,255,0.05) 1px, transparent 0)",
            backgroundSize: "20px 20px",
          }}
        >
          {rooms.map((r) => {
            const occupied = (live[r.id]?.people ?? 0) > 0;
            const isSel = r.id === selectedRoomId;
            return (
              <button
                key={r.id}
                onClick={() => selectRoom(r.id)}
                style={{
                  position: "absolute",
                  left: `${r.map.x}%`,
                  top: `${r.map.y}%`,
                  width: `${r.map.w}%`,
                  height: `${r.map.h}%`,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 4,
                  padding: 6,
                  textAlign: "center",
                  overflow: "hidden",
                  borderRadius: 12,
                  background: occupied
                    ? "rgba(29, 158, 117, 0.16)"
                    : "rgba(37, 37, 37, 0.7)",
                  border: `1.5px solid ${
                    isSel
                      ? "var(--accent)"
                      : occupied
                      ? "rgba(29, 158, 117, 0.55)"
                      : "var(--border)"
                  }`,
                  boxShadow: isSel
                    ? "0 0 0 2px var(--accent-soft)"
                    : "none",
                  color: "var(--text)",
                  transition: "0.15s",
                }}
              >
                <span
                  className={`room-icon ${occupied ? "on" : ""}`}
                  style={{ width: 34, height: 34, flex: "none" }}
                >
                  <RoomIcon icon={r.icon} size={18} />
                </span>
                <span
                  style={{
                    fontSize: 12,
                    fontWeight: 700,
                    lineHeight: 1.1,
                    maxWidth: "100%",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {r.name}
                </span>
                {occupied && (
                  <span className="tag live">
                    {live[r.id]?.people}{" "}
                    {(live[r.id]?.people ?? 0) > 1 ? "personnes" : "personne"}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}

      <div className="muted small" style={{ margin: "14px 4px 0" }}>
        Touchez une pièce pour la renommer et ajuster sa position sur le plan.
      </div>

      <button
        className="fab"
        onClick={handleAdd}
        aria-label="Ajouter une pièce"
      >
        <Plus size={24} color="#fff" />
      </button>

      {selected && (
        <RoomEditor
          key={selected.id}
          room={selected}
          onClose={() => selectRoom(null)}
          onUpdate={(patch) => updateRoom(selected.id, patch)}
          onUpdateMap={(patch) =>
            updateRoom(selected.id, { map: { ...selected.map, ...patch } })
          }
          onRemove={() => {
            removeRoom(selected.id);
            selectRoom(null);
          }}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Feuille d'édition d'une pièce
// ---------------------------------------------------------------------------
function RoomEditor({
  room,
  onClose,
  onUpdate,
  onUpdateMap,
  onRemove,
}: {
  room: Room;
  onClose: () => void;
  onUpdate: (patch: Partial<Room>) => void;
  onUpdateMap: (patch: Partial<Room["map"]>) => void;
  onRemove: () => void;
}) {
  return (
    <Sheet title="Éditer la pièce" onClose={onClose}>
      <Field label="Nom">
        <input
          className="input"
          value={room.name}
          placeholder="Nom de la pièce"
          onChange={(e) => onUpdate({ name: e.target.value })}
        />
      </Field>

      <Field label="Zone / étage">
        <input
          className="input"
          value={room.area}
          placeholder="Ex : Rez-de-chaussée, Étage…"
          onChange={(e) => onUpdate({ area: e.target.value })}
        />
      </Field>

      <Field label="Icône">
        <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
          {ROOM_ICON_KEYS.map((key) => {
            const active = key === room.icon;
            return (
              <button
                key={key}
                onClick={() => onUpdate({ icon: key })}
                className={`room-icon ${active ? "on" : ""}`}
                style={{
                  width: 46,
                  height: 46,
                  border: `1px solid ${
                    active ? "var(--accent)" : "var(--border)"
                  }`,
                }}
                aria-label={`Icône ${key}`}
                aria-pressed={active}
              >
                <RoomIcon icon={key} size={22} />
              </button>
            );
          })}
        </div>
      </Field>

      <Field label="Source caméra">
        <input
          className="input"
          value={room.cameraSource}
          placeholder="Identifiant du flux (ex : salon)"
          onChange={(e) => onUpdate({ cameraSource: e.target.value })}
        />
      </Field>

      <div className="section-label" style={{ margin: "18px 4px 6px" }}>
        Bornes de température
      </div>
      <div className="grid cols2">
        <div className="card" style={{ padding: 14 }}>
          <div
            className="row"
            style={{ gap: 6, marginBottom: 10, color: "var(--blue)" }}
          >
            <Thermometer size={16} />
            <span className="small" style={{ fontWeight: 700 }}>
              Min
            </span>
          </div>
          <Stepper
            value={room.minTemp}
            step={0.5}
            suffix="°C"
            onChange={(delta) =>
              onUpdate({
                minTemp: Math.min(
                  room.maxTemp - 0.5,
                  Math.round((room.minTemp + delta) * 2) / 2
                ),
              })
            }
          />
        </div>
        <div className="card" style={{ padding: 14 }}>
          <div
            className="row"
            style={{ gap: 6, marginBottom: 10, color: "var(--red)" }}
          >
            <Thermometer size={16} />
            <span className="small" style={{ fontWeight: 700 }}>
              Max
            </span>
          </div>
          <Stepper
            value={room.maxTemp}
            step={0.5}
            suffix="°C"
            onChange={(delta) =>
              onUpdate({
                maxTemp: Math.max(
                  room.minTemp + 0.5,
                  Math.round((room.maxTemp + delta) * 2) / 2
                ),
              })
            }
          />
        </div>
      </div>

      <div className="section-label" style={{ margin: "18px 4px 6px" }}>
        Position sur le plan
      </div>

      <MapSlider
        icon={<Move size={15} />}
        label="Position horizontale"
        value={room.map.x}
        onChange={(v) => onUpdateMap({ x: clampPct(v) })}
      />
      <MapSlider
        icon={<Move size={15} />}
        label="Position verticale"
        value={room.map.y}
        onChange={(v) => onUpdateMap({ y: clampPct(v) })}
      />
      <MapSlider
        icon={<Maximize2 size={15} />}
        label="Largeur"
        value={room.map.w}
        min={5}
        onChange={(v) => onUpdateMap({ w: clampPct(v) })}
      />
      <MapSlider
        icon={<Maximize2 size={15} />}
        label="Hauteur"
        value={room.map.h}
        min={5}
        onChange={(v) => onUpdateMap({ h: clampPct(v) })}
      />

      <button
        className="btn danger block"
        style={{ marginTop: 22 }}
        onClick={onRemove}
      >
        <Trash2 size={18} />
        Supprimer la pièce
      </button>
    </Sheet>
  );
}

// ---------------------------------------------------------------------------
// Curseur de position/taille (en %)
// ---------------------------------------------------------------------------
function MapSlider({
  icon,
  label,
  value,
  min = 0,
  onChange,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
  min?: number;
  onChange: (v: number) => void;
}) {
  return (
    <Field label={label}>
      <div className="row" style={{ gap: 12 }}>
        <span className="muted" style={{ display: "inline-flex" }}>
          {icon}
        </span>
        <Slider value={value} min={min} max={100} step={1} onChange={onChange} />
        <span
          className="small"
          style={{
            minWidth: 42,
            textAlign: "right",
            fontWeight: 700,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {Math.round(value)}%
        </span>
      </div>
    </Field>
  );
}
