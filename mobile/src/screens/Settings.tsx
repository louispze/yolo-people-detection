// ============================================================================
// Écran Réglages — connexion au Raspberry Pi, mode démo, moteur de climat.
// ============================================================================

import { useStore } from "../store";
import type { ClimateConfig } from "../types";
import { baseUrl } from "../lib/api";
import { Toggle, Slider, StatusPill, Field } from "../ui/components";
import { Wifi, WifiOff, Plug, FlaskConical, Thermometer, Trash2 } from "../ui/icons";

const round1 = (v: number) => Math.round(v * 10) / 10;

export default function Settings() {
  // --- état connexion -------------------------------------------------------
  const connection = useStore((s) => s.connection);
  const status = useStore((s) => s.status);
  const setConnection = useStore((s) => s.setConnection);
  const connect = useStore((s) => s.connect);
  const disconnect = useStore((s) => s.disconnect);

  // --- mode démo ------------------------------------------------------------
  const demoMode = useStore((s) => s.demoMode);
  const setDemoMode = useStore((s) => s.setDemoMode);

  // --- moteur de climat -----------------------------------------------------
  const climate = useStore((s) => s.climate);
  const setClimate = useStore((s) => s.setClimate);

  const online = status === "online";
  const url = baseUrl(connection);

  const patchClimate = (patch: Partial<ClimateConfig>) => setClimate(patch);

  const onPortChange = (raw: string) => {
    const n = parseInt(raw, 10);
    setConnection({ port: Number.isFinite(n) ? n : 0 });
  };

  const resetConfig = () => {
    if (!window.confirm("Effacer toute la configuration locale et recharger l'application ?")) return;
    localStorage.removeItem("p2f-smarthome-config-v1");
    location.reload();
  };

  // Texte d'aperçu de la règle de climat active
  const previewParts: string[] = [
    `−${round1(climate.occupancyDelta)}°C toutes les ${climate.occupancyStep} personne${climate.occupancyStep > 1 ? "s" : ""}`,
    `−${round1(climate.activityDelta)}°C par personne active`,
    `−${round1(climate.workoutDelta)}°C par personne en exercice`,
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {/* ================================================================== */}
      {/* 1) Connexion au Raspberry Pi                                       */}
      {/* ================================================================== */}
      <div className="section-label">Connexion au Raspberry Pi</div>
      <div className="card" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <div className="row between">
          <span className="muted small">État de la liaison</span>
          <StatusPill status={status} />
        </div>

        <Field label="Adresse (IP ou nom d'hôte)">
          <input
            className="input"
            type="text"
            inputMode="url"
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            placeholder="192.168.1.50"
            value={connection.host}
            onChange={(e) => setConnection({ host: e.target.value })}
          />
        </Field>

        <Field label="Port">
          <input
            className="input"
            type="number"
            inputMode="numeric"
            min={1}
            max={65535}
            placeholder="8000"
            value={connection.port}
            onChange={(e) => onPortChange(e.target.value)}
          />
        </Field>

        <div className="row between" style={{ margin: "6px 0" }}>
          <div>
            <div style={{ fontWeight: 600 }}>TLS (HTTPS / WSS)</div>
            <div className="muted small">Chiffrer la connexion au backend</div>
          </div>
          <Toggle on={connection.useTls} onChange={(v) => setConnection({ useTls: v })} />
        </div>

        <button
          className={`btn block ${online ? "danger" : "primary"}`}
          onClick={() => (online ? disconnect() : connect())}
        >
          {online ? <WifiOff size={18} /> : <Plug size={18} />}
          {online ? "Se déconnecter" : "Se connecter"}
        </button>

        <div className="row" style={{ gap: 8, marginTop: 8 }}>
          <Wifi size={14} className="muted" />
          <span className="muted small" style={{ wordBreak: "break-all" }}>{url}</span>
        </div>
      </div>

      {/* ================================================================== */}
      {/* 2) Mode démo                                                       */}
      {/* ================================================================== */}
      <div className="section-label">Mode démo</div>
      <div className="card">
        <div className="row between">
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span className="room-icon on"><FlaskConical size={20} /></span>
            <div>
              <div style={{ fontWeight: 600 }}>Données simulées</div>
              <div className="muted small" style={{ maxWidth: 260 }}>
                Simule des données d'occupation et d'activité quand aucun Raspberry Pi n'est connecté.
              </div>
            </div>
          </div>
          <Toggle on={demoMode} onChange={(v) => setDemoMode(v)} />
        </div>
      </div>

      {/* ================================================================== */}
      {/* 3) Moteur de climat adaptatif                                      */}
      {/* ================================================================== */}
      <div className="section-label">Moteur de climat adaptatif</div>
      <div className="card" style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        {/* Palier d'occupation */}
        <div>
          <div className="row between">
            <label className="muted small">Palier d'occupation (personnes)</label>
            <span className="tag">{climate.occupancyStep} pers.</span>
          </div>
          <Slider
            value={climate.occupancyStep}
            min={1}
            max={8}
            step={1}
            onChange={(v) => patchClimate({ occupancyStep: Math.round(v) })}
          />
          <div className="muted small" style={{ marginTop: 4 }}>
            Nombre de personnes par palier avant d'abaisser la consigne.
          </div>
        </div>

        {/* Delta d'occupation */}
        <div>
          <div className="row between">
            <label className="muted small">Baisse par palier d'occupation</label>
            <span className="tag">−{round1(climate.occupancyDelta)}°C</span>
          </div>
          <Slider
            value={climate.occupancyDelta}
            min={0}
            max={3}
            step={0.5}
            onChange={(v) => patchClimate({ occupancyDelta: v })}
          />
        </div>

        {/* Delta activité */}
        <div>
          <div className="row between">
            <label className="muted small">Baisse par personne en mouvement</label>
            <span className="tag">−{round1(climate.activityDelta)}°C</span>
          </div>
          <Slider
            value={climate.activityDelta}
            min={0}
            max={2}
            step={0.25}
            onChange={(v) => patchClimate({ activityDelta: v })}
          />
        </div>

        {/* Delta exercice */}
        <div>
          <div className="row between">
            <label className="muted small">Baisse par personne en exercice</label>
            <span className="tag">−{round1(climate.workoutDelta)}°C</span>
          </div>
          <Slider
            value={climate.workoutDelta}
            min={0}
            max={3}
            step={0.25}
            onChange={(v) => patchClimate({ workoutDelta: v })}
          />
        </div>

        {/* Mode de combinaison */}
        <div>
          <label className="muted small" style={{ display: "block", marginBottom: 8 }}>
            Combinaison des préférences des présents
          </label>
          <div className="row" style={{ gap: 10 }}>
            <button
              className={`btn block ${climate.blendMode === "average" ? "primary" : "ghost"}`}
              onClick={() => patchClimate({ blendMode: "average" })}
            >
              Moyenne
            </button>
            <button
              className={`btn block ${climate.blendMode === "coolest" ? "primary" : "ghost"}`}
              onClick={() => patchClimate({ blendMode: "coolest" })}
            >
              Plus fraîche
            </button>
          </div>
        </div>

        {/* Aperçu de la règle active */}
        <div
          style={{
            background: "var(--inner)",
            border: "1px solid var(--border)",
            borderRadius: "var(--r-md)",
            padding: 14,
          }}
        >
          <div className="row" style={{ gap: 8, marginBottom: 8 }}>
            <Thermometer size={16} className="accent" />
            <span style={{ fontWeight: 700, fontSize: 13 }}>Règle actuelle</span>
          </div>
          <div className="muted small" style={{ lineHeight: 1.6 }}>
            {previewParts.join(", ")}.
          </div>
          <div className="muted small" style={{ marginTop: 8, lineHeight: 1.6 }}>
            Base de consigne :{" "}
            {climate.blendMode === "coolest"
              ? "préférence la plus fraîche des personnes présentes."
              : "moyenne des préférences des personnes présentes."}
          </div>
        </div>
      </div>

      {/* ================================================================== */}
      {/* 4) À propos / réinitialiser                                        */}
      {/* ================================================================== */}
      <div className="section-label">À propos / réinitialiser</div>
      <div className="card" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div className="muted small" style={{ lineHeight: 1.6 }}>
          P2F · Domotique adaptative. La configuration (pièces, personnes, lumières,
          réglages de climat et connexion) est enregistrée localement sur cet appareil.
        </div>
        <button className="btn danger block" onClick={resetConfig}>
          <Trash2 size={18} />
          Effacer la configuration
        </button>
      </div>
    </div>
  );
}
