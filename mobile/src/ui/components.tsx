// Composants UI réutilisables (design-system).
import { useEffect, type ReactNode } from "react";
import { X } from "lucide-react";

export function Toggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      className={`toggle ${on ? "on" : "off"}`}
      onClick={() => onChange(!on)}
      aria-pressed={on}
    />
  );
}

export function Slider({
  value, min, max, step = 1, onChange,
}: { value: number; min: number; max: number; step?: number; onChange: (v: number) => void }) {
  return (
    <input
      className="slider"
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
    />
  );
}

export function Stepper({
  value, step, suffix = "", onChange,
}: { value: number; step: number; suffix?: string; onChange: (delta: number) => void }) {
  return (
    <div className="row" style={{ gap: 8 }}>
      <button className="icon-btn" onClick={() => onChange(-step)}>−</button>
      <div style={{ minWidth: 64, textAlign: "center", fontWeight: 700 }}>
        {value.toFixed(1)}{suffix}
      </div>
      <button className="icon-btn" onClick={() => onChange(step)}>+</button>
    </div>
  );
}

export function Sheet({
  title, children, onClose,
}: { title?: string; children: ReactNode; onClose: () => void }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);
  return (
    <div className="overlay" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>
        <div className="sheet-grip" />
        {title && (
          <div className="row between" style={{ marginBottom: 8 }}>
            <h2>{title}</h2>
            <button className="icon-btn" onClick={onClose}><X size={18} /></button>
          </div>
        )}
        {children}
      </div>
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="field">
      <label>{label}</label>
      {children}
    </div>
  );
}

export function StatusPill({ status }: { status: "online" | "connecting" | "offline" }) {
  const txt = status === "online" ? "Connecté" : status === "connecting" ? "Connexion…" : "Hors-ligne";
  return (
    <span className={`pill ${status}`}>
      <span className="led" />
      {txt}
    </span>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
