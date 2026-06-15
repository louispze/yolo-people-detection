import { useEffect } from "react";
import { useStore } from "./store";
import type { ScreenId } from "./types";
import { NAV_ICONS } from "./ui/icons";
import { StatusPill } from "./ui/components";

import Dashboard from "./screens/Dashboard";
import HouseMap from "./screens/HouseMap";
import People from "./screens/People";
import Cameras from "./screens/Cameras";
import SettingsScreen from "./screens/Settings";

const NAV: { id: ScreenId; label: string }[] = [
  { id: "dashboard", label: "Accueil" },
  { id: "map", label: "Plan" },
  { id: "people", label: "Personnes" },
  { id: "cameras", label: "Caméras" },
  { id: "settings", label: "Réglages" },
];

const TITLES: Record<ScreenId, string> = {
  dashboard: "Mon domicile",
  map: "Plan de la maison",
  people: "Personnes",
  cameras: "Caméras",
  settings: "Réglages",
};

export default function App() {
  const screen = useStore((s) => s.screen);
  const status = useStore((s) => s.status);
  const setScreen = useStore((s) => s.setScreen);
  const connect = useStore((s) => s.connect);

  useEffect(() => {
    connect(); // tente la connexion live au démarrage (repli démo si offline)
  }, [connect]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-title">{TITLES[screen]}</div>
        <StatusPill status={status} />
      </header>

      <main className="app-content">
        {screen === "dashboard" && <Dashboard />}
        {screen === "map" && <HouseMap />}
        {screen === "people" && <People />}
        {screen === "cameras" && <Cameras />}
        {screen === "settings" && <SettingsScreen />}
      </main>

      <nav className="nav">
        {NAV.map((n) => {
          const Icon = NAV_ICONS[n.id];
          return (
            <button
              key={n.id}
              className={`nav-item ${screen === n.id ? "active" : ""}`}
              onClick={() => setScreen(n.id)}
            >
              <Icon size={22} />
              {n.label}
            </button>
          );
        })}
      </nav>
    </div>
  );
}
