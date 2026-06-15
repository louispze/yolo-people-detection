// Mapping clé -> icône lucide, + ré-exports utiles pour les écrans.
import {
  Sofa, Utensils, BedDouble, Monitor, Bath, Home, DoorOpen, Car, Baby, Sun,
  LayoutDashboard, Map, Users, Cctv, Settings, type LucideIcon,
} from "lucide-react";

export const ROOM_ICONS: Record<string, LucideIcon> = {
  sofa: Sofa,
  kitchen: Utensils,
  bed: BedDouble,
  desk: Monitor,
  bath: Bath,
  home: Home,
  door: DoorOpen,
  garage: Car,
  kids: Baby,
  terrace: Sun,
};

/** Liste pour le sélecteur d'icône dans l'éditeur de pièce. */
export const ROOM_ICON_KEYS = Object.keys(ROOM_ICONS);

export function RoomIcon({ icon, size = 20 }: { icon: string; size?: number }) {
  const Cmp = ROOM_ICONS[icon] ?? Home;
  return <Cmp size={size} />;
}

export const NAV_ICONS = {
  dashboard: LayoutDashboard,
  map: Map,
  people: Users,
  cameras: Cctv,
  settings: Settings,
} as const;

export * from "lucide-react";
