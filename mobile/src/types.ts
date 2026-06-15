// ============================================================================
// Types partagés — contrat unique pour toute l'app
// ============================================================================

/** Une personne du foyer, avec sa préférence de température. */
export interface Person {
  id: string;
  name: string;
  color: string;          // couleur d'avatar (hex)
  preferredTemp: number;  // température préférée (°C)
  /** Pièce où la personne est marquée présente (assignation manuelle), ou null. */
  presentRoomId: string | null;
}

/** Données live issues du backend YOLO pour une pièce. */
export interface LiveRoomState {
  people: number;                    // nb de personnes détectées
  active: number;                    // nb de personnes "en mouvement"
  workout: number;                   // nb de personnes faisant un exercice
  exercises: Record<string, number>; // { squat: 1, pompes: 2, ... }
  updatedAt: number;                 // timestamp (ms)
}

/** Une pièce de la maison. */
export interface Room {
  id: string;
  name: string;
  area: string;          // zone de la maison (ex: "Rez-de-chaussée", "Étage")
  icon: string;          // clé d'icône (voir ui/icons)
  cameraSource: string;  // identifiant de flux caméra côté backend (ex: "salon")
  // Climat
  currentTemp: number;   // température mesurée (°C)
  baseTarget: number;    // consigne de base (°C) quand vide
  minTemp: number;       // borne basse de sécurité
  maxTemp: number;       // borne haute de sécurité
  // Position sur le plan de la maison (grille en %)
  map: { x: number; y: number; w: number; h: number };
}

/** Paramètres du moteur de climat adaptatif (réglables). */
export interface ClimateConfig {
  occupancyStep: number;   // nb de personnes par palier (ex: 3 -> -1°C toutes les 3 pers.)
  occupancyDelta: number;  // °C retirés par palier d'occupation
  activityDelta: number;   // °C retirés par personne "en mouvement"
  workoutDelta: number;    // °C retirés par personne en exercice
  blendMode: "average" | "coolest"; // comment combiner les préférences des présents
}

/** Réglages de connexion au backend (Raspberry Pi). */
export interface Connection {
  host: string;   // ip/hostname du Pi
  port: number;   // port du backend
  useTls: boolean;
}

export type ConnStatus = "connecting" | "online" | "offline";

/** Résultat calculé du moteur pour une pièce. */
export interface ClimateResult {
  target: number;        // consigne finale appliquée (°C)
  base: number;          // base avant pénalités
  occupancyPenalty: number;
  activityPenalty: number;
  workoutPenalty: number;
  reasons: string[];     // explications lisibles ("3 personnes -1°C", ...)
}

/** Un appareil simple (lumière) — repris de l'interface existante. */
export interface Light {
  id: string;
  name: string;
  roomId: string | null;
  on: boolean;
  brightness: number; // 0..100
}

export type ScreenId = "dashboard" | "map" | "people" | "cameras" | "settings";
