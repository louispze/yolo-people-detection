// ============================================================================
// Moteur de climat adaptatif — règles de température
// ============================================================================
// Encode les règles demandées :
//   - température de base = préférence des personnes présentes (ou base pièce)
//   - plusieurs personnes dans une pièce  -> on baisse (chaleur corporelle)
//   - personnes qui bougent beaucoup       -> on baisse
//   - personnes en activité physique       -> on baisse (plus fort)
// Reprend et généralise la logique de l'interface d'origine
// (computeTarget : -1°C toutes les 3 personnes).
// ============================================================================

import type {
  Room,
  Person,
  LiveRoomState,
  ClimateConfig,
  ClimateResult,
} from "../types";

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const round1 = (v: number) => Math.round(v * 10) / 10;

/**
 * Calcule la consigne finale d'une pièce à partir de sa config, des personnes
 * présentes et de l'état live YOLO.
 */
export function computeClimate(
  room: Room,
  presentPeople: Person[],
  live: LiveRoomState | undefined,
  cfg: ClimateConfig
): ClimateResult {
  const reasons: string[] = [];

  // 1) Base : préférences des personnes présentes, sinon base de la pièce
  let base = room.baseTarget;
  if (presentPeople.length > 0) {
    const prefs = presentPeople.map((p) => p.preferredTemp);
    base =
      cfg.blendMode === "coolest"
        ? Math.min(...prefs)
        : prefs.reduce((a, b) => a + b, 0) / prefs.length;
    reasons.push(
      `Base ${round1(base)}°C (${cfg.blendMode === "coolest" ? "préf. la + fraîche" : "moyenne des préférences"} de ${presentPeople
        .map((p) => p.name)
        .join(", ")})`
    );
  } else {
    reasons.push(`Base ${round1(base)}°C (pièce)`);
  }

  const people = live?.people ?? 0;
  const active = live?.active ?? 0;
  const workout = live?.workout ?? 0;

  // 2) Pénalité d'occupation : -occupancyDelta par palier de occupancyStep personnes
  const steps = cfg.occupancyStep > 0 ? Math.floor(people / cfg.occupancyStep) : 0;
  const occupancyPenalty = steps * cfg.occupancyDelta;
  if (occupancyPenalty > 0) {
    reasons.push(`${people} personnes → −${round1(occupancyPenalty)}°C`);
  }

  // 3) Pénalité de mouvement
  const activityPenalty = active * cfg.activityDelta;
  if (activityPenalty > 0) {
    reasons.push(`${active} en mouvement → −${round1(activityPenalty)}°C`);
  }

  // 4) Pénalité d'activité physique (plus forte)
  const workoutPenalty = workout * cfg.workoutDelta;
  if (workoutPenalty > 0) {
    reasons.push(`${workout} en exercice → −${round1(workoutPenalty)}°C`);
  }

  const raw = base - occupancyPenalty - activityPenalty - workoutPenalty;
  const target = round1(clamp(raw, room.minTemp, room.maxTemp));
  if (raw < room.minTemp) reasons.push(`Plancher ${room.minTemp}°C atteint`);

  return {
    target,
    base: round1(base),
    occupancyPenalty: round1(occupancyPenalty),
    activityPenalty: round1(activityPenalty),
    workoutPenalty: round1(workoutPenalty),
    reasons,
  };
}

/** Config climat par défaut (équivaut à l'ancienne règle -1°C/3 pers). */
export const DEFAULT_CLIMATE: ClimateConfig = {
  occupancyStep: 3,
  occupancyDelta: 1,
  activityDelta: 0.5,
  workoutDelta: 1,
  blendMode: "average",
};
