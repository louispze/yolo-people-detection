from person import Person


class Tracker:
    """
    Associe chaque détection YOLO à une personne existante (par distance)
    ou crée une nouvelle personne si aucun match trouvé.
    """

    def __init__(self, max_age: int = 30, max_distance: int = 80):
        """
        Args:
            max_age      : frames sans détection avant suppression d'une personne
            max_distance : distance max (px) pour associer une détection à une personne
        """
        self.persons      = []       # liste des Person actives
        self.next_id      = 0        # compteur d'IDs
        self.max_distance = max_distance
        self.max_age      = max_age

    # ------------------------------------------------------------------

    def update(self, detections: list) -> list:
        """
        Met à jour le tracker avec les nouvelles détections YOLO.

        Args:
            detections : liste de (cx, cy, x1, y1, x2, y2, confidence)

        Returns:
            Liste des Person actives
        """
        matched_ids = set()

        # --- Associe chaque détection à la personne la plus proche ---
        for det in detections:
            cx, cy, x1, y1, x2, y2, conf = det

            best_person   = None
            best_distance = self.max_distance   # seuil max

            for person in self.persons:
                if person.timedOut():
                    continue
                d = person.distanceTo(cx, cy)
                if d < best_distance:
                    best_distance = d
                    best_person   = person

            if best_person is not None:
                # Personne existante trouvée → on met à jour
                best_person.updateCoords(cx, cy, bbox=(x1, y1, x2, y2), confidence=conf)
                matched_ids.add(best_person.getId())
            else:
                # Aucun match → nouvelle personne
                p = Person(self.next_id, cx, cy, max_age=self.max_age)
                p.updateCoords(cx, cy, bbox=(x1, y1, x2, y2), confidence=conf)
                self.persons.append(p)
                matched_ids.add(self.next_id)
                self.next_id += 1

        # --- Vieillit les personnes non détectées ce tour ---
        for person in self.persons:
            if person.getId() not in matched_ids:
                person.age_one()

        # --- Supprime les personnes expirées ---
        self.persons = [p for p in self.persons if not p.timedOut()]

        return self.persons

    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Nombre de personnes actuellement visibles."""
        return len(self.persons)

    @property
    def total_seen(self) -> int:
        """Nombre total de personnes vues depuis le début (IDs assignés)."""
        return self.next_id
