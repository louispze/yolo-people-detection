from random import randint
import time


class Person:
    """
    Représente une personne détectée par YOLO et suivie dans la scène.
    Inspiré de MyPerson (epcm18/PeopleCounting-ComputerVision).

    Chaque personne a :
    - un ID unique
    - une couleur RGB aléatoire pour l'affichage
    - un historique de trajectoire
    - une direction (up / down / None)
    - un système de vieillissement (age / max_age)
    """

    def __init__(self, person_id: int, cx: int, cy: int, max_age: int = 30):
        """
        Args:
            person_id : identifiant unique de la personne
            cx, cy    : centre initial de la boîte englobante
            max_age   : nombre max de frames sans détection avant suppression
        """
        self.i       = person_id
        self.x       = cx
        self.y       = cy
        self.max_age = max_age
        self.age     = 0          # frames consécutives sans détection

        # Couleur aléatoire pour dessiner la trajectoire (comme dans l'original)
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)

        self.tracks = []          # historique [(x, y), ...]
        self.done   = False       # True = personne expirée, à supprimer
        self.state  = '0'         # '0' = pas encore compté, '1' = compté
        self.dir    = None        # 'up' ou 'down'

        # Boîte englobante complète (utile pour affichage YOLO)
        self.bbox       = None    # (x1, y1, x2, y2)
        self.confidence = 1.0     # score YOLO

        # Timestamp pour stats
        self.first_seen = time.time()

    # ------------------------------------------------------------------
    # Getters (compatibilité avec l'original)
    # ------------------------------------------------------------------

    def getRGB(self):
        return (self.R, self.G, self.B)

    def getTracks(self):
        return self.tracks

    def getId(self):
        return self.i

    def getState(self):
        return self.state

    def getDir(self):
        return self.dir

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    # ------------------------------------------------------------------
    # Mise à jour de position
    # ------------------------------------------------------------------

    def updateCoords(self, xn: int, yn: int, bbox=None, confidence: float = 1.0):
        """
        Met à jour la position avec les nouvelles coordonnées détectées.

        Args:
            xn, yn     : nouveau centre détecté
            bbox       : (x1, y1, x2, y2) boîte englobante complète
            confidence : score YOLO
        """
        self.age = 0                        # reset : la personne est re-détectée
        self.tracks.append((self.x, self.y))
        self.x = xn
        self.y = yn

        if bbox is not None:
            self.bbox = bbox
        self.confidence = confidence

        # Limite la trajectoire aux 60 derniers points
        if len(self.tracks) > 60:
            self.tracks.pop(0)

    def age_one(self):
        """
        Vieillit la personne d'une frame (pas de détection ce tour).
        Retourne False si la personne dépasse max_age (à supprimer).
        """
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return not self.done

    # ------------------------------------------------------------------
    # Détection de direction (ligne virtuelle horizontale)
    # ------------------------------------------------------------------

    def going_UP(self, mid_start: int, mid_end: int) -> bool:
        """
        Détecte si la personne traverse la ligne vers le haut.

        Args:
            mid_start : y début de la zone de comptage
            mid_end   : y fin   de la zone de comptage
        Returns:
            True si franchissement vers le haut détecté (compté une seule fois)
        """
        if len(self.tracks) >= 2 and self.state == '0':
            if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end:
                self.state = '1'
                self.dir   = 'up'
                return True
        return False

    def going_DOWN(self, mid_start: int, mid_end: int) -> bool:
        """
        Détecte si la personne traverse la ligne vers le bas.

        Args:
            mid_start : y début de la zone de comptage
            mid_end   : y fin   de la zone de comptage
        Returns:
            True si franchissement vers le bas détecté (compté une seule fois)
        """
        if len(self.tracks) >= 2 and self.state == '0':
            if self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start:
                self.state = '1'
                self.dir   = 'down'
                return True
        return False

    # ------------------------------------------------------------------
    # Gestion du cycle de vie
    # ------------------------------------------------------------------

    def setDone(self):
        """Force la suppression de la personne."""
        self.done = True

    def timedOut(self):
        """Retourne True si la personne doit être supprimée."""
        return self.done

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def distanceTo(self, cx: int, cy: int) -> float:
        """Distance euclidienne entre le centre actuel et un point (cx, cy)."""
        return ((self.x - cx) ** 2 + (self.y - cy) ** 2) ** 0.5

    @property
    def duration(self) -> float:
        """Durée en secondes depuis la première détection."""
        return time.time() - self.first_seen

    def __repr__(self):
        return (
            f"Person(id={self.i}, pos=({self.x},{self.y}), "
            f"dir={self.dir}, state={self.state}, age={self.age})"
        )
