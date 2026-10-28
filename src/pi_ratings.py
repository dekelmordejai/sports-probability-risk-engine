import numpy as np
import pandas as pd
from typing import Dict, Tuple

class PiRatingSystem:
    """
    Adaptación de los Pi-Ratings de Constantinou & Fenton (2013), con mezcla de xG.
    Mantiene vectores separados de rendimiento Local (R_H) y Visitante (R_A).
    """

    def __init__(self, lambda_rate: float = 0.035, gamma_transfer: float = 0.70, c_scale: float = 3.0):
        self.lam = lambda_rate
        self.gamma = gamma_transfer
        self.c = c_scale
        self.ratings_home: Dict[str, float] = {}
        self.ratings_away: Dict[str, float] = {}

    def _rating_to_goals(self, r: float) -> float:
        """Transforma el rating latente en diferencial de goles esperado."""
        sign = 1.0 if r >= 0.0 else -1.0
        return sign * ((10.0 ** (abs(r) / self.c)) - 1.0)

    def get_team_ratings(self, team: str) -> Tuple[float, float]:
        return self.ratings_home.get(team, 0.0), self.ratings_away.get(team, 0.0)

    def predict_expected_goal_diff(self, home_team: str, away_team: str) -> float:
        r_hh, _ = self.get_team_ratings(home_team)
        _, r_aa = self.get_team_ratings(away_team)
        return self._rating_to_goals(r_hh) - self._rating_to_goals(r_aa)

    def update(self, home_team: str, away_team: str, home_goals: float, away_goals: float):
        """Actualiza los 4 parámetros de rating tras observar el resultado del encuentro."""
        r_hh, r_ha = self.get_team_ratings(home_team)
        r_ah, r_aa = self.get_team_ratings(away_team)

        expected_gd = self._rating_to_goals(r_hh) - self._rating_to_goals(r_aa)
        actual_gd = float(home_goals - away_goals)

        error = abs(actual_gd - expected_gd)
        psi = self.c * np.log10(1.0 + error)
        delta = psi if actual_gd > expected_gd else (-psi if actual_gd < expected_gd else 0.0)

        # Actualización del equipo local
        self.ratings_home[home_team] = r_hh + self.lam * delta
        self.ratings_away[home_team] = r_ha + self.gamma * (self.lam * delta)

        # Actualización simétrica del equipo visitante
        self.ratings_away[away_team] = r_aa - self.lam * delta
        self.ratings_home[away_team] = r_ah - self.gamma * (self.lam * delta)

    def compute_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula features antes de incorporar resultados del mismo día.
        Se supone que los resultados de días anteriores ya estaban disponibles.
        """
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values(["date", "home_team", "away_team"], kind="stable")
        pi_hh_list, pi_aa_list, pi_egd_list = [], [], []

        for _, day in data.groupby(data["date"].dt.normalize(), sort=True):
            for row in day.itertuples():
                h, a = row.home_team, row.away_team
                r_hh, _ = self.get_team_ratings(h)
                _, r_aa = self.get_team_ratings(a)
                pi_hh_list.append(r_hh)
                pi_aa_list.append(r_aa)
                pi_egd_list.append(self.predict_expected_goal_diff(h, a))
            # Publicar los resultados después de producir todas las features del día.
            for row in day.itertuples():
                eff_hg = 0.5 * row.home_goals + 0.5 * getattr(row, "home_xg", row.home_goals)
                eff_ag = 0.5 * row.away_goals + 0.5 * getattr(row, "away_xg", row.away_goals)
                self.update(row.home_team, row.away_team, eff_hg, eff_ag)

        data["pi_home_h"] = pi_hh_list
        data["pi_away_a"] = pi_aa_list
        data["pi_expected_gd"] = pi_egd_list
        return data
