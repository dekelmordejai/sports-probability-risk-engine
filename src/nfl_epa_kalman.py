import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from typing import Dict, List, Tuple


class NFLKalmanKeyNumberEngine:
    """
    Motor Cuantitativo para la NFL basado en:
    1. Agregación Play-by-Play (EPA/play de pase y carrera filtrando Garbage Time).
    2. Filtro de Kalman Dinámico (seguimiento de media y matriz de covarianza de los 32 equipos).
    3. Distribución Discreta de Números Clave (3, 7, 6, 10, 4, 14) para Spread y Moneyline.
    """

    # Multiplicadores empíricos de masa de probabilidad en márgenes clave de la NFL
    KEY_NUMBER_WEIGHTS = {
        3: 2.35,   # ~15% de todos los partidos de NFL se deciden por exactamente 3 puntos
        7: 1.95,   # ~9% se deciden por 7 puntos (Touchdown + PAT)
        6: 1.40,   # Touchdown sin punto extra o dos goles de campo
        10: 1.45,  # TD + FG
        4: 1.25,   # TD vs FG
        14: 1.35,  # Dos Touchdowns
        1: 1.15,
        2: 1.10,
        8: 1.15,   # TD + conversión de 2 puntos
        17: 1.20   # 2 TDs + FG
    }

    def __init__(
        self,
        process_var_q: float = 0.45,
        obs_var_r: float = 135.0,
        base_game_std: float = 13.2,
        hfa_base: float = 1.85,
        phi_decay: float = 0.985
    ):
        """
        :param process_var_q: Varianza de transición semanal (evolución de la fuerza del equipo).
        :param obs_var_r: Varianza del ruido de observación de un partido individual.
        :param base_game_std: Desviación estándar estructural del marcador en la NFL (~13.2 pts).
        :param hfa_base: Ventaja de localía base en la NFL moderna (~1.85 puntos).
        :param phi_decay: Factor de regresión a la media semanal.
        """
        self.q = process_var_q
        self.r = obs_var_r
        self.base_game_std = base_game_std
        self.hfa_base = hfa_base
        self.phi = phi_decay
        if min(process_var_q, obs_var_r, base_game_std) <= 0 or not 0 < phi_decay <= 1:
            raise ValueError("Invalid Kalman variances or decay")

        self.teams: List[str] = []
        self.team_to_idx: Dict[str, int] = {}
        self.x_state: np.ndarray = np.array([])  # Vector de medias latentes (N,)
        self.P_cov: np.ndarray = np.array([])    # Matriz de covarianza de estado (N, N)

    @staticmethod
    def load_and_aggregate_nflverse(seasons: List[int]) -> pd.DataFrame:
        """
        Descarga Play-by-Play y calendarios desde nfl_data_py, filtra Garbage Time
        y construye el DataFrame por partido con EPA ponderado, clima, descanso y cuotas.
        """
        import nfl_data_py as nfl
        import io
        import requests

        pbp = nfl.import_pbp_data(seasons, columns=["game_id", "posteam", "play_type", "epa", "qtr", "wp"],
                                  include_participation=False, downcast=False)
        response = requests.get("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv", timeout=30)
        response.raise_for_status()
        sched = pd.read_csv(io.StringIO(response.text))
        sched = sched[sched.season.isin(seasons)]

        # Filtrar solo jugadas reales de pase o carrera y eliminar Garbage Time (WP entre 5% y 95% en Q3/Q4)
        valid_plays = pbp[
            (pbp["play_type"].isin(["pass", "run"])) &
            (pbp["epa"].notna()) &
            ~((pbp["qtr"] >= 3) & ((pbp["wp"] < 0.05) | (pbp["wp"] > 0.95)))
        ].copy()

        # Calcular EPA/play separado por Pase y Carrera para cada equipo por partido
        grp = valid_plays.groupby(["game_id", "posteam", "play_type"])["epa"].mean().unstack(fill_value=0.0)
        if "pass" not in grp.columns:
            grp["pass"] = 0.0
        if "run" not in grp.columns:
            grp["run"] = 0.0

        # Índice compuesto: 62% Pase + 38% Carrera escalado a 65 jugadas estándar
        grp["weighted_epa_pts"] = 65.0 * (0.62 * grp["pass"] + 0.38 * grp["run"])
        epa_df = grp[["weighted_epa_pts"]].reset_index()

        # Unir con tabla de partidos (schedules) que incluye líneas de apuesta, descanso y viento
        games = sched[sched["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
        games = games.dropna(subset=["home_score", "away_score"]).copy()

        games = games.merge(
            epa_df.rename(columns={"posteam": "home_team", "weighted_epa_pts": "home_epa_pts"}),
            on=["game_id", "home_team"], how="left"
        ).merge(
            epa_df.rename(columns={"posteam": "away_team", "weighted_epa_pts": "away_epa_pts"}),
            on=["game_id", "away_team"], how="left"
        )

        games["home_epa_pts"] = games["home_epa_pts"].fillna(0.0)
        games["away_epa_pts"] = games["away_epa_pts"].fillna(0.0)
        games["actual_margin"] = games["home_score"] - games["away_score"]

        # Señal compuesta para el Filtro de Kalman: 65% diferencial de EPA + 35% margen real acotado a [-28, 28]
        epa_diff = games["home_epa_pts"] - games["away_epa_pts"]
        capped_margin = games["actual_margin"].clip(-28.0, 28.0)
        games["kalman_obs_y"] = 0.65 * epa_diff + 0.35 * capped_margin

        # Covariables contextuales: diferencial de días de descanso y viento en estadios abiertos
        games["rest_diff"] = (games["home_rest"].fillna(7) - games["away_rest"].fillna(7)).clip(-7, 7)
        games["wind_mph"] = pd.to_numeric(games["wind"], errors="coerce").fillna(0.0)
        games["is_outdoors"] = games["roof"].isin(["outdoors", "open"]).astype(int)

        return games.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    def initialize_states(self, teams: List[str], prior_std: float = 4.5):
        """Inicializa el vector de estado x_0 = 0 y covarianza P_0 = prior_std^2 * I."""
        if len(set(teams)) < 2 or prior_std <= 0:
            raise ValueError("At least two teams and positive prior_std are required")
        self.teams = sorted(list(set(teams)))
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        self.x_state = np.zeros(n, dtype=float)
        self.P_cov = np.eye(n, dtype=float) * (prior_std ** 2)

    def apply_offseason_regression(self, regression_factor: float = 0.65, offseason_var_boost: float = 6.0):
        """
        Al cambiar de temporada en la NFL (Draft, agencia libre, tope salarial),
        los ratings regresan un 35% hacia la media (0) y aumenta la incertidumbre P.
        """
        self.x_state *= regression_factor
        self.P_cov = (regression_factor ** 2) * self.P_cov + np.eye(len(self.teams)) * offseason_var_boost

    def step_predict_and_update(
        self, home_team: str, away_team: str, rest_diff: float = 0.0, observed_y: float = None
    ) -> Tuple[float, float]:
        """
        Ejecuta el paso de predicción a priori (devolviendo mu_spread y sigma_total)
        y, si observed_y está presente, actualiza el estado a posteriori de Kalman.
        """
        i = self.team_to_idx[home_team]
        j = self.team_to_idx[away_team]
        n = len(self.teams)

        # Vector de diseño H_t: +1 para el local, -1 para el visitante
        h_vec = np.zeros(n, dtype=float)
        h_vec[i] = 1.0
        h_vec[j] = -1.0

        # Ajuste contextual: cada día extra de descanso (ej. volver de Bye Week) vale ~0.22 puntos
        context_shift = self.hfa_base + 0.22 * rest_diff

        if observed_y is not None:
            if not np.isfinite(observed_y):
                raise ValueError("Observation must be finite")
            transition = np.eye(n)
            transition[i, i] = transition[j, j] = self.phi
            self.x_state = transition @ self.x_state
            self.P_cov = transition @ self.P_cov @ transition.T
        # 1. Predicción a priori del margen (Local - Visita)
        mu_spread = float(h_vec @ self.x_state) + context_shift
        epistemic_var = float(h_vec @ self.P_cov @ h_vec)
        total_std = float(np.sqrt(self.base_game_std ** 2 + epistemic_var))

        # 2. Actualización de Kalman (Measurement Update) si el partido ya se jugó
        if observed_y is not None:
            # Añadir ruido de proceso local a los dos equipos involucrados
            self.P_cov[i, i] += self.q
            self.P_cov[j, j] += self.q

            innovation = observed_y - mu_spread
            innovation_var = float(h_vec @ self.P_cov @ h_vec) + self.r
            kalman_gain = (self.P_cov @ h_vec) / innovation_var  # Vector (N,)

            # Actualizar estado y matriz de covarianza (Forma estabilizada)
            self.x_state = self.x_state + kalman_gain * innovation
            self.x_state -= np.mean(self.x_state)  # Centrar suma cero en la liga
            residual = np.eye(n) - np.outer(kalman_gain, h_vec)
            self.P_cov = residual @ self.P_cov @ residual.T + self.r*np.outer(kalman_gain, kalman_gain)
            self.P_cov = (self.P_cov + self.P_cov.T)/2

        return mu_spread, total_std

    def build_key_number_pmf(self, mu_spread: float, sigma_spread: float, max_margin: int = 55) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construye la distribución de probabilidad discreta exacta para D = Home_Score - Away_Score,
        inyectando los picos multimodales en los números clave de la NFL (3, 7, 6, 10, 4, 14).
        """
        if not np.isfinite(mu_spread + sigma_spread) or sigma_spread <= 0 or max_margin < 1:
            raise ValueError("Invalid margin distribution parameters")
        margins = np.arange(-max_margin, max_margin + 1)
        # Densidad base con colas pesadas (Student-t con 6 grados de libertad)
        raw_probs = student_t.pdf(margins, df=6, loc=mu_spread, scale=sigma_spread)

        # En la NFL casi no existen empates al final del tiempo extra (~0.4%); atenuar k=0
        for idx, k in enumerate(margins):
            if k == 0:
                raw_probs[idx] *= 0.08
            else:
                abs_k = abs(int(k))
                if abs_k in self.KEY_NUMBER_WEIGHTS:
                    raw_probs[idx] *= self.KEY_NUMBER_WEIGHTS[abs_k]

        pmf = raw_probs / np.sum(raw_probs)
        return margins, pmf

    def price_nfl_markets(
        self, home_team: str, away_team: str, bookie_spread_home: float, rest_diff: float = 0.0
    ) -> Dict:
        """
        Valora exactamente una línea de Spread (ej. Home -3.0 o Home -2.5) y el Moneyline,
        contabilizando explícitamente la probabilidad de PUSH cuando la línea es entera.
        Convención: bookie_spread_home = -3.0 significa que el equipo local es favorito por 3 puntos.
        Para cubrir el spread, necesitamos: (Home_Score - Away_Score) + bookie_spread_home > 0.
        """
        mu_spread, sigma_spread = self.step_predict_and_update(
            home_team, away_team, rest_diff=rest_diff, observed_y=None
        )
        margins, pmf = self.build_key_number_pmf(mu_spread, sigma_spread)

        # Condición de cobertura del equipo local: D + spread_home > 0
        net_result = margins + bookie_spread_home
        p_home_cover = float(np.sum(pmf[net_result > 1e-6]))
        p_push = float(np.sum(pmf[np.abs(net_result) <= 1e-6]))
        p_away_cover = float(np.sum(pmf[net_result < -1e-6]))

        # Probabilidad efectiva de cubrir excluyendo el reembolso por Push (Draw No Bet equivalente)
        denom_spread = p_home_cover + p_away_cover
        p_home_cover_no_push = p_home_cover / denom_spread if denom_spread > 0 else 0.5

        # Probabilidad Moneyline (Victoria directa)
        p_home_win = float(np.sum(pmf[margins > 0]))
        p_away_win = float(np.sum(pmf[margins < 0]))
        p_home_ml_clean = p_home_win / (p_home_win + p_away_win)

        return {
            "Match": f"{away_team} @ {home_team}",
            "Kalman_Fair_Spread": round(-mu_spread, 2),
            "Epistemic_Std": round(sigma_spread, 2),
            "Bookie_Spread_Home": bookie_spread_home,
            "P_Home_Cover_Raw": p_home_cover,
            "P_Push": p_push,
            "P_Away_Cover_Raw": p_away_cover,
            "P_Home_Cover_NoPush": round(p_home_cover_no_push, 4),
            "P_Home_Moneyline": round(p_home_ml_clean, 4),
            "P_Away_Moneyline": round(1.0 - p_home_ml_clean, 4)
        }
