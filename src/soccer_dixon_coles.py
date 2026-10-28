import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from scipy.optimize import minimize
from typing import Dict
from src.validation import vector


class DixonColesEngine:
    """
    Modelo de goles con corrección de Dixon-Coles y decaimiento temporal
    exponencial y soporte para xG ponderado.
    """

    def __init__(self, xi: float = 0.0035, xg_weight: float = 0.65, max_goals: int = 10):
        """
        :param xi: Tasa de decaimiento temporal (en días). 0.0035 da una vida media de ~200 días.
        :param xg_weight: Peso otorgado al xG frente a los goles reales en el ajuste [0, 1].
        :param max_goals: Límite superior de la rejilla de marcador exacto.
        """
        self.xi = xi
        vector([xi], "time decay", 0)
        vector([xg_weight], "xG weight", 0, 1)
        if max_goals < 2:
            raise ValueError("max_goals must be at least 2")
        self.xg_weight = xg_weight
        self.max_goals = max_goals
        self.teams = []
        self.params_ = {}

    @staticmethod
    def _tau_correction(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
        """Factor de correlación de Dixon-Coles para marcadores bajos (0-0, 1-0, 0-1, 1-1)."""
        tau = np.ones_like(lam, dtype=float)
        mask_00 = (x == 0) & (y == 0)
        mask_01 = (x == 0) & (y == 1)
        mask_10 = (x == 1) & (y == 0)
        mask_11 = (x == 1) & (y == 1)

        tau[mask_00] = 1.0 - lam[mask_00] * mu[mask_00] * rho
        tau[mask_01] = 1.0 + lam[mask_01] * rho
        tau[mask_10] = 1.0 + mu[mask_10] * rho
        tau[mask_11] = 1.0 - rho
        return np.clip(tau, 1e-8, None)

    def fit(self, df: pd.DataFrame, current_date: str = None):
        """
        Ajusta los parámetros de ataque, defensa, ventaja de localía y correlación rho.
        Espera columnas: ['date', 'home_team', 'away_team', 'home_goals', 'away_goals']
        Opcional: ['home_xg', 'away_xg']
        """
        data = df.copy()
        required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
        if data.empty or not set(required).issubset(data) or data[required].isna().any().any():
            raise ValueError("Non-empty complete match history is required")
        for key in ["home_goals", "away_goals", "home_xg", "away_xg"]:
            if key in data:
                vector(data[key], key, 0)
        if any(np.any(data[key] != data[key].astype(int)) for key in ["home_goals", "away_goals"]):
            raise ValueError("Observed goals must be integers")
        data['date'] = pd.to_datetime(data['date'])
        ref_date = pd.to_datetime(current_date) if current_date else data['date'].max()
        if (data['date'] > ref_date).any():
            raise ValueError("Training history contains results after the reference date")
        altitude = data.get("alt_boost", pd.Series(0., index=data.index)).to_numpy()

        # Decaimiento exponencial en días: phi(t) = exp(-xi * delta_t)
        days_elapsed = (ref_date - data['date']).dt.days.clip(lower=0).values
        weights = np.exp(-self.xi * days_elapsed)

        # Métrica compuesta (Goles + xG) para estabilizar tasas subyacentes
        if 'home_xg' in data.columns and 'away_xg' in data.columns:
            h_eff = self.xg_weight * data['home_xg'].values + (1.0 - self.xg_weight) * data['home_goals'].values
            a_eff = self.xg_weight * data['away_xg'].values + (1.0 - self.xg_weight) * data['away_goals'].values
        else:
            h_eff = data['home_goals'].values.astype(float)
            a_eff = data['away_goals'].values.astype(float)

        h_goals_int = data['home_goals'].values.astype(int)
        a_goals_int = data['away_goals'].values.astype(int)

        self.teams = sorted(list(set(data['home_team']).union(set(data['away_team']))))
        n_teams = len(self.teams)
        team_to_idx = {t: i for i, t in enumerate(self.teams)}

        h_idx = data['home_team'].map(team_to_idx).values
        a_idx = data['away_team'].map(team_to_idx).values

        # Vector de parámetros: [attack_0..N-1, defense_0..N-1, home_adv, rho]
        x0 = np.concatenate([np.ones(n_teams), np.ones(n_teams), [0.25, -0.05]])

        def neg_log_likelihood(params: np.ndarray) -> float:
            att = params[:n_teams]
            dfn = params[n_teams:2*n_teams]
            h_adv = params[-2]
            rho = params[-1]

            lam = np.exp(att[h_idx] - dfn[a_idx] + h_adv + altitude)
            mu = np.exp(att[a_idx] - dfn[h_idx] - .5*altitude)

            # Log-verosimilitud de Poisson continua usando goles efectivos + corrección DC
            ll_home = h_eff * np.log(lam) - lam
            ll_away = a_eff * np.log(mu) - mu
            ll_tau = np.log(self._tau_correction(h_goals_int, a_goals_int, lam, mu, rho))

            # Penalización de identificabilidad: suma de ataques = 0 en escala log
            penalty = 100.0 * (np.mean(att) ** 2)
            return -float(np.sum(weights * (ll_home + ll_away + ll_tau))) + penalty

        bounds = [(-3.0, 3.0)] * (2 * n_teams) + [(0.0, 1.0), (-0.25, 0.25)]
        res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds,
                       options={"maxiter": 1500, "maxfun": 100000})
        if not res.success or not np.isfinite(res.fun):
            raise RuntimeError(f"Dixon-Coles fit failed: {res.message}")

        att_opt = res.x[:n_teams] - np.mean(res.x[:n_teams])
        dfn_opt = res.x[n_teams:2*n_teams] - np.mean(res.x[:n_teams])

        self.params_ = {
            'attack': dict(zip(self.teams, att_opt)),
            'defense': dict(zip(self.teams, dfn_opt)),
            'home_adv': float(res.x[-2]),
            'rho': float(res.x[-1])
        }
        return self

    def predict_match_markets(self, home_team: str, away_team: str, alt_boost: float = 0.0) -> Dict:
        """
        Genera la matriz conjunta de goles y calcula las probabilidades exactas para:
        - 1X2 (Local, Empate, Visita)
        - Over / Under 2.5 goles
        - Ambos Equipos Anotan (BTTS)
        - Hándicap Asiático (-0.5, 0.0 / DNB, -1.0)
        :param alt_boost: Ajuste aditivo en escala log por choque de altitud (ej. Liga MX).
        """
        if not self.params_ or home_team not in self.teams or away_team not in self.teams:
            raise ValueError("Both teams must be present in the fitted model")
        vector([alt_boost], "altitude boost")
        att = self.params_['attack']
        dfn = self.params_['defense']
        h_adv = self.params_['home_adv']
        rho = self.params_['rho']

        lam = float(np.exp(att[home_team] - dfn[away_team] + h_adv + alt_boost))
        mu = float(np.exp(att[away_team] - dfn[home_team] - 0.5 * alt_boost))

        if not np.isfinite(lam + mu) or max(lam, mu) > 100:
            raise ValueError("Goal rate outside supported model range")
        goal_limit = max(self.max_goals, int(poisson.ppf(1-1e-10, max(lam, mu))))
        goals = np.arange(goal_limit + 1)
        pmf_h = poisson.pmf(goals, lam)
        pmf_a = poisson.pmf(goals, mu)
        matrix = np.outer(pmf_h, pmf_a)

        # Aplicar factor de correlación tau en la esquina superior izquierda (0,0), (0,1), (1,0), (1,1)
        matrix[0, 0] *= max(0.0, 1.0 - lam * mu * rho)
        matrix[0, 1] *= max(0.0, 1.0 + lam * rho)
        matrix[1, 0] *= max(0.0, 1.0 + mu * rho)
        matrix[1, 1] *= max(0.0, 1.0 - rho)
        matrix /= np.sum(matrix)  # Renormalizar matriz conjunta

        p_home = float(np.sum(np.tril(matrix, -1)))
        p_draw = float(np.sum(np.diag(matrix)))
        p_away = float(np.sum(np.triu(matrix, 1)))

        # Totales (Over / Under 2.5)
        i_grid, j_grid = np.meshgrid(goals, goals, indexing='ij')
        total_goals = i_grid + j_grid
        p_over_25 = float(np.sum(matrix[total_goals > 2.5]))
        p_under_25 = 1.0 - p_over_25

        # Ambos Anotan (BTTS)
        p_btts_yes = float(np.sum(matrix[1:, 1:]))

        # Distribución de Skellam para Diferencia de Goles (Local - Visita)
        # Probabilidad de ganar Hándicap Asiático 0.0 (Draw No Bet)
        p_ah_home_0 = p_home / (p_home + p_away)
        # Probabilidad analítica Skellam de ganar por 2 o más goles (Hándicap -1.5)
        p_skellam_minus_15 = float(1.0 - skellam.cdf(1, lam, mu))

        return {
            "lambda_home": round(lam, 3),
            "mu_away": round(mu, 3),
            "1X2": np.array([p_home, p_draw, p_away]),
            "Over_Under_2.5": np.array([p_over_25, p_under_25]),
            "BTTS": np.array([p_btts_yes, 1.0 - p_btts_yes]),
            "AH_0.0_Home": round(p_ah_home_0, 4),
            "Skellam_Home_-1.5": round(p_skellam_minus_15, 4),
            "score_matrix": matrix
        }
