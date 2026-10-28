import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norm
from src.validation import vector, probability_odds, simplex
from itertools import product
from typing import List, Dict, Tuple, Union


class MarketMechanics:
    """Microestructura de mercado: Método de Shin, Entropía, Early Payout y Middling."""

    @staticmethod
    def devig_multiplicative(odds: Union[List[float], np.ndarray]) -> np.ndarray:
        inv_odds = 1.0 / vector(odds, "odds", 1, strict_lower=True)
        if len(inv_odds) < 2:
            raise ValueError("A market needs at least two outcomes")
        return inv_odds / np.sum(inv_odds)

    @staticmethod
    def devig_shin(odds: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, float]:
        odds_arr = vector(odds, "odds", 1, strict_lower=True)
        if len(odds_arr) < 2:
            raise ValueError("A market needs at least two outcomes")
        if np.any(odds_arr <= 1.0):
            raise ValueError("Todas las cuotas decimales deben ser estrictamente mayores a 1.0")

        pi = 1.0 / odds_arr
        s = np.sum(pi)
        if s <= 1.0 + 1e-7:
            return pi / s, 0.0

        def shin_probs(z: float) -> np.ndarray:
            return (2.0 * pi**2 / s) / (np.sqrt(z**2 + 4.0 * (1.0 - z) * pi**2 / s) + z)

        def objective(z: float) -> float:
            return float(np.sum(shin_probs(z)) - 1.0)

        z_star = brentq(objective, 1e-12, 1.0 - 1e-12)
        return shin_probs(z_star), float(z_star)

    @staticmethod
    def shannon_entropy(probs: Union[List[float], np.ndarray], normalized: bool = True) -> float:
        p = simplex(probs)
        p = p[p > 0]
        h = -np.sum(p * np.log2(p))
        if normalized and len(probs) > 1:
            return float(h / np.log2(len(probs)))
        return float(h)

    @staticmethod
    def calculate_ev(model_probs: np.ndarray, bookie_odds: np.ndarray, push_probs=None) -> np.ndarray:
        p, o = probability_odds(model_probs, bookie_odds)
        pushes = np.zeros_like(p) if push_probs is None else vector(push_probs, "push probabilities", 0, 1)
        if pushes.shape != p.shape or np.any(p + pushes > 1 + 1e-10):
            raise ValueError("Invalid win/push probabilities")
        return p * o + pushes - 1

    @staticmethod
    def adjust_prob_for_early_payout(win_prob: float, sport: str = "Soccer") -> float:
        """
        Incrementa la probabilidad efectiva de victoria cuando la casa ofrece Pago Anticipado
        (2 goles de ventaja en Fútbol, 20 pts en NBA, 17 pts en NFL) usando aproximación de
        probabilidad de cruce de barrera de reflexión estocástica.
        """
        vector([win_prob], "probability", 0, 1)
        if win_prob <= 0.05 or win_prob >= 0.95:
            return win_prob
        early_triggers = {"Soccer": 0.038, "NBA": 0.032, "NFL": 0.029}
        boost = early_triggers.get(sport, 0.025) * 4.0 * win_prob * (1.0 - win_prob)
        return float(np.clip(win_prob + boost, 0.0, 0.99))

    @staticmethod
    def evaluate_middle_opportunity(
        line_fav: float, odds_fav: float, line_dog: float, odds_dog: float,
        mu_margin: float, sigma_margin: float
    ) -> Dict[str, float]:
        """
        Evalúa un corredor de Middling (ej. Favorito -4.5 vs Underdog +6.5).
        Ambas apuestas ganan si el margen final cae dentro de (|line_fav|, line_dog).
        """
        vector([sigma_margin], "sigma", 0, strict_lower=True)
        probability_odds([.5, .5], [odds_fav, odds_dog])
        lower_bound = abs(line_fav)
        upper_bound = line_dog
        if upper_bound <= lower_bound:
            return {"is_valid_middle": 0.0, "p_middle_hit": 0.0, "ev_combined_%": -100.0}

        p_middle = float(norm.cdf(upper_bound, mu_margin, sigma_margin) - norm.cdf(lower_bound, mu_margin, sigma_margin))
        # Suponiendo 1 unidad apostada a cada lado (inversión total = 2 unidades)
        ret_middle = (odds_fav - 1.0) + (odds_dog - 1.0)
        left = norm.cdf(lower_bound, mu_margin, sigma_margin)
        right = norm.sf(upper_bound, mu_margin, sigma_margin)
        ev_total = (p_middle * ret_middle + left * (odds_dog-2) + right * (odds_fav-2)) / 2
        return {
            "is_valid_middle": 1.0,
            "p_middle_hit": round(p_middle, 4),
            "ev_combined_%": round(ev_total * 100.0, 2)
        }


class PortfolioManager:
    """Asignación Kelly, redondeo hacia abajo y simulación Monte Carlo de escenarios."""

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_total_exposure: float = 0.15,
        min_ev: float = 0.025,
        round_step_mxn: float = 50.0
    ):
        vector([kelly_fraction], "Kelly fraction", 0, 1, strict_lower=True)
        vector([max_total_exposure], "exposure", 0, .999999)
        vector([round_step_mxn], "rounding step", 0, strict_lower=True)
        self.kelly_fraction = kelly_fraction
        self.max_total_exposure = max_total_exposure
        self.min_ev = min_ev
        self.round_step_mxn = round_step_mxn

    @staticmethod
    def humanize_stake_mxn(raw_amount: float, step: float = 50.0, max_cap_mxn: float = None) -> float:
        """
        Redondea hacia abajo a un paso monetario, respetando el tope configurado.
        El nombre del método se conserva por compatibilidad con la API original.
        """
        vector([raw_amount], "amount", 0)
        vector([step], "step", 0, strict_lower=True)
        if max_cap_mxn is not None:
            vector([max_cap_mxn], "cap", 0)
            raw_amount = min(raw_amount, max_cap_mxn)
        return float(np.floor(raw_amount / step) * step)

    def optimize_simultaneous_kelly(
        self, probs: List[float], odds: List[float], labels: List[str] = None, push_probs=None
    ) -> pd.DataFrame:
        p_arr, o_arr = probability_odds(probs, odds)
        pushes = np.zeros_like(p_arr) if push_probs is None else vector(push_probs, "push probabilities", 0, 1)
        ev_arr = MarketMechanics.calculate_ev(p_arr, o_arr, pushes)

        valid_mask = ev_arr >= self.min_ev
        n_total = len(p_arr)
        if labels is None:
            labels = [f"Bet_{i+1}" for i in range(n_total)]

        if len(labels) != n_total:
            raise ValueError("Label count must match probabilities")
        final_stakes = np.zeros(n_total)
        if not np.any(valid_mask):
            return pd.DataFrame({
                "Selección": labels, "Prob_Modelo": p_arr,
                "Cuota": o_arr, "EV_%": ev_arr * 100, "Stake_%": final_stakes * 100
            })

        p_val = p_arr[valid_mask]
        o_val = o_arr[valid_mask]
        b_val = o_val - 1.0
        m = len(p_val)
        push_val = pushes[valid_mask]
        has_push = np.any(push_val > 0)

        if (not has_push and m <= 12) or (has_push and m <= 8):
            states = np.array(list(product([0, 1, 2] if has_push else [0, 1], repeat=m)))
            state_probs = np.prod(np.where(states == 1, p_val, np.where(states == 2, push_val, 1-p_val-push_val)), axis=1)
            returns_matrix = np.where(states == 1, b_val, np.where(states == 2, 0., -1.0))
        else:
            rng = np.random.default_rng(42)
            n_sims = 20000
            uniform = rng.random((n_sims, m))
            states = np.where(uniform < p_val, 1, np.where(uniform < p_val+push_val, 2, 0))
            state_probs = np.full(n_sims, 1.0 / n_sims)
            returns_matrix = np.where(states == 1, b_val, np.where(states == 2, 0., -1.0))

        def neg_expected_log_utility(f: np.ndarray) -> float:
            portfolio_returns = returns_matrix @ f
            if np.any(1.0 + portfolio_returns <= 1e-9):
                return 1e10
            return -float(np.sum(state_probs * np.log(1.0 + portfolio_returns)))

        def jac_neg_log_utility(f: np.ndarray) -> np.ndarray:
            denom = 1.0 + (returns_matrix @ f)
            weights = state_probs / denom
            return -(weights @ returns_matrix)

        raw_cap = min(1.0 - 1e-7, self.max_total_exposure / self.kelly_fraction)
        f0 = np.zeros(m)
        bounds = [(0.0, raw_cap) for _ in range(m)]
        constraints = ({'type': 'ineq', 'fun': lambda f: raw_cap - np.sum(f)})

        res = minimize(
            neg_expected_log_utility, f0, method='SLSQP',
            jac=jac_neg_log_utility, bounds=bounds, constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-12}
        )

        if not res.success:
            raise RuntimeError(f"Kelly optimization failed: {res.message}")
        optimal_f = np.maximum(0, res.x) * self.kelly_fraction
        if np.sum(optimal_f) > self.max_total_exposure:
            optimal_f *= (self.max_total_exposure / np.sum(optimal_f))

        final_stakes[valid_mask] = optimal_f

        return pd.DataFrame({
            "Selección": labels,
            "Prob_Modelo": np.round(p_arr, 4),
            "Cuota": o_arr,
            "EV_%": np.round(ev_arr * 100.0, 2),
            "Stake_%": final_stakes * 100.0
        })

    @staticmethod
    def simulate_portfolio_monte_carlo(
        probs: List[float], odds: List[float], stakes_mxn: List[float], n_sims: int = 10000, push_probs=None
    ) -> Dict[str, float]:
        """Simula 10,000 escenarios de la jornada para estimar E[PnL], P(Ganancia > 0) y VaR 95%."""
        p_vec, o_vec = probability_odds(probs, odds)
        w_vec = vector(stakes_mxn, "stakes", 0)
        pushes = np.zeros_like(p_vec) if push_probs is None else vector(push_probs, "push probabilities", 0, 1)
        MarketMechanics.calculate_ev(p_vec, o_vec, pushes)
        if w_vec.shape != p_vec.shape or n_sims < 1:
            raise ValueError("Invalid simulation dimensions")
        if len(p_vec) == 0 or np.sum(w_vec) == 0:
            return {"Expected_PnL_MXN": 0.0, "Prob_Profitable_Day_%": 0.0, "VaR_95_MXN": 0.0}

        rng = np.random.default_rng(2026)
        p_vec = np.asarray(probs, dtype=float)
        o_vec = np.asarray(odds, dtype=float)
        w_vec = np.asarray(stakes_mxn, dtype=float)

        uniform = rng.random((n_sims, len(p_vec)))
        net_pnl_matrix = np.where(uniform < p_vec, w_vec * (o_vec - 1.0), np.where(uniform < p_vec+pushes, 0., -w_vec))
        daily_pnl = np.sum(net_pnl_matrix, axis=1)

        return {
            "Expected_PnL_MXN": round(float(np.mean(daily_pnl)), 2),
            "Prob_Profitable_Day_%": round(float(np.mean(daily_pnl > 0) * 100.0), 2),
            "VaR_95_MXN": round(float(np.percentile(daily_pnl, 5)), 2)
        }


class ModelEvaluator:
    """Reglas de puntuación estrictas (RPS) y auditoría de Closing Line Value (CLV)."""

    @staticmethod
    def ranked_probability_score(probs: np.ndarray, actual_idx: np.ndarray) -> float:
        probs = np.atleast_2d(np.asarray(probs, dtype=float))
        n_samples, n_outcomes = probs.shape
        if n_samples == 0 or n_outcomes < 2:
            raise ValueError("RPS requires observations and at least two outcomes")
        for row in probs:
            simplex(row)
        actual_idx = np.atleast_1d(actual_idx)
        if len(actual_idx) != n_samples or np.any(actual_idx != actual_idx.astype(int)) or np.any((actual_idx < 0) | (actual_idx >= n_outcomes)):
            raise ValueError("Invalid actual outcomes")
        actual_idx = actual_idx.astype(int)
        actual_onehot = np.zeros_like(probs)
        actual_onehot[np.arange(n_samples), actual_idx] = 1.0

        cdf_pred = np.cumsum(probs, axis=1)[:, :-1]
        cdf_actual = np.cumsum(actual_onehot, axis=1)[:, :-1]

        rps_per_match = np.sum((cdf_pred - cdf_actual) ** 2, axis=1) / (n_outcomes - 1)
        return float(np.mean(rps_per_match))

    @staticmethod
    def closing_line_value(bet_odds: Union[float, np.ndarray], closing_fair_probs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        closing_fair_odds = 1.0 / np.asarray(closing_fair_probs, dtype=float)
        return (np.asarray(bet_odds, dtype=float) / closing_fair_odds) - 1.0
