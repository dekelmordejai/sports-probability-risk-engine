import numpy as np
import pandas as pd
from scipy.stats import nbinom, gamma, norm, poisson, multivariate_normal
from scipy.integrate import quad
from typing import Dict, List, Tuple
from src.validation import vector, probability_odds


class AmbiguityAverseKelly:
    """
    Frontera 1: Criterio de Kelly Robusto penalizado por Incertidumbre Epistémica (sigma_p).
    Reduce automáticamente el stake cuando la varianza de estimación del modelo es alta.
    """

    @staticmethod
    def compute_robust_fraction(
        prob_mean: float,
        prob_std: float,
        decimal_odds: float,
        kelly_fraction: float = 0.25,
        ambiguity_aversion: float = 1.5,
        max_cap: float = 0.05
    ) -> Dict[str, float]:
        """
        :param prob_mean: Probabilidad esperada por el modelo (p_hat).
        :param prob_std: Error estándar epistémico de la probabilidad (sigma_p).
        :param decimal_odds: Cuota decimal disponible.
        :param ambiguity_aversion: Coeficiente de aversión a la ambigüedad del parámetro.
        """
        probability_odds([prob_mean], [decimal_odds])
        vector([prob_std, ambiguity_aversion], "uncertainty", 0)
        vector([kelly_fraction, max_cap], "fractions", 0, 1)
        b = decimal_odds - 1.0
        if b <= 0 or prob_mean <= 0 or prob_mean >= 1:
            return {"raw_kelly_%": 0.0, "robust_kelly_%": 0.0, "uncertainty_haircut_%": 0.0}

        q = 1.0 - prob_mean
        f_classical = (prob_mean * b - q) / b
        if f_classical <= 0:
            return {"raw_kelly_%": 0.0, "robust_kelly_%": 0.0, "uncertainty_haircut_%": 0.0}

        # Factor de contracción analítico por varianza epistémica sigma_p^2
        bernoulli_var = prob_mean * q
        penalty_ratio = (ambiguity_aversion * ((1.0 + b) ** 2) * (prob_std ** 2)) / (bernoulli_var * (b ** 2))
        shrinkage_factor = max(0.0, 1.0 - penalty_ratio)

        # Worst-case probability in a one-sided ambiguity interval.
        conservative_p = max(0.0, prob_mean - ambiguity_aversion * prob_std)
        f_robust = max(0.0, (conservative_p * decimal_odds - 1) / b) * kelly_fraction
        f_classical_scaled = f_classical * kelly_fraction

        haircut = (1.0 - f_robust / f_classical_scaled) * 100.0
        return {
            "raw_kelly_%": round(min(f_classical_scaled, max_cap) * 100.0, 3),
            "robust_kelly_%": round(min(f_robust, max_cap) * 100.0, 3),
            "uncertainty_haircut_%": round(haircut, 2)
        }


class PlayerPropsEngine:
    """
    Frontera 2: Motor de Proposiciones de Jugadores (Player Props)
    - NBA / Fútbol (Conteos discretos sobredispersos): Distribución Binomial Negativa.
    - NFL / NBA (Yardas o Puntos continuos): Distribución Gamma condicionada al Game Script.
    """

    @staticmethod
    def price_discrete_prop_nbinom(
        proj_minutes: float,
        base_rate_per_min: float,
        pace_multiplier: float = 1.0,
        usage_vacuum_boost: float = 0.0,
        overdispersion_alpha: float = 0.12,
        bookie_line: float = 7.5,
        over_odds: float = 1.91,
        under_odds: float = 1.91
    ) -> Dict:
        """
        Valora mercados discretos (Rebotes, Asistencias, Triples, Remates a Puerta).
        usage_vacuum_boost > 0 refleja el incremento de rol cuando un compañero titular es baja.
        """
        vector([proj_minutes, base_rate_per_min, pace_multiplier, overdispersion_alpha], "prop inputs", 0)
        vector([usage_vacuum_boost], "usage boost", -1)
        probability_odds([.5, .5], [over_odds, under_odds])
        mu = proj_minutes * base_rate_per_min * pace_multiplier * (1.0 + usage_vacuum_boost)
        variance = mu + overdispersion_alpha * (mu ** 2)

        # Conversión de momentos (mu, var) a parámetros (n, p) de scipy.stats.nbinom
        distribution = poisson(mu) if overdispersion_alpha == 0 or mu == 0 else nbinom(1 / overdispersion_alpha, 1 / (1 + overdispersion_alpha * mu))

        # Probabilidad exacta de superar la línea (ej. > 7.5 equivale a >= 8)
        threshold_k = int(np.floor(bookie_line))
        p_over = float(distribution.sf(threshold_k))
        p_push = float(distribution.pmf(bookie_line)) if float(bookie_line).is_integer() else 0.
        p_under = max(0., 1.0 - p_over - p_push)

        return {
            "Expected_Mean": round(mu, 2),
            "Std_Dev": round(float(np.sqrt(variance)), 2),
            "P_Push": p_push,
            "Line": bookie_line,
            "P_Over": round(p_over, 4),
            "P_Under": round(p_under, 4),
            "EV_Over_%": round(((p_over * over_odds) + p_push - 1.0) * 100.0, 2),
            "EV_Under_%": round(((p_under * under_odds) + p_push - 1.0) * 100.0, 2)
        }

    @staticmethod
    def price_continuous_yards_gamma(
        base_yards_mean: float,
        yards_cv: float = 0.42,
        kalman_spread_home: float = 0.0,
        is_passing_or_receiving: bool = True,
        is_home_player: bool = True,
        bookie_line: float = 64.5,
        over_odds: float = 1.91,
        under_odds: float = 1.91
    ) -> Dict:
        """
        Valora yardas de jugadores en la NFL mediante una Distribución Gamma ajustada por Game Script.
        Si se proyecta que el equipo irá perdiendo (spread positivo para ese lado), aumenta el volumen aéreo
        y disminuye el volumen de acarreos terrestres.
        """
        vector([base_yards_mean, yards_cv], "gamma parameters", 0, strict_lower=True)
        probability_odds([.5, .5], [over_odds, under_odds])
        team_expected_deficit = kalman_spread_home if is_home_player else -kalman_spread_home
        # Cada punto de desventaja proyectada incrementa el volumen de pase en ~1.2% y reduce carrera en ~1.4%
        script_mult = 1.0 + (0.012 * team_expected_deficit if is_passing_or_receiving else -0.014 * team_expected_deficit)
        script_mult = float(np.clip(script_mult, 0.75, 1.30))

        adj_mean = base_yards_mean * script_mult
        adj_std = adj_mean * yards_cv

        shape_k = (adj_mean / adj_std) ** 2
        scale_theta = (adj_std ** 2) / adj_mean

        p_under = float(gamma.cdf(bookie_line, a=shape_k, scale=scale_theta))
        p_over = 1.0 - p_under

        return {
            "GameScript_Multiplier": round(script_mult, 3),
            "Adj_Mean_Yards": round(adj_mean, 1),
            "Line": bookie_line,
            "P_Over": round(p_over, 4),
            "P_Under": round(p_under, 4),
            "EV_Over_%": round(((p_over * over_odds) - 1.0) * 100.0, 2),
            "EV_Under_%": round(((p_under * under_odds) - 1.0) * 100.0, 2)
        }


class InPlayStochasticEngine:
    """
    Frontera 3: Motor de Valoración en Tiempo Real (In-Play).
    - NBA / NFL: Puente Browniano con deriva ajustada por regresión a la media de tiro.
    - Fútbol: Proceso de nacimiento de Markov (Dixon-Robinson) con ajuste por tarjetas rojas.
    """

    @staticmethod
    def price_live_diffusion_nba_nfl(
        pregame_fair_spread_home: float,
        pregame_std: float,
        current_home_score: int,
        current_away_score: int,
        fraction_time_elapsed: float,
        live_bookie_spread_home: float,
        unsustainable_shooting_Luck_pts: float = 0.0
    ) -> Dict:
        """
        Calcula la distribución condicional del margen final en el tiempo restante (1 - t).
        :param unsustainable_shooting_Luck_pts: Puntos anotados por encima de la expectativa de calidad de tiro (xFG / EPA)
               por el equipo local. Se descuenta ligeramente por regresión a la media táctica.
        """
        vector([fraction_time_elapsed], "elapsed fraction", 0, 1)
        vector([pregame_std], "standard deviation", 0, strict_lower=True)
        vector([current_home_score, current_away_score], "scores", 0)
        rem_time = 1.0 - fraction_time_elapsed
        current_margin = float(current_home_score - current_away_score)

        # Deriva pre-partido (Local - Visita) = -pregame_fair_spread_home
        pregame_drift = -pregame_fair_spread_home
        # Ajuste táctico (efecto goma elástica / urgencia del equipo que va perdiendo)
        rubber_band_adj = -0.08 * current_margin * rem_time - (0.25 * unsustainable_shooting_Luck_pts * rem_time)

        expected_final_margin = current_margin + rem_time * pregame_drift + rubber_band_adj
        remaining_std = pregame_std * np.sqrt(rem_time)

        # Probabilidad de cubrir el Spread en vivo: Final_Margin + live_bookie_spread_home > 0
        target_margin = -live_bookie_spread_home
        if rem_time == 0:
            p_home_cover_live = float(current_margin > target_margin)
            p_home_win_live = float(current_margin > 0)
            p_push = float(current_margin == target_margin)
        else:
            p_home_cover_live = float(norm.sf(target_margin, loc=expected_final_margin, scale=remaining_std))
            p_home_win_live = float(norm.sf(0.0, loc=expected_final_margin, scale=remaining_std))
            p_push = 0.

        return {
            "Current_Margin": current_margin,
            "Remaining_Time_%": round(rem_time * 100.0, 1),
            "Live_Fair_Spread_Home": round(-expected_final_margin, 2),
            "Live_Remaining_Std": round(remaining_std, 2),
            "P_Live_Home_Cover": round(p_home_cover_live, 4),
            "P_Live_Away_Cover": round(1.0 - p_home_cover_live - p_push, 4),
            "P_Push": p_push,
            "P_Live_Home_ML": round(p_home_win_live, 4)
        }

    @staticmethod
    def price_live_soccer_dixon_robinson(
        pregame_lambda_h: float,
        pregame_mu_a: float,
        minute_elapsed: float,
        current_goals_h: int,
        current_goals_a: int,
        red_cards_h: int = 0,
        red_cards_a: int = 0
    ) -> Dict:
        """
        Actualiza en vivo las probabilidades 1X2 y Over/Under en fútbol considerando
        el desgaste del reloj, urgencia por estado del marcador y tarjetas rojas.
        """
        vector([pregame_lambda_h, pregame_mu_a, minute_elapsed, current_goals_h, current_goals_a, red_cards_h, red_cards_a], "soccer inputs", 0)
        rem_ratio = float(np.clip((92.0 - minute_elapsed) / 92.0, 0.0, 1.0))
        score_diff = current_goals_h - current_goals_a

        # Modificadores de intensidad de ataque de Dixon-Robinson según el marcador actual
        if score_diff == 1:
            urgency_h, urgency_a = 0.88, 1.18
        elif score_diff == -1:
            urgency_h, urgency_a = 1.22, 0.85
        elif abs(score_diff) >= 2:
            urgency_h, urgency_a = 0.85, 0.90
        else:
            urgency_h, urgency_a = 1.0, 1.0

        # Penalización/Impulso exponencial por tarjetas rojas
        red_diff = red_cards_a - red_cards_h
        lam_rem = pregame_lambda_h * (rem_ratio ** 0.92) * urgency_h * np.exp(0.38 * red_diff)
        mu_rem = pregame_mu_a * (rem_ratio ** 0.92) * urgency_a * np.exp(-0.38 * red_diff)

        max_rem_goals = 8
        g_grid = np.arange(max_rem_goals + 1)
        mat_rem = np.outer(poisson.pmf(g_grid, lam_rem), poisson.pmf(g_grid, mu_rem))
        mat_rem /= np.sum(mat_rem)

        # Desplazar matriz por los goles ya anotados
        i_add, j_add = np.meshgrid(g_grid, g_grid, indexing="ij")
        final_h = current_goals_h + i_add
        final_a = current_goals_a + j_add

        p_h = float(np.sum(mat_rem[final_h > final_a]))
        p_d = float(np.sum(mat_rem[final_h == final_a]))
        p_a = float(np.sum(mat_rem[final_h < final_a]))

        return {
            "Minute": minute_elapsed,
            "Rem_Lambda_H": round(float(lam_rem), 3),
            "Rem_Mu_A": round(float(mu_rem), 3),
            "Live_1X2": np.array([p_h, p_d, p_a])
        }


class CopulaCorrelatedParlayPricer:
    """
    Frontera 5: Valoración de Same-Game Parlays (SGP) mediante Cópula Gaussiana Bivariada.
    Detecta cuando una casa subestima la correlación positiva entre dos mercados de un mismo partido.
    """

    @staticmethod
    def price_bivariate_sgp(p_leg1: float, p_leg2: float, physical_rho: float, bookie_sgp_odds: float) -> Dict:
        """
        Calcula P(Leg_1 n Leg_2) transformando las marginales al espacio normal estándar:
        z_i = Phi^{-1}(1 - p_i), e integrando la cola conjunta con correlación `physical_rho`.
        """
        vector([p_leg1, p_leg2], "leg probabilities", 0, 1, strict_lower=True)
        if p_leg1 == 1 or p_leg2 == 1:
            joint_prob = min(p_leg1, p_leg2)
        vector([physical_rho], "correlation", -1, 1)
        probability_odds([p_leg1], [bookie_sgp_odds])
        z1 = float(norm.ppf(p_leg1))
        z2 = float(norm.ppf(p_leg2))

        cov_matrix = np.array([[1.0, physical_rho], [physical_rho, 1.0]])
        if physical_rho == 1:
            joint_prob = min(p_leg1, p_leg2)
        elif physical_rho == -1:
            joint_prob = max(0., p_leg1 + p_leg2 - 1)
        elif p_leg1 < 1 and p_leg2 < 1:
            joint_prob = float(quad(lambda x: norm.pdf(x)*norm.cdf((z2-physical_rho*x)/np.sqrt(1-physical_rho**2)),
                                    -np.inf, z1, epsabs=1e-10)[0])
        indep_prob = p_leg1 * p_leg2

        ev_pct = ((joint_prob * bookie_sgp_odds) - 1.0) * 100.0
        return {
            "Indep_Prob": round(indep_prob, 4),
            "Copula_Joint_Prob": round(joint_prob, 4),
            "Correlation_Boost_%": round(((joint_prob / indep_prob) - 1.0) * 100.0, 2),
            "Fair_SGP_Odds": round(1.0 / joint_prob, 2) if joint_prob > 0 else 999.0,
            "Bookie_SGP_Odds": bookie_sgp_odds,
            "EV_%": round(ev_pct, 2)
        }
