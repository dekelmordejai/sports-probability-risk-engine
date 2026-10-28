import pytest
import numpy as np
import pandas as pd

from src.quant_core import MarketMechanics, PortfolioManager, ModelEvaluator
from src.soccer_dixon_coles import DixonColesEngine
from src.nfl_epa_kalman import NFLKalmanKeyNumberEngine
from src.nba_possession_engine import NBAPossessionEngine
from src.advanced_alpha_engines import (
    AmbiguityAverseKelly,
    PlayerPropsEngine,
    InPlayStochasticEngine,
    CopulaCorrelatedParlayPricer
)
from src.entity_resolver import EntityResolver


# ==============================================================================
# 1. INVARIANTES DE MICROESTRUCTURA DE MERCADO Y MÉTODO DE SHIN
# ==============================================================================

def test_shin_devig_simplex_and_favorite_longshot_bias():
    """
    Verifica que Shin proyecte al símplex de probabilidad exacto (suma = 1.0)
    y corrija el sesgo favorite-longshot (penalizando más la cuota alta que el método multiplicativo).
    """
    odds_1x2 = [1.45, 4.50, 7.50]  # Favorito claro vs Underdog
    probs_shin, z_val = MarketMechanics.devig_shin(odds_1x2)
    probs_mult = MarketMechanics.devig_multiplicative(odds_1x2)

    # Axioma 1: Suma exacta a 1.0 y probabilidades en (0, 1)
    assert np.isclose(np.sum(probs_shin), 1.0, atol=1e-9)
    assert np.all((probs_shin > 0.0) & (probs_shin < 1.0))
    assert 0.0 <= z_val < 1.0

    # Axioma 2: Shin asigna mayor probabilidad real al favorito y menor al longshot que el método proporcional
    assert probs_shin[0] > probs_mult[0], "Shin debe proteger la probabilidad del favorito."
    assert probs_shin[2] < probs_mult[2], "Shin debe descontar el sesgo de insiders en el longshot."


def test_corrupted_odds_rejection():
    """Garantiza que cualquier cuota corrupta <= 1.0 proveniente de la API dispare ValueError."""
    with pytest.raises(ValueError):
        MarketMechanics.devig_shin([1.90, 0.95])
    with pytest.raises(ValueError):
        MarketMechanics.devig_shin([1.00, 2.10, 3.40])


def test_shannon_entropy_bounds():
    """La entropía normalizada debe ser 1.0 en equiprobabilidad y cercana a 0 en certeza."""
    uniform_3way = [1/3, 1/3, 1/3]
    near_certain = [0.999, 0.0005, 0.0005]

    h_max = MarketMechanics.shannon_entropy(uniform_3way, normalized=True)
    h_min = MarketMechanics.shannon_entropy(near_certain, normalized=True)

    assert np.isclose(h_max, 1.0, atol=1e-6)
    assert 0.0 <= h_min < 0.02


# ==============================================================================
# 2. INVARIANTES DE GESTIÓN DE RIESGO Y KELLY SIMULTÁNEO / ROBUSTO
# ==============================================================================

def test_simultaneous_kelly_hard_exposure_cap():
    """
    Aunque existan 8 apuestas simultáneas con un +EV descomunal (>25%), la suma
    total de stakes jamás debe exceder max_total_exposure (ej. 15% de la banca).
    """
    max_cap = 0.15
    pm = PortfolioManager(kelly_fraction=0.25, max_total_exposure=max_cap, min_ev=0.025)

    # 6 apuestas con +EV masivo y 2 apuestas con EV negativo/insuficiente
    probs = [0.65, 0.62, 0.58, 0.70, 0.55, 0.60, 0.45, 0.50]
    odds  = [2.10, 2.05, 2.20, 1.95, 2.30, 2.15, 2.00, 2.02]  # Últimas dos tienen EV <= 1%

    df_alloc = pm.optimize_simultaneous_kelly(probs, odds)
    stakes_pct = df_alloc["Stake_%"].values

    # Axioma 1: Ninguna apuesta negativa o bajo umbral recibe un solo centavo
    assert stakes_pct[6] == 0.0
    assert stakes_pct[7] == 0.0

    # Axioma 2: No negatividad y respeto estricto del tope de exposición de portafolio
    assert np.all(stakes_pct >= 0.0)
    total_allocated_fraction = np.sum(stakes_pct) / 100.0
    assert total_allocated_fraction <= max_cap + 1e-5, (
        f"Violación de riesgo: Se asignó {total_allocated_fraction*100:.2f}% > {max_cap*100:.2f}%"
    )


def test_ambiguity_averse_kelly_monotonicity():
    """
    A mayor incertidumbre epistémica (sigma_p), el Criterio de Kelly Robusto
    debe contraer monótonamente el tamaño de la apuesta.
    """
    k_low_unc = AmbiguityAverseKelly.compute_robust_fraction(prob_mean=0.57, prob_std=0.010, decimal_odds=1.95)
    k_mid_unc = AmbiguityAverseKelly.compute_robust_fraction(prob_mean=0.57, prob_std=0.035, decimal_odds=1.95)
    k_high_unc = AmbiguityAverseKelly.compute_robust_fraction(prob_mean=0.57, prob_std=0.080, decimal_odds=1.95)

    assert k_low_unc["robust_kelly_%"] > k_mid_unc["robust_kelly_%"] >= k_high_unc["robust_kelly_%"]
    assert k_high_unc["robust_kelly_%"] == 0.0, "Con incertidumbre extrema (8%), Kelly Robusto debe abstenerse."


# ==============================================================================
# 3. INVARIANTES DE MOTORES ESTOCÁSTICOS (FÚTBOL, NFL Y NBA)
# ==============================================================================

def test_dixon_coles_joint_matrix_and_altitude_monotonicity():
    """
    Verifica que la matriz conjunta de goles de Dixon-Coles sume exactamente 1.0
    y que el factor fisiológico de altitud (Liga MX) incremente la probabilidad local.
    """
    rng = np.random.default_rng(42)
    teams = ["Toluca", "Mazatlan FC", "America", "UNAM"]
    rows = []
    for d in pd.date_range("2026-01-01", periods=40, freq="3D"):
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append({
            "date": d, "home_team": h, "away_team": a,
            "home_goals": int(rng.poisson(1.5)), "away_goals": int(rng.poisson(1.1)),
            "home_xg": 1.45, "away_xg": 1.10
        })
    df_sample = pd.DataFrame(rows)
    engine = DixonColesEngine(xi=0.003, xg_weight=0.65).fit(df_sample)

    pred_neutral = engine.predict_match_markets("Toluca", "Mazatlan FC", alt_boost=0.0)
    pred_altitude = engine.predict_match_markets("Toluca", "Mazatlan FC", alt_boost=0.18)

    # Axioma 1: Matriz conjunta de marcadores válida y mercados que suman 1.0
    assert np.all(pred_neutral["score_matrix"] >= 0.0)
    assert np.isclose(np.sum(pred_neutral["score_matrix"]), 1.0, atol=1e-7)
    assert np.isclose(np.sum(pred_neutral["1X2"]), 1.0, atol=1e-7)
    assert np.isclose(np.sum(pred_neutral["Over_Under_2.5"]), 1.0, atol=1e-7)

    # Axioma 2: Monotonía del boost de altitud sobre goles esperados y victoria local
    assert pred_altitude["lambda_home"] > pred_neutral["lambda_home"]
    assert pred_altitude["mu_away"] < pred_neutral["mu_away"]
    assert pred_altitude["1X2"][0] > pred_neutral["1X2"][0]


def test_nfl_kalman_covariance_contraction_and_key_numbers():
    """
    Verifica que el Filtro de Kalman reduzca la incertidumbre epistémica tras observar
    un partido y que la PMF de números clave concentre picos en 3 y 7 puntos.
    """
    nfl = NFLKalmanKeyNumberEngine()
    nfl.initialize_states(["KC", "BUF", "SF", "BAL"], prior_std=4.5)

    # Incertidumbre a priori antes de jugar en Semana 1
    _, std_before = nfl.step_predict_and_update("KC", "BUF", observed_y=None)
    # Actualización de Kalman con observación real
    nfl.step_predict_and_update("KC", "BUF", observed_y=6.5)
    # Incertidumbre a posteriori
    _, std_after = nfl.step_predict_and_update("KC", "BUF", observed_y=None)

    assert std_after < std_before, "La varianza epistémica del Filtro de Kalman debe contraerse tras observar datos."

    # Verificar resonancia en números clave 3 y 7 frente a márgenes adyacentes (2, 4 y 6, 8)
    margins, pmf = nfl.build_key_number_pmf(mu_spread=0.0, sigma_spread=13.2)
    m_to_p = dict(zip(margins, pmf))

    assert np.isclose(np.sum(pmf), 1.0, atol=1e-8)
    assert m_to_p[3] > m_to_p[2] and m_to_p[3] > m_to_p[4], "El margen +3 debe ser un pico multimodal."
    assert m_to_p[7] > m_to_p[6] and m_to_p[7] > m_to_p[8], "El margen +7 debe ser un pico multimodal."


def test_nba_fatigue_and_injury_monotonicity():
    """
    Verifica que un equipo jugando Back-to-Back o con una baja titular (EPM > 0)
    vea reducida su probabilidad proyectada de cubrir el spread.
    """
    nba = NBAPossessionEngine()
    df_logs = pd.DataFrame([
        {"date": "2026-10-10", "home_team": "DEN", "away_team": "LAL", "pace": 100.0,
         "home_ortg": 118.0, "away_ortg": 114.0, "home_pts": 118, "away_pts": 114},
        {"date": "2026-10-14", "home_team": "LAL", "away_team": "DEN", "pace": 99.0,
         "home_ortg": 115.0, "away_ortg": 116.0, "home_pts": 114, "away_pts": 115}
    ])
    nba.fit_historical_logs(df_logs)

    # Escenario A: Sin bajas
    p_healthy = nba.price_nba_markets("DEN", "LAL", "2026-10-15", bookie_spread_home=-5.5, bookie_total_line=225.5)
    # Escenario B: Baja de estrella local en DEN (EPM +6.5, 36 min)
    p_injured = nba.price_nba_markets(
        "DEN", "LAL", "2026-10-15", bookie_spread_home=-5.5, bookie_total_line=225.5,
        home_injuries=[{"name": "Star_Center", "epm": 6.5, "proj_mins": 36.0}]
    )

    assert p_injured["P_Home_Cover"] < p_healthy["P_Home_Cover"]
    assert np.isclose(p_healthy["P_Home_Cover"] + p_healthy["P_Away_Cover"], 1.0, atol=1e-6)


# ==============================================================================
# 4. INVARIANTES DE LAS 5 FRONTERAS Y RESOLUCIÓN DE ENTIDADES
# ==============================================================================

def test_copula_positive_correlation_boost():
    """Con correlación positiva (rho > 0), la probabilidad conjunta de la Cópula debe superar P(A)*P(B)."""
    res = CopulaCorrelatedParlayPricer.price_bivariate_sgp(
        p_leg1=0.55, p_leg2=0.55, physical_rho=0.45, bookie_sgp_odds=3.20
    )
    assert res["Copula_Joint_Prob"] > res["Indep_Prob"]
    assert res["Correlation_Boost_%"] > 0.0


def test_entity_resolver_canonical_mapping():
    """Verifica que las variantes de The-Odds-API resuelvan exactamente a los IDs de cada motor."""
    assert EntityResolver.resolve("Pumas UNAM", "Liga_MX") == "UNAM"
    assert EntityResolver.resolve("Wolverhampton Wanderers", "EPL") == "Wolves"
    assert EntityResolver.resolve("Atlético Madrid", "La_Liga") == "Ath Madrid"
    assert EntityResolver.resolve("Kansas City Chiefs", "NFL") == "KC"
    assert EntityResolver.resolve("Golden State Warriors", "NBA") == "GSW"

from src.parlay_builder import QuantitativeParlayBuilder


def test_humanized_stake_rounding_never_exceeds_cap():
    """El redondeo hacia abajo usa múltiplos de $50 MXN y respeta el tope."""
    assert PortfolioManager.humanize_stake_mxn(437.82, step=50.0) == 400.0
    assert PortfolioManager.humanize_stake_mxn(12.50, step=50.0) == 0.0
    assert PortfolioManager.humanize_stake_mxn(680.00, step=50.0, max_cap_mxn=500.0) == 500.0


def test_parlay_builder_compounded_ev_and_exposure_cap():
    """Verifica que los parlays generados respeten el tope del 3% de la banca y compongan +EV."""
    sample_singles = pd.DataFrame([
        {"Sport": "Soccer", "League": "Liga_MX", "Match": "Toluca vs Mazatlan FC",
         "Market": "1X2", "Selection": "Toluca", "Prob_Model": 0.64, "Odds": 1.72},
        {"Sport": "NFL", "League": "NFL", "Match": "SF @ KC",
         "Market": "Spread -2.5", "Selection": "KC -2.5", "Prob_Model": 0.57, "Odds": 1.91},
        {"Sport": "NBA", "League": "NBA", "Match": "LAL @ DEN",
         "Market": "Spread -6.5", "Selection": "DEN -6.5", "Prob_Model": 0.59, "Odds": 1.91}
    ])
    builder = QuantitativeParlayBuilder(max_parlay_bankroll_share=0.03, parlay_kelly_fraction=0.12, min_combined_ev=0.08)
    parlays_df = builder.generate_optimal_parlays(sample_singles, bankroll_mxn=25000.0, profit_boost_pct=0.05)

    assert not parlays_df.empty
    assert np.all(parlays_df["EV_Compuesto_%"].values >= 8.0)
    assert parlays_df["Stake_%"].sum() <= 3.0 + 1e-4
    assert np.all(parlays_df["Monto_MXN"].values % 50.0 == 0.0)
