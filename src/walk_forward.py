import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from src.quant_core import MarketMechanics, ModelEvaluator, PortfolioManager
from src.soccer_dixon_coles import DixonColesEngine
from src.pi_ratings import PiRatingSystem

class WalkForwardBacktester:
    """
    Evaluación por días con entrenamiento anterior a cada bloque de predicción.
    Requiere datos y cuotas disponibles en las fechas indicadas; no verifica su procedencia.
    Re-entrena periódicamente Dixon-Coles + Pi-Ratings y gestiona la banca con Kelly Simultáneo.
    """

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        warmup_matches: int = 180,
        retrain_Step_days: int = 7,
        pi_blend_weight: float = 0.12,
        max_entropy: float = 0.985,
        kelly_fraction: float = 0.20,
        max_exposure: float = 0.12,
        min_ev: float = 0.03
    ):
        self.initial_bankroll = initial_bankroll
        self.warmup_matches = warmup_matches
        self.retrain_step_days = retrain_Step_days
        self.pi_blend_weight = pi_blend_weight
        self.max_entropy = max_entropy
        self.pm = PortfolioManager(
            kelly_fraction=kelly_fraction,
            max_total_exposure=max_exposure,
            min_ev=min_ev
        )

    def run(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        if len(df_raw) <= self.warmup_matches:
            raise ValueError("Insufficient history for the configured warm-up")
        # Calcular Pi-Ratings sin incorporar resultados del mismo día.
        pi_system = PiRatingSystem(lambda_rate=0.035, gamma_transfer=0.70)
        df = pi_system.compute_historical_features(df_raw)

        bankroll = self.initial_bankroll
        peak_bankroll = self.initial_bankroll

        all_predictions = []
        bet_ledger = []

        # Definir fecha de inicio tras el periodo de calentamiento (Warm-up)
        start_date = df.iloc[self.warmup_matches]["date"].normalize()
        end_date = df["date"].max().normalize()
        current_date = start_date

        dc_model = DixonColesEngine(xi=0.0032, xg_weight=0.65)
        last_fit = None

        while current_date <= end_date:
            next_date = current_date + pd.Timedelta(days=1)

            # Conjunto de entrenamiento estrictamente anterior a current_date
            train_df = df[df["date"] < current_date]
            # Ventana de prueba (jornada/semana actual)
            test_df = df[(df["date"] >= current_date) & (df["date"] < next_date)]

            if len(test_df) == 0:
                current_date = next_date
                continue

            # Re-entrenar el motor estocástico con los datos disponibles hasta hoy
            if last_fit is None or (current_date-last_fit).days >= self.retrain_step_days:
                dc_model.fit(train_df, current_date=str(current_date.date()))
                last_fit = current_date
            known_teams = set(dc_model.teams)

            batch_candidates = []

            for row in test_df.itertuples():
                if row.home_team not in known_teams or row.away_team not in known_teams:
                    continue

                # Ajuste híbrido: combinar altitud + corrección de Pi-Rating en escala log
                pi_boost = self.pi_blend_weight * np.tanh(row.pi_expected_gd / 2.0)
                total_boost = row.alt_boost + pi_boost

                pred = dc_model.predict_match_markets(row.home_team, row.away_team, alt_boost=total_boost)
                probs_model = pred["1X2"]

                odds_open = np.array([row.odds_h, row.odds_d, row.odds_a], dtype=float)
                odds_close = np.array([row.close_h, row.close_d, row.close_a], dtype=float)

                # Desmarginalización de Shin de apertura y cierre
                fair_open, z_open = MarketMechanics.devig_shin(odds_open)
                fair_close, _ = MarketMechanics.devig_shin(odds_close)
                entropy = MarketMechanics.shannon_entropy(fair_open, normalized=True)

                all_predictions.append({
                    "date": row.date,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "probs_model": probs_model,
                    "probs_market": fair_open,
                    "actual_idx": row.actual_idx
                })

                # Filtro de Caos por Entropía de Shannon
                if entropy > self.max_entropy:
                    continue

                ev_vec = MarketMechanics.calculate_ev(probs_model, odds_open)
                best_idx = int(np.argmax(ev_vec))

                if ev_vec[best_idx] >= self.pm.min_ev:
                    clv = ModelEvaluator.closing_line_value(odds_open[best_idx], fair_close[best_idx])
                    batch_candidates.append({
                        "date": row.date,
                        "match": f"{row.home_team} vs {row.away_team}",
                        "pick_idx": best_idx,
                        "pick_name": ["Local", "Empate", "Visita"][best_idx],
                        "prob_model": probs_model[best_idx],
                        "odds_taken": odds_open[best_idx],
                        "fair_close_prob": fair_close[best_idx],
                        "clv_pct": float(clv) * 100.0,
                        "actual_idx": row.actual_idx,
                        "entropy": entropy,
                        "z_shin": z_open
                    })

            # Si hay apuestas con +EV en la ventana, optimizar portafolio con Kelly Simultáneo
            if batch_candidates:
                c_probs = [c["prob_model"] for c in batch_candidates]
                c_odds = [c["odds_taken"] for c in batch_candidates]
                c_labels = [c["match"] for c in batch_candidates]

                kelly_df = self.pm.optimize_simultaneous_kelly(c_probs, c_odds, c_labels)
                stakes_pct = kelly_df["Stake_%"].values / 100.0

                bankroll_snapshot = bankroll
                for cand, stake_f in zip(batch_candidates, stakes_pct):
                    if stake_f <= 0.0005:
                        continue
                    wager = bankroll_snapshot * stake_f
                    won = int(cand["pick_idx"] == cand["actual_idx"])
                    pnl = wager * (cand["odds_taken"] - 1.0) if won else -wager
                    bankroll += pnl
                    peak_bankroll = max(peak_bankroll, bankroll)
                    drawdown_pct = ((peak_bankroll - bankroll) / peak_bankroll) * 100.0

                    bet_ledger.append({
                        **cand,
                        "stake_pct": round(stake_f * 100.0, 2),
                        "wager": round(wager, 2),
                        "won": won,
                        "pnl": round(pnl, 2),
                        "bankroll": round(bankroll, 2),
                        "drawdown_pct": round(drawdown_pct, 2)
                    })

            current_date = next_date

        # Consolidar métricas globales de evaluación
        preds_df = pd.DataFrame(all_predictions)
        ledger_df = pd.DataFrame(bet_ledger)

        if preds_df.empty:
            raise ValueError("No out-of-sample predictions: check teams and warm-up")

        probs_mat = np.vstack(preds_df["probs_model"].values)
        market_mat = np.vstack(preds_df["probs_market"].values)
        actual_vec = preds_df["actual_idx"].values

        rps_model = ModelEvaluator.ranked_probability_score(probs_mat, actual_vec)
        rps_market = ModelEvaluator.ranked_probability_score(market_mat, actual_vec)

        total_staked = ledger_df["wager"].sum() if not ledger_df.empty else 0.0
        total_pnl = ledger_df["pnl"].sum() if not ledger_df.empty else 0.0
        yield_pct = (total_pnl / total_staked * 100.0) if total_staked > 0 else 0.0

        summary = {
            "Partidos_Evaluados": len(preds_df),
            "RPS_Modelo": round(rps_model, 5),
            "RPS_Mercado_Shin": round(rps_market, 5),
            "Total_Apuestas": len(ledger_df),
            "Hit_Rate_%": round(ledger_df["won"].mean() * 100.0, 2) if not ledger_df.empty else 0.0,
            "CLV_Promedio_%": round(ledger_df["clv_pct"].mean(), 2) if not ledger_df.empty else 0.0,
            "Yield_%": round(yield_pct, 2),
            "ROI_Banca_%": round((bankroll - self.initial_bankroll) / self.initial_bankroll * 100.0, 2),
            "Max_Drawdown_%": round(ledger_df["drawdown_pct"].max(), 2) if not ledger_df.empty else 0.0,
            "Banca_Final": round(bankroll, 2)
        }

        return ledger_df, preds_df, summary
