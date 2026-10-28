import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List
from src.quant_core import PortfolioManager
from src.validation import probability_odds, vector


class QuantitativeParlayBuilder:
    """
    Construye Parlays y estructuras Round Robin óptimas a partir de selecciones
    independientes con +EV verificado, clasificándolas por nivel de probabilidad
    y aplicando Kelly Fraccionario con penalización por varianza compuesta.
    """

    def __init__(
        self,
        max_parlay_bankroll_share: float = 0.03,
        parlay_kelly_fraction: float = 0.12,
        min_combined_ev: float = 0.08
    ):
        """
        :param max_parlay_bankroll_share: Tope máximo de la banca destinado a todos los parlays del día (ej. 3%).
        :param parlay_kelly_fraction: Multiplicador conservador de Kelly para parlays (0.12 = ~1/8 Kelly).
        :param min_combined_ev: Valor Esperado compuesto mínimo exigido a un parlay (ej. +8.0%).
        """
        vector([max_parlay_bankroll_share, parlay_kelly_fraction], "fractions", 0, 1)
        self.max_parlay_share = max_parlay_bankroll_share
        self.parlay_kelly = parlay_kelly_fraction
        self.min_combined_ev = min_combined_ev

    def generate_optimal_parlays(
        self,
        singles_ticket_df: pd.DataFrame,
        bankroll_mxn: float,
        profit_boost_pct: float = 0.0,
        round_step_mxn: float = 50.0
    ) -> pd.DataFrame:
        """
        Recibe el DataFrame de apuestas individuales independientes (1 por partido)
        y genera las mejores combinaciones de 2 y 3 selecciones clasificadas en:
        - 'Ancla Alta Probabilidad' (todas las patas con P >= 0.58)
        - 'Potenciador +EV Compuesto' (máximo Ratio de Sharpe multi-pata)
        """
        vector([bankroll_mxn, round_step_mxn], "bankroll and step", 0, strict_lower=True)
        vector([profit_boost_pct], "profit boost", 0)
        if singles_ticket_df.empty or len(singles_ticket_df) < 2:
            return pd.DataFrame()

        # Filtrar candidatas sólidas para parlays (Prob >= 48% y EV positivo)
        p, o = probability_odds(singles_ticket_df.Prob_Model, singles_ticket_df.Odds)
        candidates = singles_ticket_df[(p >= 0.48) & (p * o > 1)].copy()
        candidates = candidates.drop_duplicates(["League", "Event_ID" if "Event_ID" in candidates else "Match"]).head(24)
        if "Selection_ID" not in candidates:
            candidates["Selection_ID"] = [str(i) for i in candidates.index]
        if "Prob_Push" in candidates:
            candidates = candidates[candidates.Prob_Push == 0]
        if len(candidates) < 2:
            return pd.DataFrame()

        parlay_candidates = []
        rows = list(candidates.itertuples())

        # Evaluar todas las combinaciones independientes de K = 2 y K = 3 patas
        for k in [2, 3]:
            if len(rows) < k:
                continue
            for combo in combinations(rows, k):
                joint_prob = float(np.prod([r.Prob_Model for r in combo]))
                raw_comb_odds = float(np.prod([r.Odds for r in combo]))

                # Aplicar Profit Boost de casa de apuestas si existe (sobre la ganancia neta b)
                net_b = (raw_comb_odds - 1.0) * (1.0 + profit_boost_pct)
                eff_odds = 1.0 + net_b

                comb_ev = (joint_prob * eff_odds) - 1.0
                if comb_ev < self.min_combined_ev:
                    continue

                # Clasificación por régimen de probabilidad
                min_leg_prob = min(r.Prob_Model for r in combo)
                if min_leg_prob >= 0.58 and joint_prob >= 0.34:
                    tier = "Ancla Alta Prob"
                else:
                    tier = "Compuesto +EV"

                # Kelly Fraccionario para la combinación
                q_joint = 1.0 - joint_prob
                f_raw = max(0.0, (joint_prob * net_b - q_joint) / net_b)
                f_scaled = f_raw * self.parlay_kelly

                # Ratio de Sharpe del Parlay: EV / sqrt(p * (1-p) * eff_odds^2)
                parlay_std = float(np.sqrt(joint_prob * q_joint * (eff_odds ** 2)))
                sharpe_parlay = comb_ev / parlay_std if parlay_std > 0 else 0.0

                legs_str = " + ".join([f"[{r.League}] {r.Selection} (@{r.Odds})" for r in combo])
                parlay_candidates.append({
                    "Leg_IDs": [str(r.Selection_ID) for r in combo],
                    "Legs": [{"League": r.League, "Match": r.Match, "Market": r.Market,
                              "Selection": r.Selection, "Odds": r.Odds, "Prob_Model": r.Prob_Model,
                              "Event_ID": getattr(r, "Event_ID", "")} for r in combo],
                    "Profit_Boost": profit_boost_pct,
                    "Tipo_Parlay": f"{tier} ({k} Patas)",
                    "Selecciones_Combinadas": legs_str,
                    "Num_Patas": k,
                    "Prob_Conjunta_%": round(joint_prob * 100.0, 2),
                    "Cuota_Combinada": eff_odds,
                    "EV_Compuesto_%": round(comb_ev * 100.0, 2),
                    "Sharpe_Parlay": sharpe_parlay,
                    "Raw_Stake_Frac": f_scaled
                })

        if not parlay_candidates:
            return pd.DataFrame()

        df_parlays = (
            pd.DataFrame(parlay_candidates)
            .sort_values("Sharpe_Parlay", ascending=False)
            .head(3)  # Conservar únicamente el Top 3 de Parlays del día para evitar sobre-exposición
            .reset_index(drop=True)
        )

        # Normalizar stakes para que la suma de los parlays nunca supere max_parlay_share (ej. 3% de la banca)
        total_raw_frac = df_parlays["Raw_Stake_Frac"].sum()
        if total_raw_frac > self.max_parlay_share:
            df_parlays["Raw_Stake_Frac"] *= (self.max_parlay_share / total_raw_frac)

        df_parlays["Stake_%"] = np.round(df_parlays["Raw_Stake_Frac"] * 100.0, 3)

        # Redondeo monetario hacia abajo para respetar la exposición máxima.
        raw_mxn = df_parlays["Raw_Stake_Frac"] * bankroll_mxn
        df_parlays["Monto_MXN"] = [PortfolioManager.humanize_stake_mxn(x, round_step_mxn) for x in raw_mxn]
        df_parlays["Stake_%"] = df_parlays.Monto_MXN / bankroll_mxn * 100
        df_parlays = df_parlays[df_parlays.Monto_MXN > 0].copy()
        df_parlays["Ganancia_Potencial_MXN"] = np.round(
            df_parlays["Monto_MXN"] * (df_parlays["Cuota_Combinada"] - 1.0), 2
        )

        cols = [
            "Tipo_Parlay", "Selecciones_Combinadas", "Prob_Conjunta_%",
            "Cuota_Combinada", "EV_Compuesto_%", "Stake_%", "Monto_MXN", "Ganancia_Potencial_MXN",
            "Leg_IDs", "Legs", "Profit_Boost"
        ]
        return df_parlays[cols]
