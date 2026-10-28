import os
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.quant_core import MarketMechanics


class TacticalMarketAgent:
    """
    Detector de Microestructura en Tiempo Real (Steam Moves & Stale Lines),
    Traductor de Bajas/Rotaciones de Última Hora y Despachador de Alertas a Telegram.
    """

    def __init__(self, telegram_bot_token: Optional[str] = None, telegram_chat_id: Optional[str] = None):
        self.bot_token = telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")

    def detect_steam_and_stale_lines(
        self,
        match_name: str,
        selection_names: List[str],
        pinnacle_odds_t0: List[float],
        pinnacle_odds_t1: List[float],
        best_soft_odds_t1: List[float],
        best_soft_books: List[str],
        model_probs: List[float],
        steam_prob_threshold: float = 0.022
    ) -> List[Dict]:
        """
        Compara la probabilidad limpia de Shin en Pinnacle entre t_0 (apertura/mañana) y t_1 (actual).
        Si Pinnacle se mueve > 2.2% en probabilidad hacia una selección (Steam Move) que nuestro modelo
        también respalda, y una casa recreativa sigue ofreciendo una cuota vieja con +EV > 3.5%,
        genera una señal de ejecución inmediata (Stale Line).
        """
        fair_t0, _ = MarketMechanics.devig_shin(pinnacle_odds_t0)
        fair_t1, z_t1 = MarketMechanics.devig_shin(pinnacle_odds_t1)

        signals = []
        for idx, sel in enumerate(selection_names):
            prob_drift = float(fair_t1[idx] - fair_t0[idx])
            soft_odd = float(best_soft_odds_t1[idx])
            p_mod = float(model_probs[idx])

            # Probabilidad consensuada (70% nuestro modelo + 30% nueva línea Sharp de Pinnacle)
            posterior_prob = 0.70 * p_mod + 0.30 * float(fair_t1[idx])
            ev_vs_soft = (posterior_prob * soft_odd) - 1.0

            is_steam = prob_drift >= steam_prob_threshold
            if is_steam and ev_vs_soft >= 0.035:
                signals.append({
                    "Match": match_name,
                    "Selection": sel,
                    "Bookie_Stale": best_soft_books[idx],
                    "Stale_Odd": soft_odd,
                    "Pin_Odd_Dropped_To": pinnacle_odds_t1[idx],
                    "Steam_Prob_Jump_%": round(prob_drift * 100.0, 2),
                    "Posterior_Prob": round(posterior_prob, 4),
                    "EV_%": round(ev_vs_soft * 100.0, 2),
                    "Shin_z": round(z_t1, 4)
                })
        return signals

    @staticmethod
    def evaluate_soccer_lineup_shock(
        expected_starters_xova: List[float],
        confirmed_starters_xova: List[float]
    ) -> float:
        """
        Calcula el ajuste en escala logarítmica sobre la tasa de goles de Dixon-Coles
        60 minutos antes del silbatazo cuando se publican los 11 iniciales oficiales.
        Cada punto de déficit en la suma de contribución neta esperada se traduce en delta_log_lambda.
        """
        net_talent_diff = float(np.sum(confirmed_starters_xova) - np.sum(expected_starters_xova))
        return float(np.clip(0.08 * net_talent_diff, -0.35, 0.35))

    def send_telegram_flash_alert(self, signal: Dict, robust_stake_pct: float, bankroll_mxn: float):
        """Envía una notificación instantánea al móvil cuando aparece una línea rezagada (+EV + Steam)."""
        if not self.bot_token or not self.chat_id:
            return

        wager_mxn = (robust_stake_pct / 100.0) * bankroll_mxn
        msg = (
            f"⚡ *ALERTA CUANTITATIVA: STALE LINE + STEAM* ⚡\n\n"
            f"🏟 *Partido:* {signal['Match']}\n"
            f"🎯 *Selección:* `{signal['Selection']}`\n"
            f"🏦 *Casa Rezagada:* {signal['Bookie_Stale']} @ *{signal['Stale_Odd']}*\n"
            f"📉 *Pinnacle cayó a:* {signal['Pin_Odd_Dropped_To']} "
            f"(Steam +{signal['Steam_Prob_Jump_%']}%)\n"
            f"📊 *Prob. Posterior:* {signal['Posterior_Prob']*100:.1f}% | *EV:* +{signal['EV_%']}%\n"
            f"💰 *Stake Kelly Robusto:* {robust_stake_pct:.2f}% (*${wager_mxn:,.2f} MXN*)"
        )
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        response = requests.post(
            url,
            json={"chat_id": self.chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
