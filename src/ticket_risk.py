"""Monte Carlo reuses leg outcomes in singles and overlapping parlays."""
import numpy as np
from src.validation import probability_odds
from src.quant_core import MarketMechanics


def simulate_ticket_risk(singles, parlays, n_sims=10000):
    pnl = np.zeros(n_sims)
    if not singles.empty:
        p, o = probability_odds(singles.Prob_Model, singles.Odds)
        pushes = singles.Prob_Push.to_numpy() if "Prob_Push" in singles else np.zeros_like(p)
        MarketMechanics.calculate_ev(p, o, pushes)
        uniform = np.random.default_rng(2026).random((n_sims, len(p)))
        gross = np.where(uniform < p, o, np.where(uniform < p+pushes, 1., 0.))
        pnl = ((gross-1)*singles.Monto_MXN.to_numpy()).sum(axis=1)
        id_to_col = {str(k): i for i, k in enumerate(singles.Selection_ID)}
        for row in parlays.to_dict("records"):
            cols = [id_to_col[k] for k in row["Leg_IDs"]]
            raw_return = gross[:, cols].prod(axis=1)
            eff_return = np.where(raw_return > 1, 1+(raw_return-1)*(1+row["Profit_Boost"]), raw_return)
            pnl += (eff_return-1)*row["Monto_MXN"]
    return {"Expected_PnL_MXN": round(float(np.mean(pnl)), 2),
            "Prob_Profitable_Day_%": round(float(np.mean(pnl > 0)*100), 2),
            "VaR_95_MXN": round(float(np.percentile(pnl, 5)), 2),
            "Assumption": "Independent events; shared outcomes for overlapping tickets. VaR is signed PnL percentile."}
