"""Portfolio construction: one selection per event and a shared exposure budget."""
import hashlib
import numpy as np
import pandas as pd
from src.quant_core import MarketMechanics, PortfolioManager
from src.parlay_builder import QuantitativeParlayBuilder
from src.ticket_risk import simulate_ticket_risk
from src.validation import probability_odds, simplex, vector


class MultiSportMasterOrchestrator:
    def __init__(self, bankroll=25000., kelly_fraction=.25, max_portfolio_exposure=.15,
                 min_ev_threshold=.025, max_normalized_entropy=.990,
                 max_parlay_exposure=.03, round_step_mxn=50., league_multipliers=None,
                 paper_mode=False):
        vector([bankroll, round_step_mxn], "bankroll and step", 0, strict_lower=True)
        vector([max_parlay_exposure], "parlay exposure", 0, max_portfolio_exposure)
        self.bankroll, self.max_entropy = bankroll, max_normalized_entropy
        self.pm = PortfolioManager(kelly_fraction, max_portfolio_exposure, min_ev_threshold, round_step_mxn)
        self.max_parlay_exposure, self.step = max_parlay_exposure, round_step_mxn
        self.multipliers = league_multipliers or {}
        vector(list(self.multipliers.values()), "league multipliers", 0, 1)
        self.paper_mode = paper_mode
        self.raw_opportunities = []

    def register_market_opportunity(self, sport, league, match_id, market_type,
                                    selection_names, model_probs, bookie_odds,
                                    sharp_odds=None, best_books=None, event_id=None,
                                    commence_time=None, push_probability=0.):
        p, o = probability_odds(model_probs, bookie_odds)
        vector([push_probability], "push probability", 0, .999999)
        conditional = p / (1-push_probability)
        simplex(conditional)
        if len(selection_names) != len(p):
            raise ValueError("Selection count does not match market")
        fair, z = MarketMechanics.devig_shin(sharp_odds if sharp_odds is not None else o)
        if len(fair) != len(p):
            raise ValueError("Sharp market dimensions mismatch")
        entropy = MarketMechanics.shannon_entropy(conditional)
        if entropy > self.max_entropy:
            return
        books = best_books or [""]*len(p)
        if len(books) != len(p):
            raise ValueError("Bookmaker dimensions mismatch")
        for name, prob, odd, mkt, book in zip(selection_names, p, o, fair, books):
            ev = prob*odd+push_probability-1
            if ev < self.pm.min_ev:
                continue
            identifier = f"{league}|{event_id or match_id}|{market_type}|{name}"
            self.raw_opportunities.append({
                "Sport": sport, "League": league, "Match": match_id, "Market": market_type,
                "Selection": name, "Selection_ID": hashlib.sha256(identifier.encode()).hexdigest()[:20],
                "Event_ID": event_id or match_id, "Commence_Time": commence_time or "", "Bookmaker": book,
                "Prob_Model": float(prob), "Prob_Shin_Mkt": float(mkt), "Odds": float(odd),
                "Prob_Push": push_probability,
                "EV_raw": float(ev), "Shin_z": float(z), "Entropy": float(entropy),
                "Sharpe_Edge": float(ev/max(np.sqrt(max(0., prob*(odd-1)**2+(1-prob-push_probability)-ev**2)), 1e-12))})

    def build_optimal_weekend_ticket(self, profit_boost_pct=0.):
        if not self.raw_opportunities:
            empty = pd.DataFrame()
            return {"singles_ticket_df": empty, "parlays_ticket_df": empty.copy(),
                    "risk_report": simulate_ticket_risk(empty, empty), "paper_mode": self.paper_mode}
        singles = (pd.DataFrame(self.raw_opportunities).sort_values("Sharpe_Edge", ascending=False)
                   .drop_duplicates(["League", "Event_ID"]).reset_index(drop=True))
        allocations = self.pm.optimize_simultaneous_kelly(singles.Prob_Model.tolist(), singles.Odds.tolist(), push_probs=singles.Prob_Push.tolist())
        multipliers = singles.League.map(lambda lg: self.multipliers.get(lg, 1.)).to_numpy()
        raw = allocations["Stake_%"].to_numpy()/100*self.bankroll*multipliers
        singles["Monto_MXN"] = [self.pm.humanize_stake_mxn(x, self.step) for x in raw]
        singles["EV_%"] = singles.EV_raw*100
        singles["Edge_vs_Shin_%"] = (singles.Prob_Model/(1-singles.Prob_Push)-singles.Prob_Shin_Mkt)*100
        singles["Stake_%"] = singles.Monto_MXN/self.bankroll*100
        # Reduced leagues also reduce each parlay's risk allowance.
        builder = QuantitativeParlayBuilder(self.max_parlay_exposure)
        eligible = singles[multipliers > 0].copy()
        parlays = builder.generate_optimal_parlays(eligible, self.bankroll, profit_boost_pct, self.step)
        if not parlays.empty:
            id_multiplier = dict(zip(singles.Selection_ID, multipliers))
            parlays["Monto_MXN"] = [self.pm.humanize_stake_mxn(r.Monto_MXN*min(id_multiplier[x] for x in r.Leg_IDs), self.step)
                                    for r in parlays.itertuples()]
            parlays = parlays[parlays.Monto_MXN > 0].copy()
        parlay_total = float(parlays.Monto_MXN.sum()) if not parlays.empty else 0.
        available = self.pm.max_total_exposure*self.bankroll-parlay_total
        total = float(singles.Monto_MXN.sum())
        if total > available:
            singles["Monto_MXN"] = [self.pm.humanize_stake_mxn(x*available/total, self.step) for x in singles.Monto_MXN]
        singles["Stake_%"] = singles.Monto_MXN/self.bankroll*100
        if not parlays.empty:
            parlays["Stake_%"] = parlays.Monto_MXN/self.bankroll*100
            parlays["Ganancia_Potencial_MXN"] = parlays.Monto_MXN*(parlays.Cuota_Combinada-1)
        # Retain zero-stake leg rows so the simulation can resolve all parlay legs.
        risk = simulate_ticket_risk(singles, parlays)
        risk["Total_Exposure_MXN"] = float(singles.Monto_MXN.sum())+parlay_total
        risk["Total_Exposure_%"] = risk["Total_Exposure_MXN"]/self.bankroll*100
        risk["Singles_Independent_MC"] = PortfolioManager.simulate_portfolio_monte_carlo(
            singles.Prob_Model.tolist(), singles.Odds.tolist(), singles.Monto_MXN.tolist(), push_probs=singles.Prob_Push.tolist())
        return {"singles_ticket_df": singles, "parlays_ticket_df": parlays,
                "risk_report": risk, "paper_mode": self.paper_mode}
