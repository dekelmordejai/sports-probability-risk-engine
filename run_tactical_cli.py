"""Evaluate in-play spreads, soccer, player props and copula scenarios."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import yaml
from src.state_registry import ProductionStateRegistry
from src.advanced_alpha_engines import AmbiguityAverseKelly, InPlayStochasticEngine, PlayerPropsEngine, CopulaCorrelatedParlayPricer
from src.persistence_sync import QuantPersistenceSync, match_teams
from src.quant_core import PortfolioManager
from src.drift_and_weather import CUSUMDriftAndKillSwitch

ROOT = Path(__file__).resolve().parent


def run_inplay_nba_nfl_evaluation(sport, match_name, pregame_fair_spread, current_home,
        current_away, time_elapsed_frac, live_spread_home, live_odds_home, shooting_luck_pts=0.,
        epistemic_std_p=.025, commit_to_ledger=False, state_dir=None, sync_external=False,
        event_id=None):
    if sport not in ("NBA", "NFL"):
        raise ValueError("Diffusion supports NBA or NFL")
    folder = Path(state_dir or ROOT/"state/live")
    config = yaml.safe_load((ROOT/"config/settings.yaml").read_text(encoding="utf-8"))["portfolio"]
    sync = QuantPersistenceSync(folder/"ledger.sqlite", external_enabled=sync_external)
    bankroll, peak, pending = sync.bankroll_state(config["default_bankroll_mxn"], paper=True)
    health = CUSUMDriftAndKillSwitch(config["max_drawdown_killswitch_pct"]).evaluate_portfolio_health(sync.ledger_dataframe(), bankroll, peak)
    if pregame_fair_spread is None:
        snapshot = ProductionStateRegistry.load_daily_snapshot(folder/"quant_engines_snapshot.json")
        home, away = match_teams(match_name)
        model = snapshot["nfl_engine" if sport == "NFL" else "nba_engine"]
        if model is None:
            raise ValueError("No model for this sport in the daily snapshot")
        if sport == "NFL":
            pregame_fair_spread = model.price_nfl_markets(home, away, 0)["Kalman_Fair_Spread"]
        else:
            pregame_fair_spread = model.price_nba_markets(home, away, pd.Timestamp.now().normalize(), 0, 220.5)["Fair_Spread_Home"]
    res = InPlayStochasticEngine.price_live_diffusion_nba_nfl(pregame_fair_spread,
        13.2 if sport == "NFL" else 11.85, current_home, current_away, time_elapsed_frac,
        live_spread_home, shooting_luck_pts)
    p = res["P_Live_Home_Cover"]
    robust = AmbiguityAverseKelly.compute_robust_fraction(p, epistemic_std_p, live_odds_home,
                                                        kelly_fraction=config["kelly_fraction"])
    cap = max(0., bankroll*config["max_total_exposure"]-pending)
    multiplier = health["league_kelly_multipliers"].get(sport, 1.)
    amount = PortfolioManager.humanize_stake_mxn(bankroll*robust["robust_kelly_%"]/100*multiplier,
                                                config["human_round_step_mxn"], cap)
    # The continuous diffusion cannot represent pre-terminal integer-line pushes.
    if float(live_spread_home).is_integer() or time_elapsed_frac == 1:
        amount = 0.
    ev = (p*live_odds_home + res.get("P_Push", 0.)-1)*100
    if ev < config["min_ev_threshold"]*100:
        amount = 0.
    result = {**res, **robust, "EV_%": ev, "Monto_MXN": amount, "Paper": True,
              "health": health, "recorded_rows": 0}
    if commit_to_ledger and amount > 0:
        if not event_id:
            raise ValueError("--event-id is required to record an idempotent paper fill")
        home, away = match_teams(match_name)
        selection = f"{home} {live_spread_home:+g}"
        market = f"InPlay Spread {live_spread_home:+g}"
        signature = hashlib.sha256(f"{sport}|{event_id}|{market}|{selection}".encode()).hexdigest()[:20]
        ticket = pd.DataFrame([dict(Sport=sport, League=sport, Match=match_name, Market=market,
            Selection=selection, Selection_ID=signature, Event_ID=event_id, Prob_Model=p,
            Odds=live_odds_home, Monto_MXN=amount, EV_=ev)])
        result["recorded_rows"] = sync.append_ticket(ticket, paper=True)
        sync.sync_ledger_to_sheets()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["inplay", "soccer", "discrete_prop", "gamma_prop", "sgp"], default="inplay")
    parser.add_argument("--sport", choices=["NBA", "NFL"], default="NBA")
    parser.add_argument("--match", default="Away @ Home")
    parser.add_argument("--event-id")
    parser.add_argument("--pre_spread", type=float)
    parser.add_argument("--h_score", type=int, default=0)
    parser.add_argument("--a_score", type=int, default=0)
    parser.add_argument("--elapsed", type=float, default=0.)
    parser.add_argument("--live_line", type=float, default=-.5)
    parser.add_argument("--live_odds", type=float, default=1.91)
    parser.add_argument("--luck_pts", type=float, default=0.)
    parser.add_argument("--sigma-p", type=float, default=.025)
    parser.add_argument("--commit", action="store_true", help="Record a paper fill; never places a wager")
    parser.add_argument("--sync", action="store_true", help="Mirror the ledger to configured Sheets")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--minutes", type=float, default=30.)
    parser.add_argument("--rate", type=float, default=.2)
    parser.add_argument("--line", type=float, default=6.5)
    parser.add_argument("--mean-yards", type=float, default=65.)
    parser.add_argument("--lambda-home", type=float, default=1.5)
    parser.add_argument("--mu-away", type=float, default=1.1)
    parser.add_argument("--reds-home", type=int, default=0)
    parser.add_argument("--reds-away", type=int, default=0)
    parser.add_argument("--p1", type=float, default=.55)
    parser.add_argument("--p2", type=float, default=.55)
    parser.add_argument("--rho", type=float, default=.3)
    args = parser.parse_args(argv)
    if args.mode == "inplay":
        result = run_inplay_nba_nfl_evaluation(args.sport, args.match, args.pre_spread, args.h_score,
            args.a_score, args.elapsed, args.live_line, args.live_odds, args.luck_pts, args.sigma_p,
            args.commit, args.state_dir, args.sync, args.event_id)
    elif args.mode == "soccer":
        result = InPlayStochasticEngine.price_live_soccer_dixon_robinson(args.lambda_home, args.mu_away,
            args.minutes, args.h_score, args.a_score, args.reds_home, args.reds_away)
        result["Live_1X2"] = result["Live_1X2"].tolist()
    elif args.mode == "discrete_prop":
        result = PlayerPropsEngine.price_discrete_prop_nbinom(args.minutes, args.rate, bookie_line=args.line)
    elif args.mode == "gamma_prop":
        result = PlayerPropsEngine.price_continuous_yards_gamma(args.mean_yards, bookie_line=args.line)
    else:
        result = CopulaCorrelatedParlayPricer.price_bivariate_sgp(args.p1, args.p2, args.rho, args.live_odds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
