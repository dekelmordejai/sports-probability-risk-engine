"""Daily research desk. Offline demo is explicit; live data never falls back to fixtures."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import requests
from src.demo_data import soccer_history, nba_history, register_demo_markets
from src.drift_and_weather import CUSUMDriftAndKillSwitch, StadiumWeatherOracle
from src.entity_resolver import EntityResolver
from src.hyperparameter_tuner import MonthlyBayesianTuner
from src.live_odds_connector import LiveOddsConnector
from src.master_orchestrator import MultiSportMasterOrchestrator
from src.nba_possession_engine import NBAPossessionEngine
from src.nfl_epa_kalman import NFLKalmanKeyNumberEngine
from src.persistence_sync import QuantPersistenceSync
from src.quant_core import MarketMechanics
from src.soccer_dixon_coles import DixonColesEngine
from src.soccer_loader import SoccerDataLoader
from src.state_registry import ProductionStateRegistry
from src.tactical_alert_agent import TacticalMarketAgent
from src.advanced_alpha_engines import AmbiguityAverseKelly

ROOT = Path(__file__).resolve().parent
SOCCER = {"EPL": "E0", "La_Liga": "SP1", "Liga_MX": "MEX"}
MODES = ["morning_scan", "tactical_steam", "clv_audit", "auto_settle", "monthly_tune"]


def utcnow():
    return datetime.now(timezone.utc)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def season_codes(now):
    start = now.year if now.month >= 7 else now.year-1
    return [f"{y%100:02d}{(y+1)%100:02d}" for y in (start-1, start)]


def cached_frame(path, loader, max_age_hours=12):
    if path.exists() and (utcnow().timestamp()-path.stat().st_mtime)/3600 <= max_age_hours:
        return pd.read_parquet(path)
    frame = loader()
    if frame.empty:
        raise ValueError("Provider returned no training data")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)
    return frame


class DailyPipeline:
    def __init__(self, config, demo=False, state_dir=None, data_dir=None, report_dir=None,
                 sync=False, notify=False, record_live=False):
        self.config, self.demo = config, demo
        self.state = Path(state_dir or ROOT/"state"/("demo" if demo else "live"))
        self.data = Path(data_dir or ROOT/"data")
        self.reports = Path(report_dir or ROOT/"reports"/("demo" if demo else "live"))
        for folder in (self.state, self.data, self.reports):
            folder.mkdir(parents=True, exist_ok=True)
        if demo and (sync or notify or record_live):
            raise ValueError("Demo cannot sync, notify or record live fills")
        self.sync = QuantPersistenceSync(self.state/"ledger.sqlite", external_enabled=sync)
        self.odds = LiveOddsConnector()
        self.notify, self.record_live = notify, record_live
        self.errors, self.skipped, self.observations, self.weather = [], [], [], {}
        self.snapshot = self.state/"quant_engines_snapshot.json"

    def health(self):
        initial = self.config["portfolio"]["default_bankroll_mxn"]
        current, peak, pending = self.sync.bankroll_state(initial, paper=not self.record_live)
        ledger = self.sync.ledger_dataframe()
        if not ledger.empty:
            ledger = ledger[ledger.Paper == (not self.record_live)]
        health = CUSUMDriftAndKillSwitch(self.config["portfolio"]["max_drawdown_killswitch_pct"]).evaluate_portfolio_health(ledger, current, peak)
        health.update(current_bankroll=current, peak_bankroll=peak, pending_exposure=pending)
        return health

    def desk(self, health):
        p = self.config["portfolio"]
        current = health["current_bankroll"]
        if current <= 0:
            raise ValueError("Bankroll depleted; no further allocation is possible")
        remaining = max(0., p["max_total_exposure"]-health["pending_exposure"]/current)
        return MultiSportMasterOrchestrator(current, p["kelly_fraction"], remaining,
            p["min_ev_threshold"], p["max_normalized_entropy"], min(p["max_parlay_exposure"], remaining),
            p["human_round_step_mxn"], health["league_kelly_multipliers"],
            paper_mode=not self.record_live or health["kill_switch_paper_mode"])

    def train_soccer(self, league):
        cfg = self.config["leagues"]["soccer"].copy()
        tuned_path = self.state/"tuned_settings.json"
        if tuned_path.exists():
            cfg.update(json.loads(tuned_path.read_text(encoding="utf-8")).get(league, {}))
        if self.demo:
            teams = {"EPL": ["Arsenal", "Chelsea", "Liverpool", "Man City"],
                     "La_Liga": ["Real Madrid", "Barcelona", "Ath Madrid", "Betis"]}.get(league)
            history = soccer_history(teams)
        else:
            history = SoccerDataLoader(self.data).fetch_with_parquet_cache(SOCCER[league], season_codes(utcnow()))
            history = history[history.date < pd.Timestamp.now().normalize()].tail(1600)
        return DixonColesEngine(xi=cfg["xi_decay"], xg_weight=cfg["xg_weight"]).fit(history)

    def train_nfl(self):
        model = NFLKalmanKeyNumberEngine(**self.config["leagues"]["nfl"])
        if self.demo:
            model.initialize_states(["KC", "SF", "BUF", "BAL"])
            for observed in [6., -3., 4., 8.]:
                model.step_predict_and_update("KC", "SF", observed_y=observed)
            return model
        year = utcnow().year-(utcnow().month < 3)
        history = cached_frame(self.data/f"nfl_{year}.parquet", lambda: model.load_and_aggregate_nflverse([year-1, year]))
        model.initialize_states(sorted(set(history.home_team) | set(history.away_team)))
        last_season = None
        for row in history.sort_values(["season", "week", "gameday"]).itertuples():
            if last_season is not None and last_season != row.season:
                model.apply_offseason_regression()
            model.step_predict_and_update(row.home_team, row.away_team, row.rest_diff, row.kalman_obs_y)
            last_season = row.season
        return model

    def train_nba(self):
        model = NBAPossessionEngine(**self.config["leagues"]["nba"])
        if self.demo:
            return model.fit_historical_logs(nba_history())
        # October may precede opening night; include the completed prior season.
        year = utcnow().year-(utcnow().month < 10)
        season = f"{year-1}-{year%100:02d}"
        frames = [cached_frame(self.data/f"nba_{season}.parquet", lambda: model.fetch_nba_api_gamelogs(season))]
        current = f"{year}-{(year+1)%100:02d}"
        if utcnow().month != 10 or utcnow().day >= 20:
            try:
                frames.append(cached_frame(self.data/f"nba_{current}.parquet", lambda: model.fetch_nba_api_gamelogs(current)))
            except (ValueError, KeyError):
                self.skipped.append({"league": "NBA", "reason": "Current season has no completed records"})
        return model.fit_historical_logs(pd.concat(frames, ignore_index=True).tail(1800))

    def record_market(self, desk, event, sport, league, match, market, names, probs, quote, push=0.):
        desk.register_market_opportunity(sport, league, match, market, names, probs, quote["best_odds"],
             sharp_odds=quote["sharp_odds"], best_books=quote["best_books"],
             event_id=event["id"], commence_time=event["commence_time"], push_probability=push)
        self.observations.append({"event_id": event["id"], "commence_time": event["commence_time"],
             "observed_at": utcnow().isoformat(), "league": league, "match": match,
             "market": market, "selections": names, "model_probs": list(map(float, probs)), "push_probability": push,
             "sharp_odds": quote["sharp_odds"], "best_odds": quote["best_odds"], "best_books": quote["best_books"]})

    def evaluate_events(self, desk, league, events, engine):
        for event in events:
            kickoff = pd.Timestamp(event["commence_time"])
            hours = (kickoff-pd.Timestamp.now(tz="UTC")).total_seconds()/3600
            if not 0 < hours <= 14:
                continue
            known = engine.teams if league != "NBA" else list(engine.team_states)
            home = EntityResolver.resolve(event["home_team"], league, known)
            away = EntityResolver.resolve(event["away_team"], league, known)
            if not home or not away:
                self.skipped.append({"event": event["id"], "reason": "Unresolved or untrained teams"})
                continue
            match = f"{home} vs {away}" if league in SOCCER else f"{away} @ {home}"
            if league in SOCCER:
                quote = self.odds.extract_best_and_sharp_h2h(event, True)
                if quote:
                    alt = SoccerDataLoader(self.data).compute_altitude_boost(home, away, SOCCER[league],
                          self.config["leagues"]["soccer"]["liga_mx_altitude_gamma"])
                    pred = engine.predict_match_markets(home, away, alt_boost=alt)
                    self.record_market(desk, event, "Soccer", league, match, "1X2", [home, "Empate", away], pred["1X2"], quote)
                continue
            markets = self.odds.extract_main_spread_and_total(event)
            spread, total = markets["spread"], markets["total"]
            if league == "NFL":
                self.weather[event["id"]] = StadiumWeatherOracle.fetch_stadium_weather(home, event["commence_time"])
                if spread:
                    pred = engine.price_nfl_markets(home, away, spread["line_home"])
                    prob = pred["P_Home_Cover_Raw"]
            elif spread or total:
                pred = engine.price_nba_markets(home, away, kickoff.tz_convert("UTC").tz_localize(None).normalize(),
                         spread["line_home"] if spread else 0., total["line_total"] if total else 220.5)
                prob = pred["P_Home_Cover"]
            if spread:
                line = spread["line_home"]
                push = pred["P_Push"]
                self.record_market(desk, event, league, league, match, f"Spread {line:+g}",
                     [f"{home} {line:+g}", f"{away} {-line:+g}"], [prob, 1-prob-push], spread, push)
            if league == "NBA" and total:
                line = total["line_total"]
                self.record_market(desk, event, league, league, match, f"Total {line:g}",
                     [f"Over {line:g}", f"Under {line:g}"], [pred["P_Over"], pred["P_Under"]], total, pred["P_Total_Push"])

    def save_observations(self):
        path = self.state/"odds_observations.json"
        prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        cutoff = pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=7)
        merged = [x for x in prior+self.observations if pd.Timestamp(x["observed_at"]) > cutoff]
        write_json(path, merged)
        return prior

    def auto_settle(self):
        completed = {}
        if self.demo:
            completed = {"demo-mx": (3, 1), "demo-nfl": (27, 20), "demo-nba": (115, 110)}
        else:
            pending = self.sync.ledger_dataframe()
            if not pending.empty:
                pending = pending[pending.Estado == "PENDIENTE"]
            leagues = set(pending.Liga) if not pending.empty else set()
            if "Multi" in leagues:
                leagues = set(self.odds.SPORT_KEYS)
            for league in sorted(leagues):
                for game in self.odds.fetch_completed_scores(league, 3):
                    scores = {item["name"]: int(item["score"]) for item in (game.get("scores") or [])}
                    if game["home_team"] in scores and game["away_team"] in scores:
                        completed[game["id"]] = (scores[game["home_team"]], scores[game["away_team"]])
        return {"settled_bets": self.sync.auto_settle_pending_bets(completed)}

    def morning_scan(self):
        if not self.demo:
            self.auto_settle()
        health = self.health()
        desk = self.desk(health)
        engines, nfl, nba = {}, None, None
        if self.demo:
            engines = {lg: self.train_soccer(lg) for lg in SOCCER}
            nfl, nba = self.train_nfl(), self.train_nba()
            register_demo_markets(desk)
        else:
            if not self.odds.api_key:
                raise ValueError("ODDS_API_KEY is required for live mode; use --demo for offline verification")
            for league in self.odds.SPORT_KEYS:
                try:
                    if not self.odds.has_upcoming_matches_within_hours(league):
                        self.skipped.append({"league": league, "reason": "No upcoming events within 14 hours"})
                        continue
                    if league in SOCCER:
                        engine = engines[league] = self.train_soccer(league)
                    elif league == "NFL":
                        engine = nfl = self.train_nfl()
                    else:
                        engine = nba = self.train_nba()
                    events = self.odds.fetch_league_odds(league, "h2h" if league in SOCCER else "spreads,totals")
                    self.evaluate_events(desk, league, events, engine)
                except Exception as error:
                    # Request exception strings can contain API keys: record only the type.
                    self.errors.append({"league": league, "error_type": type(error).__name__})
        result = desk.build_optimal_weekend_ticket()
        for key in ("singles_ticket_df", "parlays_ticket_df"):
            frame = result[key]
            frame.to_csv(self.reports/f"{key}.csv", index=False, encoding="utf-8-sig")
        paper = result["paper_mode"]
        self.sync.append_ticket(result["singles_ticket_df"], paper=paper)
        self.sync.append_ticket(result["parlays_ticket_df"], paper=paper, parlays=True)
        self.sync.sync_ledger_to_sheets()
        self.sync.publish_ticket_to_notion(result["singles_ticket_df"])
        ProductionStateRegistry.save_daily_snapshot(engines, nfl, nba, health["current_bankroll"], self.snapshot)
        self.save_observations()
        return {"data_source": "SYNTHETIC DEMO" if self.demo else "live providers", "paper_mode": paper,
                "singles_count": int((result["singles_ticket_df"].get("Monto_MXN", pd.Series(dtype=float)) > 0).sum()),
                "parlays_count": len(result["parlays_ticket_df"]), "risk": result["risk_report"],
                "health": health, "weather": self.weather, "errors": self.errors, "skipped": self.skipped}

    def tactical_steam(self):
        snapshot = ProductionStateRegistry.load_daily_snapshot(self.snapshot)
        desk = self.desk(self.health())
        if self.demo:
            signals = TacticalMarketAgent().detect_steam_and_stale_lines("DEMO", ["A", "B"],
                       [2.05, 1.85], [1.7, 2.3], [2.1, 1.9], ["Synthetic", "Synthetic"], [.65, .35])
            return {"data_source": "SYNTHETIC DEMO", "signals": signals}
        engines = {**snapshot["soccer_engines"], "NFL": snapshot["nfl_engine"], "NBA": snapshot["nba_engine"]}
        for league, engine in engines.items():
            if engine is not None and self.odds.has_upcoming_matches_within_hours(league):
                events = self.odds.fetch_league_odds(league, "h2h" if league in SOCCER else "spreads,totals")
                self.evaluate_events(desk, league, events, engine)
        prior = self.save_observations()
        agent, signals = TacticalMarketAgent(), []
        for obs in self.observations:
            if obs.get("push_probability", 0) > 0:
                continue
            earlier = [x for x in prior if (x["event_id"], x["market"], x["selections"]) == (obs["event_id"], obs["market"], obs["selections"])]
            if not earlier:
                continue
            t0 = max(earlier, key=lambda x: x["observed_at"])
            new = agent.detect_steam_and_stale_lines(obs["match"], obs["selections"], t0["sharp_odds"],
                      obs["sharp_odds"], obs["best_odds"], obs["best_books"], obs["model_probs"])
            signals.extend(new)
            if self.notify:
                for signal in new:
                    fraction = AmbiguityAverseKelly.compute_robust_fraction(signal["Posterior_Prob"], .035, signal["Stale_Odd"])
                    agent.send_telegram_flash_alert(signal, fraction["robust_kelly_%"], desk.bankroll)
        return {"signals": signals, "paper_mode": desk.paper_mode}

    def clv_audit(self):
        path = self.state/"odds_observations.json"
        history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        candidates = {}
        for obs in history:
            start, sampled = pd.Timestamp(obs["commence_time"]), pd.Timestamp(obs["observed_at"])
            # A late-day API quote is not a closing quote. Use an actual stored pre-kickoff sample.
            if start > pd.Timestamp.now(tz="UTC") or not 0 <= (start-sampled).total_seconds() <= 15*60:
                continue
            key = (obs["event_id"], obs["market"])
            if key not in candidates or sampled > pd.Timestamp(candidates[key]["observed_at"]):
                candidates[key] = obs
        closing = {}
        for obs in candidates.values():
            probs, _ = MarketMechanics.devig_shin(obs["sharp_odds"])
            for name, p in zip(obs["selections"], probs):
                closing[(obs["event_id"], obs["market"], name)] = p
        changed = self.sync.audit_clv(closing)
        ledger = self.sync.ledger_dataframe()
        missing = int((ledger.Prob_Cierre == "").sum()) if not ledger.empty else 0
        return {"audited_bets": changed, "missing_closing_samples": missing,
                "closing_window_minutes": 15, "note": "Missing pre-kickoff samples remain unaudited"}

    def monthly_tune(self, trials=20):
        results = {}
        for league, code in SOCCER.items():
            history = soccer_history() if self.demo else SoccerDataLoader(self.data).fetch_with_parquet_cache(code, season_codes(utcnow()))
            results[league] = MonthlyBayesianTuner.tune_soccer_hyperparameters(history, trials)
        write_json(self.state/"tuned_settings.json", results)
        return results

    def run(self, mode, trials=20):
        result = self.monthly_tune(trials) if mode == "monthly_tune" else getattr(self, mode)()
        write_json(self.reports/f"{mode}.json", result)
        self.sync.ledger_dataframe().to_csv(self.reports/"ledger.csv", index=False, encoding="utf-8-sig")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="morning_scan")
    parser.add_argument("--demo", action="store_true", help="Deterministic synthetic data; no external calls")
    parser.add_argument("--config", type=Path, default=ROOT/"config/settings.yaml")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--sync", action="store_true", help="Enable configured Sheets/Notion mirrors")
    parser.add_argument("--notify", action="store_true", help="Send tactical alerts to configured Telegram chat")
    parser.add_argument("--record-live", action="store_true", help="Record confirmed manual fills as real ledger entries")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pipeline = DailyPipeline(config, args.demo, args.state_dir, args.data_dir, args.report_dir,
                             args.sync, args.notify, args.record_live)
    try:
        result = pipeline.run(args.mode, args.trials)
    except requests.RequestException as error:
        print(json.dumps({"status": "provider_error", "error_type": type(error).__name__,
                          "message": "Provider request failed; credentials and request URLs are not logged"}))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 1 if pipeline.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
