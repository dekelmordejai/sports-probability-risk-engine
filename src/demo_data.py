"""Explicitly synthetic, deterministic fixtures for offline demos and integration tests."""
import numpy as np
import pandas as pd


def soccer_history(teams=None, n=64):
    teams = teams or ["Toluca", "Mazatlan FC", "America", "UNAM"]
    rng = np.random.default_rng(2026)
    rows = []
    for day in pd.date_range("2025-01-01", periods=n, freq="3D"):
        h, a = rng.choice(teams, 2, replace=False)
        hg, ag = int(rng.poisson(1.7)), int(rng.poisson(1.1))
        rows.append({"date": day, "home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag,
                     "home_xg": 1.6, "away_xg": 1.1, "alt_boost": 0.,
                     "actual_idx": 0 if hg > ag else (1 if hg == ag else 2),
                     "odds_h": 2., "odds_d": 3.5, "odds_a": 4., "close_h": 1.95, "close_d": 3.6, "close_a": 4.1})
    return pd.DataFrame(rows)


def nba_history():
    rng = np.random.default_rng(45)
    teams = ["DEN", "LAL", "BOS", "GSW"]
    rows = []
    for date in pd.date_range("2025-01-01", periods=64, freq="2D"):
        home, away = rng.choice(teams, 2, replace=False)
        pace, h, a = float(rng.normal(100, 3)), float(rng.normal(117, 8)), float(rng.normal(113, 8))
        rows.append({"date": date, "home_team": home, "away_team": away, "pace": pace,
                     "home_ortg": h, "away_ortg": a, "home_pts": round(h*pace/100), "away_pts": round(a*pace/100)})
    return pd.DataFrame(rows)


def register_demo_markets(orchestrator):
    for sport, league, match, market, names, probs, odds, key in [
        ("Soccer", "Liga_MX", "Toluca vs Mazatlan FC", "1X2", ["Toluca", "Empate", "Mazatlan FC"], [.64, .22, .14], [1.72, 3.8, 5.2], "demo-mx"),
        ("NFL", "NFL", "SF @ KC", "Spread -2.5", ["KC -2.5", "SF +2.5"], [.57, .43], [1.91, 1.91], "demo-nfl"),
        ("NBA", "NBA", "LAL @ DEN", "Spread -6.5", ["DEN -6.5", "LAL +6.5"], [.59, .41], [1.91, 1.91], "demo-nba")]:
        orchestrator.register_market_opportunity(sport, league, match, market, names, probs, odds,
                                                 event_id=key, best_books=["Synthetic"]*len(probs))
