"""Integration and regressions around financial risk, settlement and provider failures."""
import importlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from run_daily_pipeline import DailyPipeline, ROOT
from src.quant_core import MarketMechanics, PortfolioManager, ModelEvaluator
from src.parlay_builder import QuantitativeParlayBuilder
from src.master_orchestrator import MultiSportMasterOrchestrator
from src.demo_data import register_demo_markets, soccer_history
from src.persistence_sync import QuantPersistenceSync, settle_selection
from src.state_registry import ProductionStateRegistry
from src.drift_and_weather import CUSUMDriftAndKillSwitch, StadiumWeatherOracle
from src.soccer_loader import SoccerDataLoader
from src.advanced_alpha_engines import PlayerPropsEngine, InPlayStochasticEngine
from src.soccer_dixon_coles import DixonColesEngine
from src.live_odds_connector import LiveOddsConnector
from src.ticket_risk import simulate_ticket_risk


@pytest.mark.parametrize("name", [p.stem for p in (ROOT/"src").glob("*.py")])
def test_all_source_modules_import(name):
    importlib.import_module(f"src.{name}")


@pytest.mark.parametrize("odds", [[np.nan, 2], [np.inf, 2], [], [2], [0, 3]])
def test_invalid_odds_rejected(odds):
    with pytest.raises(ValueError):
        MarketMechanics.devig_shin(odds)


@pytest.mark.parametrize("amount,cap,expected", [(25, None, 0), (49.99, None, 0), (680, 475, 450), (100, 0, 0)])
def test_rounding_preserves_step_and_both_caps(amount, cap, expected):
    result = PortfolioManager.humanize_stake_mxn(amount, 50, cap)
    assert result == expected and result <= amount
    assert result % 50 == 0


def test_single_bet_kelly_matches_closed_form():
    table = PortfolioManager(kelly_fraction=.25, max_total_exposure=.15).optimize_simultaneous_kelly([.6], [2.])
    assert table["Stake_%"].iloc[0] == pytest.approx(5., abs=.0001)


def test_full_ticket_exposure_is_shared_with_parlays():
    desk = MultiSportMasterOrchestrator(max_portfolio_exposure=.025, max_parlay_exposure=.015)
    register_demo_markets(desk)
    result = desk.build_optimal_weekend_ticket(.05)
    singles, parlays = result["singles_ticket_df"], result["parlays_ticket_df"]
    total = singles.Monto_MXN.sum() + (parlays.Monto_MXN.sum() if not parlays.empty else 0)
    assert total <= 625
    assert result["risk_report"]["VaR_95_MXN"] >= -total
    assert np.all(singles.Monto_MXN % 50 == 0)
    assert singles.Monto_MXN.sum()/25000*100 == pytest.approx(singles["Stake_%"].sum())


def test_parlay_small_bankroll_does_not_round_up():
    desk = MultiSportMasterOrchestrator()
    register_demo_markets(desk)
    singles = pd.DataFrame(desk.raw_opportunities)
    result = QuantitativeParlayBuilder().generate_optimal_parlays(singles, 500.)
    assert result.empty


def test_parlay_rejects_same_event_and_negative_legs():
    rows = pd.DataFrame([
        dict(Sport="Soccer", League="EPL", Match="A vs B", Market="1X2", Selection="A", Prob_Model=.7, Odds=2.),
        dict(Sport="Soccer", League="EPL", Match="A vs B", Market="1X2", Selection="B", Prob_Model=.6, Odds=2.),
        dict(Sport="NBA", League="NBA", Match="C @ D", Market="Spread", Selection="D", Prob_Model=.5, Odds=1.8)])
    assert QuantitativeParlayBuilder().generate_optimal_parlays(rows, 25000).empty


@pytest.mark.parametrize("match,market,pick,score,result", [
    ("KC vs SF", "Spread -3", "KC -3", (27, 20), "GANADA"),
    ("KC vs SF", "Spread -3", "SF +3", (27, 20), "PERDIDA"),
    ("SF @ KC", "Spread -3", "KC -3", (23, 20), "PUSH"),
    ("Toluca vs UNAM", "1X2", "Empate", (1, 1), "GANADA"),
    ("LAL @ DEN", "Total 220", "Over 220", (110, 110), "PUSH"),
    ("LAL @ DEN", "Total 220", "Under 220", (109, 110), "GANADA"),
    ("LAL @ DEN", "Spread -3", "Home -3", (120, 100), "PENDIENTE"),
    ("LAL @ DEN", "Total 220", "Bogus 220", (100, 100), "PENDIENTE")])
def test_settlement_orientation_and_push(match, market, pick, score, result):
    assert settle_selection(match, market, pick, *score) == result


def test_local_ledger_append_and_settle_are_idempotent(tmp_path):
    desk = MultiSportMasterOrchestrator()
    register_demo_markets(desk)
    tickets = desk.build_optimal_weekend_ticket()
    sync = QuantPersistenceSync(tmp_path/"ledger.sqlite")
    first = sync.append_ticket(tickets["singles_ticket_df"])
    assert first > 0
    assert sync.append_ticket(tickets["singles_ticket_df"]) == 0
    assert sync.auto_settle_pending_bets({"demo-mx": (3, 1)}) == 1
    before = sync.ledger_dataframe().copy()
    assert sync.auto_settle_pending_bets({"demo-mx": (3, 1)}) == 0
    pd.testing.assert_frame_equal(before, sync.ledger_dataframe())


def test_cusum_and_drawdown_reduce_risk():
    ledger = pd.DataFrame({"Liga": ["NFL"]*12, "CLV_%": [-.8]*12})
    health = CUSUMDriftAndKillSwitch().evaluate_portfolio_health(ledger, 19000, 25000)
    assert health["kill_switch_paper_mode"]
    assert health["league_kelly_multipliers"]["NFL"] == .5
    assert health["cusum_scores"]["NFL"] < -5


def test_weather_failure_is_unknown_not_fabricated(monkeypatch):
    import requests
    def fail(*args, **kwargs):
        raise requests.Timeout()
    monkeypatch.setattr(requests, "get", fail)
    result = StadiumWeatherOracle.fetch_stadium_weather("KC")
    assert not result["available"]
    assert result["wind_mph"] is None


def test_cache_is_keyed_by_requested_seasons(tmp_path, monkeypatch):
    calls = []
    loader = SoccerDataLoader(tmp_path)
    def fetch(league, seasons):
        calls.append(tuple(seasons))
        frame = soccer_history(n=8)
        frame["season"] = seasons[0]
        frame["date"] = frame.date + pd.Timedelta(days=int(seasons[0][:2])*365)
        return frame
    monkeypatch.setattr(loader, "fetch_european_league", fetch)
    loader.fetch_with_parquet_cache("E0", ["2425"])
    result = loader.fetch_with_parquet_cache("E0", ["2425", "2526"])
    assert calls == [("2425",), ("2526",)]
    assert set(result.season) == {"2425", "2526"}


def test_snapshot_roundtrip_is_numerically_identical(tmp_path):
    engine = DixonColesEngine().fit(soccer_history(n=40))
    snapshot = tmp_path/"models.json"
    ProductionStateRegistry.save_daily_snapshot({"Liga_MX": engine}, None, None, 25000., snapshot)
    restored = ProductionStateRegistry.load_daily_snapshot(snapshot)
    before = engine.predict_match_markets("Toluca", "UNAM")["1X2"]
    after = restored["soccer_engines"]["Liga_MX"].predict_match_markets("Toluca", "UNAM")["1X2"]
    np.testing.assert_array_equal(before, after)
    payload = json.loads(snapshot.read_text())
    payload["timestamp_utc"] = "2020-01-01T00:00:00+00:00"
    snapshot.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stale"):
        ProductionStateRegistry.load_daily_snapshot(snapshot)


def test_discrete_prop_zero_mean_and_integer_push():
    zero = PlayerPropsEngine.price_discrete_prop_nbinom(0, .2, bookie_line=0)
    assert zero["P_Push"] == 1 and zero["P_Over"] == 0
    prop = PlayerPropsEngine.price_discrete_prop_nbinom(30, .2, overdispersion_alpha=0, bookie_line=6)
    assert prop["P_Over"] + prop["P_Under"] + prop["P_Push"] == pytest.approx(1., abs=1e-4)


def test_live_terminal_state_is_deterministic():
    end = InPlayStochasticEngine.price_live_diffusion_nba_nfl(-3, 12, 110, 107, 1, -3)
    assert end["P_Push"] == 1 and end["Live_Remaining_Std"] == 0
    soccer = InPlayStochasticEngine.price_live_soccer_dixon_robinson(1.4, 1.1, 95, 2, 1)
    np.testing.assert_array_equal(soccer["Live_1X2"], [1, 0, 0])


def test_offline_pipeline_all_modes(tmp_path, monkeypatch):
    import requests
    def no_network(*args, **kwargs):
        raise AssertionError("Offline demo attempted network access")
    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setattr(requests, "post", no_network)
    config = yaml.safe_load((ROOT/"config/settings.yaml").read_text())
    pipeline = DailyPipeline(config, demo=True, state_dir=tmp_path/"state", data_dir=tmp_path/"data", report_dir=tmp_path/"reports")
    scan = pipeline.run("morning_scan")
    assert scan["singles_count"] > 0 and scan["parlays_count"] > 0
    assert scan["risk"]["Total_Exposure_%"] <= 15
    assert pipeline.run("tactical_steam")["signals"]
    assert pipeline.run("auto_settle")["settled_bets"] > 0
    assert pipeline.run("auto_settle")["settled_bets"] == 0
    assert pipeline.run("clv_audit")["audited_bets"] == 0
    tune = pipeline.run("monthly_tune", trials=1)
    assert 0 <= tune["EPL"]["validation_rps"] <= 1


def test_rps_perfect_and_rejects_malformed_distribution():
    assert ModelEvaluator.ranked_probability_score([[1, 0, 0], [0, 0, 1]], [0, 2]) == 0
    with pytest.raises(ValueError):
        ModelEvaluator.ranked_probability_score([[.6, .6, .1]], [0])


def test_shared_parlay_leg_cannot_win_when_single_loses():
    singles = pd.DataFrame([dict(Selection_ID="a", Prob_Model=.5, Odds=2., Monto_MXN=100.)])
    parlays = pd.DataFrame([dict(Leg_IDs=["a"], Profit_Boost=0., Monto_MXN=100.)])
    result = simulate_ticket_risk(singles, parlays)
    assert result["VaR_95_MXN"] == -200
    assert 48 < result["Prob_Profitable_Day_%"] < 52


def test_kelly_push_matches_conditional_closed_form():
    table = PortfolioManager(kelly_fraction=.25).optimize_simultaneous_kelly([.5], [2.], push_probs=[.2])
    assert table["EV_%"].iloc[0] == pytest.approx(20.)
    assert table["Stake_%"].iloc[0] == pytest.approx(6.25, abs=.0001)
    risk = PortfolioManager.simulate_portfolio_monte_carlo([0], [2], [100], push_probs=[1])
    assert risk["Expected_PnL_MXN"] == 0 and risk["VaR_95_MXN"] == 0


def test_daily_portfolio_handles_push_without_binary_mispricing():
    desk = MultiSportMasterOrchestrator()
    desk.register_market_opportunity("NFL", "NFL", "SF @ KC", "Spread -3", ["KC -3", "SF +3"],
                                     [.5, .3], [2., 2.], push_probability=.2)
    result = desk.build_optimal_weekend_ticket()
    singles = result["singles_ticket_df"]
    assert len(singles) == 1
    assert singles.iloc[0]["EV_%"] == pytest.approx(20)
    assert result["parlays_ticket_df"].empty


def test_nba_integer_lines_have_explicit_push_mass():
    from src.nba_possession_engine import NBAPossessionEngine
    from src.demo_data import nba_history
    model = NBAPossessionEngine().fit_historical_logs(nba_history())
    result = model.price_nba_markets("DEN", "LAL", "2026-10-16", -3, 220)
    assert result["P_Push"] > 0 and result["P_Total_Push"] > 0
    assert result["P_Home_Cover"]+result["P_Away_Cover"]+result["P_Push"] == pytest.approx(1)
    assert result["P_Over"]+result["P_Under"]+result["P_Total_Push"] == pytest.approx(1)


def test_shopping_keeps_spread_lines_separate():
    event = {"id": "e", "home_team": "KC", "away_team": "SF", "commence_time": "2026-10-20T18:00:00Z", "bookmakers": []}
    for name, line, odds in [("pinnacle", -3, 1.91), ("soft", -3, 2.0), ("different", -7, 4.)]:
        event["bookmakers"].append({"key": name, "title": name, "markets": [{"key": "spreads", "outcomes": [
            {"name": "KC", "point": line, "price": odds}, {"name": "SF", "point": -line, "price": odds}]}]})
    spread = LiveOddsConnector.extract_main_spread_and_total(event)["spread"]
    assert spread["line_home"] == -3
    assert spread["best_odds"] == [2., 2.]
    assert spread["sharp_odds"] == [1.91, 1.91]


def test_smart_polling_no_games_and_provider_errors(monkeypatch):
    import requests
    class Response:
        def raise_for_status(self):
            return None
        def json(self):
            return []
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    connector = LiveOddsConnector(api_key="test-only-key")
    assert not connector.has_upcoming_matches_within_hours("NFL")
    def fail(*args, **kwargs):
        raise requests.Timeout()
    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(requests.Timeout):
        connector.has_upcoming_matches_within_hours("NFL")


def test_clv_uses_prestart_sample_not_poststart_quote(tmp_path):
    config = yaml.safe_load((ROOT/"config/settings.yaml").read_text())
    pipeline = DailyPipeline(config, demo=True, state_dir=tmp_path/"state", report_dir=tmp_path/"reports", data_dir=tmp_path/"data")
    desk = MultiSportMasterOrchestrator()
    register_demo_markets(desk)
    singles = desk.build_optimal_weekend_ticket()["singles_ticket_df"]
    pipeline.sync.append_ticket(singles)
    start = pd.Timestamp.now(tz="UTC")-pd.Timedelta(hours=1)
    base = dict(event_id="demo-mx", commence_time=start.isoformat(), market="1X2", selections=["Toluca", "Empate", "Mazatlan FC"])
    history = [{**base, "observed_at": (start-pd.Timedelta(minutes=5)).isoformat(), "sharp_odds": [1.7, 4, 5]},
               {**base, "observed_at": (start+pd.Timedelta(minutes=5)).isoformat(), "sharp_odds": [1.1, 10, 20]}]
    (pipeline.state/"odds_observations.json").write_text(json.dumps(history))
    assert pipeline.clv_audit()["audited_bets"] == 1
    record = pipeline.sync.ledger_dataframe().query("Evento_ID == 'demo-mx'").iloc[0]
    expected = MarketMechanics.devig_shin([1.7, 4, 5])[0][0]
    assert record.Prob_Cierre == pytest.approx(expected)


def test_live_scan_with_mocked_odds_preserves_canonical_identity(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT/"config/settings.yaml").read_text())
    pipeline = DailyPipeline(config, state_dir=tmp_path/"state", data_dir=tmp_path/"data", report_dir=tmp_path/"reports")
    pipeline.odds.api_key = "test-only-key"
    monkeypatch.setattr(pipeline.odds, "has_upcoming_matches_within_hours", lambda league: league == "Liga_MX")
    monkeypatch.setattr(pipeline, "train_soccer", lambda league: DixonColesEngine().fit(soccer_history(n=40)))
    event = dict(id="live-mock", home_team="Deportivo Toluca", away_team="Pumas UNAM",
                 commence_time=(pd.Timestamp.now(tz="UTC")+pd.Timedelta(hours=1)).isoformat(), bookmakers=[{
                     "key": "pinnacle", "title": "Pinnacle", "markets": [{"key": "h2h", "outcomes": [
                     {"name": "Deportivo Toluca", "price": 3.2}, {"name": "Draw", "price": 3.5}, {"name": "Pumas UNAM", "price": 3.2}]}]}])
    monkeypatch.setattr(pipeline.odds, "fetch_league_odds", lambda *args: [event])
    report = pipeline.morning_scan()
    assert not report["errors"]
    assert pipeline.observations[0]["match"] == "Toluca vs UNAM"
    assert pipeline.observations[0]["event_id"] == "live-mock"
