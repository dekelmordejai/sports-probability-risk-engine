"""Counterfactual checks on information available before a daily forecast."""
import numpy as np
import pandas as pd
import pytest

from examples.offline_evaluation import probability_scores
from src.demo_data import soccer_history
from src.pi_ratings import PiRatingSystem
from src.soccer_meta_model import SoccerMetaModel
from src.walk_forward import WalkForwardBacktester


@pytest.fixture
def same_day_history():
    data = soccer_history(teams=["A", "B"], n=5)
    data["home_team"], data["away_team"] = "A", "B"
    data["date"] = pd.to_datetime([
        "2025-01-01", "2025-01-03 10:00", "2025-01-03 18:00", "2025-01-06", "2025-01-09"
    ], format="mixed")
    return data


@pytest.mark.parametrize("kind", ["pi", "meta"])
def test_same_day_results_cannot_change_same_day_features(same_day_history, kind):
    original = same_day_history
    altered = original.copy()
    altered.loc[1, ["home_goals", "away_goals", "home_xg", "away_xg"]] = [8, 0, 0., 8.]
    if kind == "pi":
        columns = ["pi_home_h", "pi_away_a", "pi_expected_gd"]
        before = PiRatingSystem().compute_historical_features(original)
        after = PiRatingSystem().compute_historical_features(altered)
    else:
        columns = ["rest_days_diff", "xg_overperf_diff"]
        before = SoccerMetaModel().build_meta_features(original)
        after = SoccerMetaModel().build_meta_features(altered)
    np.testing.assert_array_equal(before.loc[:2, columns], after.loc[:2, columns])
    assert not np.array_equal(before.loc[3:, columns], after.loc[3:, columns])


def test_future_results_cannot_change_earlier_walk_forward_forecasts():
    history = soccer_history(n=48)
    cutoff = history.date.iloc[36]
    altered = history.copy()
    future = altered.date >= cutoff
    altered.loc[future, ["home_goals", "away_goals", "home_xg", "away_xg", "actual_idx"]] = [5, 0, 4., .1, 0]
    arguments = dict(warmup_matches=24, retrain_Step_days=14)
    _, before, _ = WalkForwardBacktester(**arguments).run(history)
    _, after, _ = WalkForwardBacktester(**arguments).run(altered)
    earlier = before.date < cutoff
    assert earlier.sum() == 12
    np.testing.assert_array_equal(np.vstack(before.loc[earlier, "probs_model"]),
                                  np.vstack(after.loc[earlier, "probs_model"]))


def test_first_forecast_day_never_trains_on_earlier_same_day_results(monkeypatch):
    import src.walk_forward as module
    history = soccer_history(n=40)
    history.loc[19, "date"] = pd.Timestamp("2025-02-27 09:00")
    history.loc[20, "date"] = pd.Timestamp("2025-02-27 18:00")
    original_fit = module.DixonColesEngine.fit
    observed = []
    def audited_fit(self, frame, current_date=None):
        cutoff = pd.Timestamp(current_date)
        assert (frame.date < cutoff).all()
        observed.append(cutoff)
        return original_fit(self, frame, current_date=current_date)
    monkeypatch.setattr(module.DixonColesEngine, "fit", audited_fit)
    WalkForwardBacktester(warmup_matches=20, retrain_Step_days=60).run(history)
    assert observed[0] == pd.Timestamp("2025-02-27")


def test_probability_scores_have_explicit_multiclass_convention():
    scores = probability_scores([[1, 0, 0], [0, 0, 1]], [0, 2])
    assert scores["rps"] == 0 and scores["brier_multiclass"] == 0
    assert scores["log_loss"] < 1e-12
    with pytest.raises(ValueError):
        probability_scores([[.7, .7, .1]], [0])
