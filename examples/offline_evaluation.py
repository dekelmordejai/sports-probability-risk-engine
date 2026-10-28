"""Evaluate chronological synthetic forecasts against three explicit baselines."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

from src.demo_data import soccer_history
from src.quant_core import ModelEvaluator
from src.walk_forward import WalkForwardBacktester


def probability_scores(probabilities, outcomes):
    """Lower is better. Multiclass Brier is the sum across the three classes."""
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    rps = ModelEvaluator.ranked_probability_score(probabilities, outcomes)
    targets = np.eye(3)[outcomes]
    return {
        "rps": float(rps),
        "brier_multiclass": float(np.mean(np.sum((probabilities-targets)**2, axis=1))),
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1, 2])),
    }


def run_evaluation(output):
    history = soccer_history(n=64)
    evaluator = WalkForwardBacktester(warmup_matches=32, retrain_Step_days=14)
    _, predictions, _ = evaluator.run(history)
    outcomes = predictions.actual_idx.to_numpy(dtype=int)
    frequencies = []
    for day in predictions.date:
        previous = history.loc[history.date < day.normalize(), "actual_idx"]
        counts = np.bincount(previous.to_numpy(dtype=int), minlength=3) + 1
        frequencies.append(counts / counts.sum())
    scores = {
        "dixon_coles_pi": probability_scores(np.vstack(predictions.probs_model), outcomes),
        "uniform": probability_scores(np.full((len(predictions), 3), 1/3), outcomes),
        "expanding_frequencies_laplace": probability_scores(frequencies, outcomes),
        "synthetic_market_shin": probability_scores(np.vstack(predictions.probs_market), outcomes),
    }
    result = {
        "data_source": "SYNTHETIC: seeded fixtures, not historical matches or market observations",
        "generator_seed": 2026,
        "history_matches": len(history),
        "warmup_matches": 32,
        "evaluated_matches": len(predictions),
        "evaluation_start": predictions.date.min().isoformat(),
        "evaluation_end": predictions.date.max().isoformat(),
        "retrain_step_days": 14,
        "class_order": ["home", "draw", "away"],
        "lower_scores_are_better": True,
        "scores": scores,
        "interpretation": "Software demonstration only; does not establish calibration, market advantage or profitability.",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "synthetic_evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    rows = predictions[["date", "home_team", "away_team", "actual_idx"]].copy()
    for idx, name in enumerate(("home", "draw", "away")):
        rows[f"model_{name}"] = [float(p[idx]) for p in predictions.probs_model]
        rows[f"synthetic_market_{name}"] = [float(p[idx]) for p in predictions.probs_market]
    rows.to_csv(output / "synthetic_predictions.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation"))
    args = parser.parse_args()
    print(json.dumps(run_evaluation(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
