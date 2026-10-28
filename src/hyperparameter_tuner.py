"""Deterministic expanding-window tuning; every validation block follows its fit."""
import numpy as np
import pandas as pd
import optuna
from src.soccer_dixon_coles import DixonColesEngine
from src.quant_core import ModelEvaluator


class MonthlyBayesianTuner:
    @staticmethod
    def tune_soccer_hyperparameters(df_hist, n_trials=20):
        data = df_hist.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date").reset_index(drop=True)
        if len(data) < 30 or n_trials < 1:
            raise ValueError("At least 30 matches and one trial are required")
        dates = sorted(data.date.unique())
        if len(dates) < 10:
            raise ValueError("Need at least ten distinct match dates for temporal folds")
        boundaries = [dates[int(len(dates)*fraction)] for fraction in (.5, .67, .83)]

        def objective(trial):
            xi = trial.suggest_float("xi_decay", .0015, .0065)
            weight = trial.suggest_float("xg_weight", .35, .85)
            preds, outcomes = [], []
            for idx, cutoff in enumerate(boundaries):
                train = data[data.date < cutoff]
                val = data[data.date >= cutoff]
                if idx < len(boundaries)-1:
                    val = val[val.date < boundaries[idx+1]]
                engine = DixonColesEngine(xi=xi, xg_weight=weight).fit(train)
                for row in val.itertuples():
                    if row.home_team not in engine.teams or row.away_team not in engine.teams:
                        continue
                    preds.append(engine.predict_match_markets(row.home_team, row.away_team,
                                 alt_boost=getattr(row, "alt_boost", 0.))["1X2"])
                    actual = getattr(row, "actual_idx", 0 if row.home_goals > row.away_goals else (1 if row.home_goals == row.away_goals else 2))
                    outcomes.append(actual)
            if not preds:
                raise ValueError("No known-team validation matches")
            return ModelEvaluator.ranked_probability_score(np.vstack(preds), np.asarray(outcomes))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=2026))
        study.optimize(objective, n_trials=n_trials)
        return {**study.best_params, "validation_rps": float(study.best_value), "n_trials": n_trials}
