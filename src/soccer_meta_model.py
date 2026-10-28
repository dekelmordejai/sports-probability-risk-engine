import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from typing import Dict, List, Tuple


class SoccerMetaModel:
    """
    Meta-modelo apilado (Stacking) que combina las salidas de Dixon-Coles, Pi-Ratings,
    regresión a la media de xG (EWMA), fatiga por días de descanso y microestructura de Shin,
    aplicando calibración isotónica en un bloque temporal posterior al entrenamiento.
    Las features de los modelos base deben generarse fuera de muestra por el usuario.
    """

    FEATURE_COLS = [
        "dc_prob_h", "dc_prob_d", "dc_prob_a",     # Probabilidades base de Dixon-Coles
        "dc_lambda", "dc_mu",                      # Tasas de Poisson esperadas
        "pi_expected_gd",                          # Diferencia de goles proyectada por Pi-Ratings
        "xg_overperf_diff",                        # Señal de regresión a la media (Goles - xG)
        "rest_days_diff",                          # Ventaja de descanso físico (días)
        "alt_boost",                               # Estrés hipóxico (Liga MX)
        "shin_z",                                  # Parámetro de insiders del mercado
        "market_entropy"                           # Entropía de Shannon del partido
    ]

    def __init__(self, ewma_span: int = 6):
        self.ewma_span = ewma_span
        self.lgb_model = None
        self.calibrators: List[IsotonicRegression] = []

    def build_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula variables temporales de regresión a la media (Goles - xG) y días de descanso
        utilizando información de días anteriores, con actualización al final del día.
        """
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values(["date", "home_team", "away_team"], kind="stable")

        # Rastreo secuencial por equipo de días de descanso y sobre-rendimiento de xG
        last_date: Dict[str, pd.Timestamp] = {}
        xg_res_history: Dict[str, List[float]] = {}

        rest_diff_list = []
        xg_overperf_diff_list = []
        alpha = 2.0 / (self.ewma_span + 1.0)

        for _, day in data.groupby(data["date"].dt.normalize(), sort=True):
            for row in day.itertuples():
                h, a, d = row.home_team, row.away_team, row.date
                rest_h = min(14.0, max(2.0, (d - last_date[h]).days)) if h in last_date else 7.0
                rest_a = min(14.0, max(2.0, (d - last_date[a]).days)) if a in last_date else 7.0
                rest_diff_list.append(rest_h - rest_a)
                res_h = xg_res_history.get(h, [0.0])[-1]
                res_a = xg_res_history.get(a, [0.0])[-1]
                xg_overperf_diff_list.append(res_h - res_a)
            for row in day.itertuples():
                h, a, d = row.home_team, row.away_team, row.date
                res_h = xg_res_history.get(h, [0.0])[-1]
                res_a = xg_res_history.get(a, [0.0])[-1]
                last_date[h] = d
                last_date[a] = d
                cur_diff_h = float(row.home_goals - row.home_xg)
                cur_diff_a = float(row.away_goals - row.away_xg)
                xg_res_history.setdefault(h, []).append(alpha * cur_diff_h + (1.0 - alpha) * res_h)
                xg_res_history.setdefault(a, []).append(alpha * cur_diff_a + (1.0 - alpha) * res_a)

        data["rest_days_diff"] = rest_diff_list
        data["xg_overperf_diff"] = xg_overperf_diff_list
        return data

    def fit(self, meta_df: pd.DataFrame, calibration_fraction: float = 0.25):
        """
        Entrena LightGBM en el primer (1 - calibration_fraction) bloque cronológico y
        ajusta los regresores isotónicos en el bloque final para evitar sobreajuste de probabilidades.
        """
        clean_df = meta_df.dropna(subset=self.FEATURE_COLS + ["actual_idx"]).sort_values("date")
        n_total = len(clean_df)
        split_idx = int(n_total * (1.0 - calibration_fraction))
        if 0 < split_idx < n_total:
            cutoff = clean_df.iloc[split_idx]["date"]
            split_idx = int((clean_df["date"] < cutoff).sum())

        train_sub = clean_df.iloc[:split_idx]
        calib_sub = clean_df.iloc[split_idx:]
        if not 0 < calibration_fraction < 1 or train_sub.empty or calib_sub.empty:
            raise ValueError("Training and temporal calibration partitions must be non-empty")
        if set(train_sub.actual_idx) != {0, 1, 2}:
            raise ValueError("Training must include home, draw and away outcomes")

        X_train = train_sub[self.FEATURE_COLS].values
        y_train = train_sub["actual_idx"].values.astype(int)

        X_calib = calib_sub[self.FEATURE_COLS].values
        y_calib = calib_sub["actual_idx"].values.astype(int)

        # LightGBM regularizado para muestras deportivas con relación señal/ruido baja
        self.lgb_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=120,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=35,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.5,
            reg_lambda=2.0,
            random_state=42,
            verbose=-1,
            n_jobs=1
        )
        self.lgb_model.fit(X_train, y_train)

        # Calibración Isotónica One-vs-Rest sobre el conjunto temporal de validación
        raw_calib_probs = self.lgb_model.predict_proba(X_calib)
        self.calibrators = []
        for c in range(3):
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            y_bin = (y_calib == c).astype(float)
            iso.fit(raw_calib_probs[:, c], y_bin)
            self.calibrators.append(iso)

        return self

    def predict_calibrated_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Genera probabilidades 1X2 calibradas y renormalizadas en el símplex de probabilidad.
        """
        if self.lgb_model is None or len(self.calibrators) != 3:
            raise RuntimeError("Fit the meta-model before prediction")
        X = X_df[self.FEATURE_COLS].values
        raw_probs = self.lgb_model.predict_proba(X)

        cal_probs = np.column_stack([
            self.calibrators[c].predict(raw_probs[:, c]) for c in range(3)
        ])
        # Mezcla conservadora: 60% Meta-Modelo Calibrado + 40% Motor Estocástico Base (Dixon-Coles)
        dc_base = X_df[["dc_prob_h", "dc_prob_d", "dc_prob_a"]].values
        blended = 0.60 * cal_probs + 0.40 * dc_base
        return blended / np.sum(blended, axis=1, keepdims=True)
