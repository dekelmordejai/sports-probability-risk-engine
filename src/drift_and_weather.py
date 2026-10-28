import requests
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class CUSUMDriftAndKillSwitch:
    """
    Control Estadístico de Procesos (CUSUM) y Cortafuegos de Drawdown.
    Monitorea el CLV móvil y el Drawdown de la banca para modular la fracción de Kelly
    o activar automáticamente el modo Paper Trading si se supera el límite de riesgo.
    """

    def __init__(self, max_drawdown_pct: float = 20.0, min_rolling_clv_pct: float = -0.5, cusum_threshold: float = 5.0, cusum_allowance: float = 0.25):
        self.max_drawdown_pct = max_drawdown_pct
        self.min_rolling_clv_pct = min_rolling_clv_pct
        self.cusum_threshold = cusum_threshold
        self.cusum_allowance = cusum_allowance

    def evaluate_portfolio_health(self, ledger_df: pd.DataFrame, current_bankroll: float, peak_bankroll: float) -> Dict:
        if not np.isfinite(current_bankroll + peak_bankroll) or current_bankroll < 0 or peak_bankroll <= 0:
            raise ValueError("Bankrolls must be finite, current nonnegative and peak positive")
        drawdown_pct = ((peak_bankroll - current_bankroll) / peak_bankroll * 100.0) if peak_bankroll > 0 else 0.0
        kill_switch_active = drawdown_pct >= self.max_drawdown_pct

        league_multipliers: Dict[str, float] = {}
        cusum_scores = {}
        if not ledger_df.empty and "Liga" in ledger_df.columns and "CLV_%" in ledger_df.columns:
            clean_clv = ledger_df.dropna(subset=["CLV_%"]).copy()
            if "Fecha_UTC" in clean_clv:
                clean_clv = clean_clv.sort_values("Fecha_UTC")
            clean_clv["CLV_%"] = pd.to_numeric(clean_clv["CLV_%"], errors="coerce")
            for lg, grp in clean_clv.groupby("Liga"):
                recent_30 = grp.tail(30)["CLV_%"].dropna()
                cusum = 0.0
                for value in recent_30:
                    cusum = min(0.0, cusum + value + self.cusum_allowance)
                cusum_scores[str(lg)] = float(cusum)
                if len(recent_30) >= 10 and (recent_30.mean() < self.min_rolling_clv_pct or -cusum >= self.cusum_threshold):
                    # Reducir exposición a la mitad en ligas con deriva negativa de CLV
                    league_multipliers[lg] = 0.50
                else:
                    league_multipliers[lg] = 1.00

        return {
            "cusum_scores": cusum_scores,
            "drawdown_pct": round(max(0., drawdown_pct), 2),
            "kill_switch_paper_mode": bool(kill_switch_active),
            "league_kelly_multipliers": league_multipliers
        }


class StadiumWeatherOracle:
    """
    Conector gratuito (sin API Key) a Open-Meteo para obtener viento (mph)
    y precipitación (mm) en la coordenada GPS del estadio a la hora del encuentro.
    """

    STADIUM_COORDS: Dict[str, Tuple[float, float]] = {
        # NFL Open-Air Stadiums clave
        "BUF": (42.7738, -78.7870), "KC": (39.0489, -94.4839), "GB": (44.5013, -88.0622),
        "CHI": (41.8623, -87.6167), "CLE": (41.5061, -81.6995), "DEN": (39.7439, -105.0201),
        "NE": (42.0909, -71.2643),  "NYG": (40.8128, -74.0742), "PHI": (39.9008, -75.1675),
        "PIT": (40.4468, -80.0158), "SF": (37.4032, -121.9698), "SEA": (47.5952, -122.3316),
        "BAL": (39.2780, -76.6227),
        # Liga MX / Europa seleccionados
        "Toluca": (19.2863, -99.6669), "UNAM": (19.3319, -99.1926), "America": (19.3029, -99.1505),
        "Pachuca": (20.1053, -98.7561)
    }

    @classmethod
    def fetch_stadium_weather(cls, home_team: str, commence_time=None) -> Dict:
        unavailable = {"available": False, "wind_mph": None, "precip_mm": None,
                       "total_scoring_penalty": 0.0}
        if home_team not in cls.STADIUM_COORDS:
            return {**unavailable, "reason": "Unknown or enclosed stadium"}
        lat, lon = cls.STADIUM_COORDS[home_team]
        params = {"latitude": lat, "longitude": lon, "wind_speed_unit": "mph", "timezone": "UTC"}
        params["hourly" if commence_time else "current"] = "precipitation,wind_speed_10m"
        try:
            response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=8)
            response.raise_for_status()
            payload = response.json()
            if commence_time:
                times = pd.to_datetime(payload["hourly"]["time"], utc=True)
                target = pd.Timestamp(commence_time)
                target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
                delta = abs(times-target)
                idx = int(delta.argmin())
                if delta[idx] > pd.Timedelta(hours=1):
                    return {**unavailable, "reason": "Kickoff outside forecast horizon"}
                wind = float(payload["hourly"]["wind_speed_10m"][idx])
                rain = float(payload["hourly"]["precipitation"][idx])
            else:
                wind = float(payload["current"]["wind_speed_10m"])
                rain = float(payload["current"]["precipitation"])
            if not np.isfinite(wind + rain) or min(wind, rain) < 0:
                raise ValueError("Invalid forecast")
        except (requests.RequestException, ValueError, TypeError, KeyError, IndexError):
            return {**unavailable, "reason": "Weather provider unavailable or malformed response"}
        penalty = max(0., (wind-15)*.25) + (1.25 if rain > 2 else 0.)
        return {"available": True, "wind_mph": round(wind, 1), "precip_mm": round(rain, 2),
                "total_scoring_penalty": round(penalty, 2), "penalty_is_heuristic": True}
