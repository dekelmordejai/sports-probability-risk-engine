import io
import requests
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path
from src.entity_resolver import EntityResolver


class SoccerDataLoader:
    """
    Pipeline ETL para Premier League (E0), La Liga (SP1) y Liga MX (MEX).
    Descarga resultados históricos, estadísticas de tiros y cuotas de apertura/cierre de Pinnacle.
    """

    # Altitudes oficiales (msnm) para estadios de Liga MX
    LIGA_MX_ALTITUDE = {
        "Toluca": 2660, "Pachuca": 2426, "Cruz Azul": 2240, "Club America": 2240,
        "America": 2240, "UNAM": 2240, "Pumas UNAM": 2240, "Puebla": 2135,
        "Necaxa": 1888, "Atl. San Luis": 1864, "San Luis": 1864, "Atlas": 1566,
        "Guadalajara": 1566, "Chivas": 1566, "Club Leon": 1815, "Leon": 1815,
        "Queretaro": 1820, "Santos Laguna": 1120, "Juarez": 1137, "Monterrey": 540,
        "U.A.N.L.- Tigres": 540, "Tigres": 540, "Tijuana": 20, "Mazatlan FC": 10, "Mazatlan": 10
    }

    def __init__(self, cache_dir="data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url_main = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
        self.base_url_extra = "https://www.football-data.co.uk/new/{league}.csv"

    @staticmethod
    def _estimate_proxy_xg(goals: np.ndarray, shots: np.ndarray, shots_on_target: np.ndarray) -> np.ndarray:
        """
        Estimador bayesiano empírico de xG cuando solo existen estadísticas de tiros (HST/AST, HS/AS).
        Pondera tiros a puerta (~0.28 xG), tiros fuera (~0.04 xG) y goles reales.
        """
        non_sot = np.clip(shots - shots_on_target, 0, None)
        shot_based_xg = 0.28 * shots_on_target + 0.035 * non_sot
        # Mezcla 75% calidad de volumen de tiro + 25% realización
        return np.clip(0.75 * shot_based_xg + 0.25 * goals, 0.05, None)

    def compute_altitude_boost(self, home_team: str, away_team: str, league: str, gamma_alt: float = 0.085) -> float:
        """Calcula el diferencial fisiológico de altitud para partidos de Liga MX."""
        if league != "MEX":
            return 0.0
        h_home = self.LIGA_MX_ALTITUDE.get(home_team, 1000)
        h_away = self.LIGA_MX_ALTITUDE.get(away_team, 1000)
        if h_home < 1800:
            return 0.0
        eff_diff_km = max(0.0, (h_home - max(h_away, 1200)) / 1000.0)
        return float(gamma_alt * eff_diff_km)

    def fetch_european_league(self, league_code: str, seasons: List[str]) -> pd.DataFrame:
        """
        Descarga múltiples temporadas de Premier League ('E0') o La Liga ('SP1').
        Formato de temporada: ['2223', '2324', '2425', '2526']
        """
        frames = []
        for s in seasons:
            url = self.base_url_main.format(season=s, league=league_code)
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            df_raw = pd.read_csv(io.StringIO(resp.text))
            df_raw["league"] = league_code
            frames.append(df_raw)

        if not frames:
            raise RuntimeError(f"No se pudieron descargar datos para {league_code}")
        df = pd.concat(frames, ignore_index=True)
        return self._standardize_dataframe(df, is_extra=False)

    def fetch_liga_mx(self) -> pd.DataFrame:
        """Descarga el histórico consolidado de Liga MX ('MEX')."""
        url = self.base_url_extra.format(league="MEX")
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError("Error al conectar con el repositorio de Liga MX.")
        df_raw = pd.read_csv(io.StringIO(resp.text))
        df_raw["league"] = "MEX"
        return self._standardize_dataframe(df_raw, is_extra=True)

    def _standardize_dataframe(self, df: pd.DataFrame, is_extra: bool = False) -> pd.DataFrame:
        """Normaliza columnas de fechas, marcadores, xG proxy y cuotas de apertura/cierre."""
        col_map = {
            "Date": "date",
            "Home": "home_team" if is_extra else None,
            "Away": "away_team" if is_extra else None,
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "HG": "home_goals" if is_extra else None,
            "AG": "away_goals" if is_extra else None,
            "FTHG": "home_goals",
            "FTAG": "away_goals",
        }
        for old_c, new_c in col_map.items():
            if new_c and old_c in df.columns:
                df = df.rename(columns={old_c: new_c})

        df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"]).copy()
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce", format="mixed")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        league_names = {"MEX": "Liga_MX", "E0": "EPL", "SP1": "La_Liga"}
        for col in ("home_team", "away_team"):
            df[col] = [EntityResolver.resolve(name, league_names.get(lg, lg)) or name
                       for name, lg in zip(df[col], df.league)]
        # Resultado real codificado: 0 = Local (H), 1 = Empate (D), 2 = Visita (A)
        df["actual_idx"] = np.where(
            df["home_goals"] > df["away_goals"], 0,
            np.where(df["home_goals"] == df["away_goals"], 1, 2)
        )

        # Construcción de xG proxy si existen columnas de tiros (HS, AS, HST, AST)
        if {"HS", "AS", "HST", "AST"}.issubset(df.columns):
            hs = df["HS"].fillna(df["home_goals"] * 8).values
            as_ = df["AS"].fillna(df["away_goals"] * 8).values
            hst = df["HST"].fillna(df["home_goals"] * 3).values
            ast = df["AST"].fillna(df["away_goals"] * 3).values
            df["home_xg"] = self._estimate_proxy_xg(df["home_goals"].values, hs, hst)
            df["away_xg"] = self._estimate_proxy_xg(df["away_goals"].values, as_, ast)
        else:
            df["home_xg"] = df["home_goals"].astype(float)
            df["away_xg"] = df["away_goals"].astype(float)

        # Extracción de cuotas de apertura/pre-partido (Pinnacle PSH/PSD/PSA o promedio AvgH/AvgD/AvgA)
        for prefix in ["PS", "B365", "Avg", "Max"]:
            cols = [f"{prefix}H", f"{prefix}D", f"{prefix}A"]
            if set(cols).issubset(df.columns):
                df["odds_h"] = pd.to_numeric(df[cols[0]], errors="coerce")
                df["odds_d"] = pd.to_numeric(df[cols[1]], errors="coerce")
                df["odds_a"] = pd.to_numeric(df[cols[2]], errors="coerce")
                break

        # Extracción de cuotas de cierre (Closing Line: PSCH/PSCD/PSCA o AvgCH/AvgCD/AvgCA)
        for prefix in ["PSC", "AvgC", "B365C"]:
            cols = [f"{prefix}H", f"{prefix}D", f"{prefix}A"]
            if set(cols).issubset(df.columns):
                df["close_h"] = pd.to_numeric(df[cols[0]], errors="coerce")
                df["close_d"] = pd.to_numeric(df[cols[1]], errors="coerce")
                df["close_a"] = pd.to_numeric(df[cols[2]], errors="coerce")
                break

        # Diferencial fisiológico de altitud
        df["alt_boost"] = [
            self.compute_altitude_boost(h, a, lg)
            for h, a, lg in zip(df["home_team"], df["away_team"], df["league"])
        ]

        for col in ["odds_h", "odds_d", "odds_a", "close_h", "close_d", "close_a"]:
            if col not in df:
                df[col] = np.nan
        return df.reset_index(drop=True)

    def fetch_with_parquet_cache(self, league_code: str, seasons: List[str], max_age_hours: int = 12) -> pd.DataFrame:
        import time
        if league_code not in ("MEX", "E0", "SP1") or max_age_hours < 0:
            raise ValueError("Invalid league or cache lifetime")
        periods = ["all"] if league_code == "MEX" else sorted(set(seasons))
        if not periods or any(x != "all" and (len(x) != 4 or not x.isdigit()) for x in periods):
            raise ValueError("Seasons must use four-digit codes such as 2526")
        frames = []
        for season in periods:
            cache = self.cache_dir / f"{league_code}_{season}.parquet"
            fresh = cache.exists() and (time.time()-cache.stat().st_mtime) <= max_age_hours*3600
            historical = season != "all" and int(season[:2])+2000 < pd.Timestamp.now().year-1
            if cache.exists() and (fresh or historical):
                frame = pd.read_parquet(cache)
            else:
                frame = self.fetch_liga_mx() if league_code == "MEX" else self.fetch_european_league(league_code, [season])
                if frame.empty:
                    raise RuntimeError(f"No completed matches for {league_code}/{season}")
                temporary = cache.with_suffix(".tmp.parquet")
                frame.to_parquet(temporary, index=False)
                temporary.replace(cache)
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True)
        return result.drop_duplicates(["date", "home_team", "away_team"]).sort_values("date").reset_index(drop=True)
