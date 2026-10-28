import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional


class NBAPossessionEngine:
    """
    Motor Estocástico de Difusión de Posesiones para la NBA.
    Integra Pace x Efficiency, Four Factors (EWMA), impacto de bajas a nivel de jugador (EPM)
    y Tensor de Fatiga (Back-to-Back, 3-en-4, husos horarios y altitud en Denver/Utah).
    """

    # Coordenadas de huso horario (UTC offset estándar) y altitud (msnm) para franquicias clave
    ARENA_CONTEXT = {
        "DEN": {"tz": -7, "alt": 1609}, "UTA": {"tz": -7, "alt": 1288}, "PHX": {"tz": -7, "alt": 331},
        "LAL": {"tz": -8, "alt": 93},   "LAC": {"tz": -8, "alt": 93},   "GSW": {"tz": -8, "alt": 10},
        "SAC": {"tz": -8, "alt": 9},    "POR": {"tz": -8, "alt": 15},   "DAL": {"tz": -6, "alt": 131},
        "HOU": {"tz": -6, "alt": 15},   "SAS": {"tz": -6, "alt": 198},  "OKC": {"tz": -6, "alt": 366},
        "MIN": {"tz": -6, "alt": 253},  "MEM": {"tz": -6, "alt": 103},  "NOP": {"tz": -6, "alt": 2},
        "MIL": {"tz": -6, "alt": 188},  "CHI": {"tz": -6, "alt": 181},  "BOS": {"tz": -5, "alt": 14},
        "NYK": {"tz": -5, "alt": 10},   "BKN": {"tz": -5, "alt": 30},   "PHI": {"tz": -5, "alt": 12},
        "MIA": {"tz": -5, "alt": 2},    "ORL": {"tz": -5, "alt": 25},   "ATL": {"tz": -5, "alt": 320},
        "CLE": {"tz": -5, "alt": 199},  "IND": {"tz": -5, "alt": 218},  "DET": {"tz": -5, "alt": 183},
        "TOR": {"tz": -5, "alt": 76},   "CHA": {"tz": -5, "alt": 229},  "WAS": {"tz": -5, "alt": 20}
    }

    def __init__(
        self,
        league_avg_pace: float = 99.4,
        league_avg_ortg: float = 115.2,
        base_hca_pts: float = 2.15,
        base_spread_std: float = 11.85,
        base_total_std: float = 15.60,
        ewma_span: int = 12
    ):
        self.league_avg_pace = league_avg_pace
        self.league_avg_ortg = league_avg_ortg
        self.base_hca_pts = base_hca_pts
        self.base_spread_std = base_spread_std
        self.base_total_std = base_total_std
        self.ewma_span = ewma_span
        if min(league_avg_pace, league_avg_ortg, base_spread_std, base_total_std, ewma_span) <= 0:
            raise ValueError("NBA scales and EWMA span must be positive")
        self.team_states: Dict[str, Dict] = {}

    @staticmethod
    def fetch_nba_api_gamelogs(season: str = "2025-26") -> pd.DataFrame:
        """
        Descarga los registros por partido desde nba_api y calcula Posesiones,
        Pace, ORTG, DRTG y Four Factors.
        """
        from nba_api.stats.endpoints import leaguegamelog

        log = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation="T", timeout=30).get_data_frames()[0]
        log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"])

        # Estimación estándar de posesiones de Dean Oliver
        log["POSS"] = (
            log["FGA"]
            + 0.44 * log["FTA"]
            - 0.96 * log["OREB"]
            + log["TOV"]
        )
        log["EFG_PCT"] = (log["FGM"] + 0.5 * log["FG3M"]) / log["FGA"].clip(lower=1)
        log["TOV_PCT"] = log["TOV"] / log["POSS"].clip(lower=1)
        log["FTR"] = log["FTA"] / log["FGA"].clip(lower=1)

        # Cruzar equipo local y visitante por GAME_ID
        home_df = log[log["MATCHUP"].str.contains("vs.", regex=False)].copy()
        away_df = log[log["MATCHUP"].str.contains("@")].copy()

        merged = pd.merge(
            home_df, away_df, on=["GAME_ID", "GAME_DATE"], suffixes=("_H", "_A")
        ).sort_values("GAME_DATE").reset_index(drop=True)

        possessions = 0.5 * (merged["POSS_H"] + merged["POSS_A"])
        minutes = merged["MIN_H"].astype(float)
        game_minutes = np.where(minutes > 100, minutes / 5, minutes)
        merged["PACE"] = possessions * 48 / np.maximum(game_minutes, 48)
        merged["ORTG_H"] = (merged["PTS_H"] / possessions) * 100.0
        merged["ORTG_A"] = (merged["PTS_A"] / possessions) * 100.0
        merged["ORB_PCT_H"] = merged["OREB_H"] / (merged["OREB_H"] + merged["DREB_A"]).clip(lower=1)
        merged["ORB_PCT_A"] = merged["OREB_A"] / (merged["OREB_A"] + merged["DREB_H"]).clip(lower=1)

        return merged.rename(columns={
            "GAME_DATE": "date", "TEAM_ABBREVIATION_H": "home_team", "TEAM_ABBREVIATION_A": "away_team",
            "PTS_H": "home_pts", "PTS_A": "away_pts", "PACE": "pace",
            "ORTG_H": "home_ortg", "ORTG_A": "away_ortg"
        })

    def fit_historical_logs(self, df: pd.DataFrame):
        """
        Actualiza secuencialmente los estados EWMA de Pace, ORTG, DRTG, Four Factors
        y el historial reciente de fechas jugadas por cada franquicia.
        """
        alpha = 2.0 / (self.ewma_span + 1.0)
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_localize(None)
        data = data.sort_values("date")

        for row in data.itertuples():
            h, a, d = row.home_team, row.away_team, pd.to_datetime(row.date)

            for team in (h, a):
                if team not in self.team_states:
                    self.team_states[team] = {
                        "pace": self.league_avg_pace,
                        "ortg": self.league_avg_ortg,
                        "drtg": self.league_avg_ortg,
                        "four_factors_net": 0.0,
                        "recent_dates": []
                    }

            st_h = self.team_states[h]
            st_a = self.team_states[a]

            # Calcular diferencial de Four Factors si las columnas están disponibles
            ff_h = getattr(row, "ff_net_h", (row.home_ortg - row.away_ortg) * 0.15)
            ff_a = -ff_h

            # Actualizar EWMA post-partido (Shrinkage hacia el estado previo)
            st_h["pace"] = alpha * row.pace + (1.0 - alpha) * st_h["pace"]
            st_h["ortg"] = alpha * row.home_ortg + (1.0 - alpha) * st_h["ortg"]
            st_h["drtg"] = alpha * row.away_ortg + (1.0 - alpha) * st_h["drtg"]
            st_h["four_factors_net"] = alpha * ff_h + (1.0 - alpha) * st_h["four_factors_net"]
            st_h["recent_dates"].append(d)

            st_a["pace"] = alpha * row.pace + (1.0 - alpha) * st_a["pace"]
            st_a["ortg"] = alpha * row.away_ortg + (1.0 - alpha) * st_a["ortg"]
            st_a["drtg"] = alpha * row.home_ortg + (1.0 - alpha) * st_a["drtg"]
            st_a["four_factors_net"] = alpha * ff_a + (1.0 - alpha) * st_a["four_factors_net"]
            st_a["recent_dates"].append(d)

        return self

    def compute_fatigue_penalty(self, team: str, opponent_home: str, match_date: pd.Timestamp, is_home: bool) -> float:
        """
        Evalúa el Tensor de Fatiga (en puntos de eficiencia) para un equipo en match_date:
        - Back-to-Back (1 día desde el último juego)
        - 3 partidos en 4 noches (3-in-4)
        - Estrés de altitud en Denver / Utah para visitantes
        """
        st = self.team_states.get(team, {})
        dates = [d for d in st.get("recent_dates", []) if d < match_date]
        penalty = 0.0

        if dates:
            days_rest = (match_date - dates[-1]).days
            if days_rest == 1:
                # Penalización B2B más severa si el equipo además viaja de visitante
                penalty -= 2.25 if not is_home else 1.55
            elif days_rest >= 3:
                penalty += 0.45  # Beneficio por piernas frescas

            # Verificar densidad 3-in-4 (dos partidos jugados en los últimos 3 días previos)
            games_last_3d = sum(1 for d in dates if 1 <= (match_date - d).days <= 3)
            if games_last_3d >= 2:
                penalty -= 0.90

        # Ajuste por hipoxia en Denver (1609m) y Utah (1288m) para visitantes
        if not is_home and opponent_home in self.ARENA_CONTEXT:
            arena_alt = self.ARENA_CONTEXT[opponent_home]["alt"]
            team_alt = self.ARENA_CONTEXT.get(team, {"alt": 100})["alt"]
            if arena_alt >= 1200 and team_alt < 600:
                penalty -= 1.15 if arena_alt > 1500 else 0.75

        return float(penalty)

    @staticmethod
    def compute_injury_impact(missing_players: Optional[List[Dict]] = None) -> float:
        """
        Cuantifica el impacto en puntos de bajas confirmadas (Load Management / lesiones).
        Cada elemento de missing_players: {'name': str, 'epm': float, 'proj_mins': float}
        donde 'epm' es Estimated Plus-Minus (ej. +6.5 para un MVP, +2.5 para un titular élite).
        """
        if not missing_players:
            return 0.0
        replacement_epm = -1.8  # Nivel de reemplazo estándar de banca profunda
        usage_redistribution = 0.78
        total_delta = 0.0
        for p in missing_players:
            net_epm_loss = max(0.0, p.get("epm", 0.0) - replacement_epm)
            min_share = np.clip(p.get("proj_mins", 0.0) / 48.0, 0.0, 0.85)
            total_delta -= min_share * net_epm_loss * usage_redistribution
        return float(total_delta)

    def price_nba_markets(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        bookie_spread_home: float,
        bookie_total_line: float,
        home_injuries: Optional[List[Dict]] = None,
        away_injuries: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Calcula las distribuciones gaussianas escaladas por ritmo para el diferencial
        de puntos (Spread y Moneyline) y la suma de puntos (Over/Under).
        """
        m_date = pd.to_datetime(match_date, utc=True).tz_localize(None)
        st_h = self.team_states[home_team]
        st_a = self.team_states[away_team]

        # 1. Ritmo proyectado del partido (Pace)
        exp_pace = st_h["pace"] + st_a["pace"] - self.league_avg_pace
        pace_scale = exp_pace / 100.0

        # 2. Eficiencias base + Four Factors residual
        eff_h_raw = st_h["ortg"] + (st_a["drtg"] - self.league_avg_ortg) + 0.20 * st_h["four_factors_net"]
        eff_a_raw = st_a["ortg"] + (st_h["drtg"] - self.league_avg_ortg) + 0.20 * st_a["four_factors_net"]

        # 3. Integrar Tensor de Fatiga e Impacto de Bajas (Injuries)
        fat_h = self.compute_fatigue_penalty(home_team, home_team, m_date, is_home=True)
        fat_a = self.compute_fatigue_penalty(away_team, home_team, m_date, is_home=False)

        inj_h = self.compute_injury_impact(home_injuries)
        inj_a = self.compute_injury_impact(away_injuries)

        # Puntos esperados finales de Local y Visitante
        mu_home = pace_scale * eff_h_raw + 0.5 * self.base_hca_pts + fat_h + inj_h
        mu_away = pace_scale * eff_a_raw - 0.5 * self.base_hca_pts + fat_a + inj_a

        # Media y desviación estándar del Diferencial (Local - Visita) y del Total (Local + Visita)
        mu_margin = float(mu_home - mu_away)
        mu_total = float(mu_home + mu_away)

        sigma_margin = float(self.base_spread_std * np.sqrt(pace_scale))
        sigma_total = float(self.base_total_std * np.sqrt(pace_scale))

        # Probabilidades analíticas sobre el proceso de difusión
        # Para cubrir bookie_spread_home (ej. -5.5), necesitamos: Margin + (-5.5) > 0 => Margin > 5.5
        target_margin = -bookie_spread_home
        p_home_cover = float(1.0 - norm.cdf(target_margin, loc=mu_margin, scale=sigma_margin))
        p_home_ml = float(1.0 - norm.cdf(0.0, loc=mu_margin, scale=sigma_margin))

        # Totales (Over / Under)
        p_over = float(1.0 - norm.cdf(bookie_total_line, loc=mu_total, scale=sigma_total))
        p_under = 1.0 - p_over
        p_spread_push = 0.
        p_total_push = 0.
        if float(bookie_spread_home).is_integer():
            p_spread_push = float(norm.cdf(target_margin+.5, mu_margin, sigma_margin)-norm.cdf(target_margin-.5, mu_margin, sigma_margin))
            p_home_cover = float(norm.sf(target_margin+.5, mu_margin, sigma_margin))
        if float(bookie_total_line).is_integer():
            p_total_push = float(norm.cdf(bookie_total_line+.5, mu_total, sigma_total)-norm.cdf(bookie_total_line-.5, mu_total, sigma_total))
            p_over = float(norm.sf(bookie_total_line+.5, mu_total, sigma_total))
            p_under = 1-p_over-p_total_push

        return {
            "Match": f"{away_team} @ {home_team}",
            "Proj_Pace": round(exp_pace, 2),
            "Proj_Score": f"{mu_home:.1f} - {mu_away:.1f}",
            "Fair_Spread_Home": round(-mu_margin, 2),
            "Fair_Total": round(mu_total, 2),
            "Fatigue_Diff": round(fat_h - fat_a, 2),
            "Injury_Diff": round(inj_h - inj_a, 2),
            "P_Home_Cover": p_home_cover,
            "P_Away_Cover": 1.0 - p_home_cover - p_spread_push,
            "P_Push": p_spread_push,
            "P_Total_Push": p_total_push,
            "P_Home_ML": round(p_home_ml, 4),
            "P_Away_ML": round(1.0 - p_home_ml, 4),
            "P_Over": p_over,
            "P_Under": p_under
        }
