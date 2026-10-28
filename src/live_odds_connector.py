import os
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class LiveOddsConnector:
    """
    Cliente para The-Odds-API (v4) con motor de Line Shopping integrado.
    Separa la cuota Sharp (Pinnacle) para auditoría de microestructura y CLV
    de la mejor cuota ejecutable (Best Market Price) para maximizar el +EV.
    """

    SPORT_KEYS = {
        "EPL": "soccer_epl",
        "La_Liga": "soccer_spain_la_liga",
        "Liga_MX": "soccer_mexico_ligamx",
        "NFL": "americanfootball_nfl",
        "NBA": "basketball_nba"
    }

    def __init__(self, api_key: Optional[str] = None, regions: str = "eu,us,uk"):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        self.regions = regions
        self.base_url = "https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        self.usage_remaining = None

    def fetch_league_odds(self, league_name: str, markets: str = "h2h,spreads,totals") -> List[Dict]:
        """
        Descarga la cartelera activa y todas las cuotas de las casas para una liga.
        Consume cuotas en formato decimal.
        """
        if not self.api_key:
            raise ValueError("Falta ODDS_API_KEY en las variables de entorno.")

        sport_key = self.SPORT_KEYS[league_name]
        url = self.base_url.format(sport_key=sport_key)
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": markets,
            "oddsFormat": "decimal"
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        self.usage_remaining = resp.headers.get("x-requests-remaining")
        return resp.json()

    @staticmethod
    def extract_best_and_sharp_h2h(event_json: Dict, is_three_way: bool = True) -> Optional[Dict]:
        """
        Extrae para un partido:
        1. Las cuotas de Pinnacle (o el promedio del mercado si Pinnacle aún no abrió) para Shin/CLV.
        2. La cuota máxima disponible (Line Shopping) y qué casa de apuestas la ofrece.
        """
        home_team = event_json["home_team"]
        away_team = event_json["away_team"]
        commence_time = event_json["commence_time"]
        bookmakers = event_json.get("bookmakers", [])

        if not bookmakers:
            return None

        outcomes_order = [home_team, "Draw", away_team] if is_three_way else [home_team, away_team]
        best_odds = [1.01] * len(outcomes_order)
        best_books = ["N/A"] * len(outcomes_order)
        pinnacle_odds = None
        all_book_matrix = []

        for book in bookmakers:
            book_title = book["title"]
            book_key = book["key"]
            for mkt in book.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                price_map = {o["name"]: float(o["price"]) for o in mkt["outcomes"]}
                if not all(k in price_map for k in outcomes_order):
                    continue

                current_vec = [price_map[k] for k in outcomes_order]
                if not all(np.isfinite(x) and x > 1 for x in current_vec):
                    continue
                all_book_matrix.append(current_vec)

                if book_key == "pinnacle":
                    pinnacle_odds = current_vec

                # Line Shopping: conservar el precio más alto por cada vía
                for idx, price in enumerate(current_vec):
                    if price > best_odds[idx]:
                        best_odds[idx] = price
                        best_books[idx] = book_title

        if not all_book_matrix:
            return None

        # Si Pinnacle no está presente en ese mercado, usar la mediana del mercado como referencia Sharp
        sharp_ref_odds = pinnacle_odds if pinnacle_odds else np.median(all_book_matrix, axis=0).tolist()

        return {
            "event_id": event_json["id"],
            "commence_time": commence_time,
            "home_team": home_team,
            "away_team": away_team,
            "outcomes": outcomes_order,
            "sharp_odds": sharp_ref_odds,
            "best_odds": best_odds,
            "best_books": best_books
        }

    @staticmethod
    def extract_main_spread_and_total(event_json: Dict) -> Dict:
        """
        Extrae la línea modal de Spread (Hándicap) y Total (Over/Under) junto con
        la mejor cuota disponible (Line Shopping) para partidos de NFL y NBA.
        """
        home_team = event_json["home_team"]
        away_team = event_json["away_team"]
        bookmakers = event_json.get("bookmakers", [])

        spread_points, total_points = [], []
        spread_records, total_records = [], []

        for book in bookmakers:
            b_title = book["title"]
            for mkt in book.get("markets", []):
                if mkt["key"] == "spreads":
                    outcomes = {o["name"]: o for o in mkt["outcomes"]}
                    if home_team in outcomes and away_team in outcomes:
                        pt = float(outcomes[home_team].get("point", 0.0))
                        spread_points.append(pt)
                        spread_records.append({
                            "book": b_title,
                            "key": book["key"],
                            "line": pt,
                            "home_odds": float(outcomes[home_team]["price"]),
                            "away_odds": float(outcomes[away_team]["price"])
                        })
                elif mkt["key"] == "totals":
                    outcomes = {o["name"]: o for o in mkt["outcomes"]}
                    if "Over" in outcomes and "Under" in outcomes:
                        pt = float(outcomes["Over"].get("point", 0.0))
                        total_points.append(pt)
                        total_records.append({
                            "book": b_title,
                            "key": book["key"],
                            "line": pt,
                            "over_odds": float(outcomes["Over"]["price"]),
                            "under_odds": float(outcomes["Under"]["price"])
                        })

        result = {"spread": None, "total": None}

        # Seleccionar la línea consenso (mediana) y buscar la mejor cuota en esa línea exacta
        if spread_points:
            vals, counts = np.unique(spread_points, return_counts=True)
            consensus_spread = float(vals[np.argmax(counts)])
            matching = [r for r in spread_records if abs(r["line"] - consensus_spread) < 1e-4]
            best_h = max(matching, key=lambda x: x["home_odds"])
            best_a = max(matching, key=lambda x: x["away_odds"])
            result["spread"] = {
                "sharp_odds": next(([r["home_odds"], r["away_odds"]] for r in matching if r["key"] == "pinnacle"),
                                   np.median([[r["home_odds"], r["away_odds"]] for r in matching], axis=0).tolist()),
                "line_home": consensus_spread,
                "best_odds": [best_h["home_odds"], best_a["away_odds"]],
                "best_books": [best_h["book"], best_a["book"]]
            }

        if total_points:
            vals, counts = np.unique(total_points, return_counts=True)
            consensus_total = float(vals[np.argmax(counts)])
            matching = [r for r in total_records if abs(r["line"] - consensus_total) < 1e-4]
            best_o = max(matching, key=lambda x: x["over_odds"])
            best_u = max(matching, key=lambda x: x["under_odds"])
            result["total"] = {
                "sharp_odds": next(([r["over_odds"], r["under_odds"]] for r in matching if r["key"] == "pinnacle"),
                                   np.median([[r["over_odds"], r["under_odds"]] for r in matching], axis=0).tolist()),
                "line_total": consensus_total,
                "best_odds": [best_o["over_odds"], best_u["under_odds"]],
                "best_books": [best_o["book"], best_u["book"]]
            }

        return result

    def has_upcoming_matches_within_hours(self, league_name: str, window_hours: float = 14.0) -> bool:
        """
        Consulta el endpoint gratuito /events (Costo = 0 créditos de API) para verificar
        si existen partidos programados dentro de las próximas `window_hours` horas.
        """
        from datetime import datetime, timezone
        if not self.api_key:
            raise ValueError("ODDS_API_KEY is required")
        sport_key = self.SPORT_KEYS[league_name]
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
        resp = requests.get(url, params={"apiKey": self.api_key}, timeout=15)
        resp.raise_for_status()
        now = datetime.now(timezone.utc)
        for ev in resp.json():
            commence = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            hours_diff = (commence - now).total_seconds() / 3600.0
            if 0 <= hours_diff <= window_hours:
                return True
        return False

    def fetch_completed_scores(self, league_name: str, days_from: int = 2) -> List[Dict]:
        """Descarga los marcadores oficiales finalizados de las últimas 48h para auto_settle."""
        sport_key = self.SPORT_KEYS[league_name]
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
        if not self.api_key or days_from not in (1, 2, 3):
            raise ValueError("Scores require ODDS_API_KEY and days_from between 1 and 3")
        params = {"apiKey": self.api_key, "daysFrom": days_from}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return [g for g in resp.json() if g.get("completed") is True]
