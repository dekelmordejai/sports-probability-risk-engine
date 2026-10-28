import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


class EntityResolver:
    """
    Sistema de Resolución de Entidades (Entity Resolution) para estandarizar nombres
    de equipos entre The-Odds-API y los motores históricos (Football-Data, nflverse, nba_api).
    """

    # 1. Mapeo Canónico: Liga MX (The-Odds-API / Variantes -> ID Canónico del Motor)
    LIGA_MX_MAP: Dict[str, str] = {
        "Club America": "America", "América": "America", "CF America": "America", "America": "America",
        "Cruz Azul": "Cruz Azul", "CD Cruz Azul": "Cruz Azul",
        "Deportivo Toluca": "Toluca", "Toluca FC": "Toluca", "Toluca": "Toluca",
        "Pumas UNAM": "UNAM", "UNAM Pumas": "UNAM", "Club Universidad Nacional": "UNAM", "UNAM": "UNAM",
        "Guadalajara": "Guadalajara", "Chivas": "Guadalajara", "CD Guadalajara": "Guadalajara", "Chivas Guadalajara": "Guadalajara",
        "Tigres UANL": "U.A.N.L.- Tigres", "U.A.N.L.- Tigres": "U.A.N.L.- Tigres", "Tigres": "U.A.N.L.- Tigres",
        "Monterrey": "Monterrey", "CF Monterrey": "Monterrey", "Rayados": "Monterrey",
        "Club Leon": "Leon", "León": "Leon", "Leon": "Leon",
        "Pachuca": "Pachuca", "CF Pachuca": "Pachuca",
        "Atlas": "Atlas", "Atlas FC": "Atlas",
        "Santos Laguna": "Santos Laguna", "Club Santos Laguna": "Santos Laguna",
        "Atletico San Luis": "Atl. San Luis", "Atlético San Luis": "Atl. San Luis", "Atl. San Luis": "Atl. San Luis",
        "Necaxa": "Necaxa", "Club Necaxa": "Necaxa",
        "Puebla": "Puebla", "Club Puebla": "Puebla", "Puebla FC": "Puebla",
        "Queretaro": "Queretaro", "Querétaro FC": "Queretaro", "Queretaro FC": "Queretaro",
        "Mazatlan FC": "Mazatlan FC", "Mazatlán FC": "Mazatlan FC", "Mazatlan": "Mazatlan FC",
        "Club Tijuana": "Tijuana", "Tijuana Xoloitzcuintles": "Tijuana", "Tijuana": "Tijuana",
        "FC Juarez": "Juarez", "FC Juárez": "Juarez", "Juarez": "Juarez"
    }

    # 2. Mapeo Canónico: Premier League (The-Odds-API -> Football-Data 'E0')
    EPL_MAP: Dict[str, str] = {
        "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
        "AFC Bournemouth": "Bournemouth", "Brentford": "Brentford",
        "Brighton and Hove Albion": "Brighton", "Brighton & Hove Albion": "Brighton", "Brighton": "Brighton",
        "Chelsea": "Chelsea", "Crystal Palace": "Crystal Palace", "Everton": "Everton",
        "Fulham": "Fulham", "Ipswich Town": "Ipswich", "Ipswich": "Ipswich",
        "Leicester City": "Leicester", "Leicester": "Leicester", "Liverpool": "Liverpool",
        "Manchester City": "Man City", "Man City": "Man City",
        "Manchester United": "Man United", "Man United": "Man United",
        "Newcastle United": "Newcastle", "Newcastle": "Newcastle",
        "Nottingham Forest": "Nott'm Forest", "Nott'm Forest": "Nott'm Forest",
        "Southampton": "Southampton", "Tottenham Hotspur": "Tottenham", "Tottenham": "Tottenham",
        "West Ham United": "West Ham", "West Ham": "West Ham",
        "Wolverhampton Wanderers": "Wolves", "Wolves": "Wolves",
        "Leeds United": "Leeds", "Burnley": "Burnley", "Sunderland": "Sunderland", "Sheffield United": "Sheffield United"
    }

    # 3. Mapeo Canónico: La Liga (The-Odds-API -> Football-Data 'SP1')
    LA_LIGA_MAP: Dict[str, str] = {
        "Real Madrid": "Real Madrid", "Barcelona": "Barcelona", "FC Barcelona": "Barcelona",
        "Atletico Madrid": "Ath Madrid", "Atlético Madrid": "Ath Madrid", "Ath Madrid": "Ath Madrid",
        "Athletic Bilbao": "Ath Bilbao", "Athletic Club": "Ath Bilbao", "Ath Bilbao": "Ath Bilbao",
        "Real Sociedad": "Sociedad", "Sociedad": "Sociedad",
        "Real Betis": "Betis", "Betis": "Betis",
        "Villarreal": "Villarreal", "Villarreal CF": "Villarreal",
        "Valencia": "Valencia", "Valencia CF": "Valencia",
        "Sevilla": "Sevilla", "Sevilla FC": "Sevilla",
        "Girona": "Girona", "Girona FC": "Girona",
        "Celta Vigo": "Celta", "RC Celta": "Celta", "Celta": "Celta",
        "Osasuna": "Osasuna", "CA Osasuna": "Osasuna",
        "Mallorca": "Mallorca", "RCD Mallorca": "Mallorca",
        "Rayo Vallecano": "Vallecano", "Vallecano": "Vallecano",
        "Getafe": "Getafe", "Getafe CF": "Getafe",
        "Alaves": "Alaves", "Deportivo Alavés": "Alaves", "Deportivo Alaves": "Alaves",
        "Las Palmas": "Las Palmas", "UD Las Palmas": "Las Palmas",
        "Leganes": "Leganes", "CD Leganés": "Leganes", "CD Leganes": "Leganes",
        "Real Valladolid": "Valladolid", "Valladolid": "Valladolid",
        "Espanyol": "Espanol", "RCD Espanyol": "Espanol", "Espanol": "Espanol"
    }

    # 4. Mapeo Canónico: NFL (The-Odds-API Full Name -> nfl_data_py 2/3-letter code)
    NFL_MAP: Dict[str, str] = {
        "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
        "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
        "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
    }

    # 5. Mapeo Canónico: NBA (The-Odds-API Full Name -> nba_api 3-letter abbreviation)
    NBA_MAP: Dict[str, str] = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
        "Washington Wizards": "WAS"
    }

    LEAGUE_DICTIONARIES = {
        "Liga_MX": LIGA_MX_MAP,
        "EPL": EPL_MAP,
        "La_Liga": LA_LIGA_MAP,
        "NFL": NFL_MAP,
        "NBA": NBA_MAP
    }

    STOPWORDS = {"fc", "cf", "cd", "club", "deportivo", "afc", "ud", "rcd", "ca", "rc"}

    @classmethod
    def _clean_text(cls, name: str) -> str:
        """Elimina acentos, convierte a minúsculas y remueve prefijos deportivos genéricos."""
        nfkd = unicodedata.normalize("NFKD", str(name))
        ascii_str = "".join([c for c in nfkd if not unicodedata.combining(c)])
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", ascii_str.lower())
        tokens = [t for t in cleaned.split() if t not in cls.STOPWORDS]
        return " ".join(tokens) if tokens else cleaned.strip()

    @classmethod
    def resolve(
        cls,
        raw_name: str,
        league: str,
        valid_engine_teams: Optional[List[str]] = None,
        min_similarity: float = 0.82
    ) -> Optional[str]:
        """
        Resuelve el nombre de un equipo hacia el identificador exacto del motor.
        1. Búsqueda directa en el diccionario de la liga.
        2. Búsqueda normalizada sin acentos/afijos.
        3. Emparejamiento difuso contra `valid_engine_teams` si se proporciona.
        """
        league_map = cls.LEAGUE_DICTIONARIES.get(league, {})
        if raw_name in set(league_map.values()) and (valid_engine_teams is None or raw_name in valid_engine_teams):
            return raw_name

        # Paso 1: Coincidencia exacta en diccionario
        if raw_name in league_map:
            candidate = league_map[raw_name]
            if valid_engine_teams is None or candidate in valid_engine_teams:
                return candidate

        # Paso 2: Coincidencia tras limpieza léxica contra las llaves del diccionario
        clean_input = cls._clean_text(raw_name)
        for k, v in league_map.items():
            if cls._clean_text(k) == clean_input:
                if valid_engine_teams is None or v in valid_engine_teams:
                    return v

        # Paso 3: Fallback Difuso contra los equipos activos en el motor entrenado
        if valid_engine_teams:
            best_match = None
            second_score = 0.0
            best_score = 0.0
            input_tokens = set(clean_input.split())

            for eng_team in valid_engine_teams:
                clean_eng = cls._clean_text(eng_team)
                eng_tokens = set(clean_eng.split())

                # Similitud de secuencia (Ratcliff/Obershelp)
                seq_ratio = SequenceMatcher(None, clean_input, clean_eng).ratio()

                # Similitud de intersección de tokens
                if input_tokens and eng_tokens:
                    overlap = len(input_tokens.intersection(eng_tokens)) / min(len(input_tokens), len(eng_tokens))
                else:
                    overlap = 0.0

                combined_score = 0.65 * seq_ratio + 0.35 * overlap
                if combined_score > best_score:
                    second_score = best_score
                    best_score = combined_score
                    best_match = eng_team
                else:
                    second_score = max(second_score, combined_score)

            if best_score >= min_similarity and best_score - second_score >= 0.05:
                return best_match

        return None
