"""Idempotent SQLite ledger with optional Google Sheets and Notion mirrors."""
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from src.validation import vector


LEDGER_COLUMNS = ["Bet_ID", "Fecha_UTC", "Deporte", "Liga", "Partido", "Mercado", "Selección",
                  "Prob_Modelo", "Cuota_Tomada", "Monto_MXN", "Evento_ID", "Inicio_UTC",
                  "Casa", "Estado", "Prob_Cierre", "CLV_%", "PnL_MXN", "Paper", "Patas_JSON", "Profit_Boost", "Liquidada_UTC"]


def money(value):
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def match_teams(match_name):
    if " vs " in match_name:
        return tuple(match_name.split(" vs ", 1))
    if " @ " in match_name:
        away, home = match_name.split(" @ ", 1)
        return home, away
    raise ValueError("Match must use 'Home vs Away' or 'Away @ Home'")


def settle_selection(match_name, market, selection, home_score, away_score):
    """Return a result; unknown/unsupported selections remain pending."""
    home, away = match_teams(match_name)
    vector([home_score, away_score], "scores", 0)
    if int(home_score) != home_score or int(away_score) != away_score:
        raise ValueError("Scores must be integers")
    if market in ("1X2", "Moneyline", "ML"):
        if selection not in (home, away, "Empate", "Draw"):
            return "PENDIENTE"
        if home_score == away_score:
            if market != "1X2":
                return "PUSH"
            return "GANADA" if selection in ("Empate", "Draw") else "PERDIDA"
        return "GANADA" if selection == (home if home_score > away_score else away) else "PERDIDA"
    if "Spread" in market or "Total" in market:
        try:
            picked, line_text = selection.rsplit(" ", 1)
            line = float(line_text)
        except (ValueError, TypeError):
            return "PENDIENTE"
        if not np.isfinite(line):
            return "PENDIENTE"
        if "Spread" in market:
            if picked not in (home, away):
                return "PENDIENTE"
            margin = home_score-away_score if picked == home else away_score-home_score
            net = margin+line
        else:
            if picked not in ("Over", "Under"):
                return "PENDIENTE"
            net = (home_score+away_score-line)*(1 if picked == "Over" else -1)
        return "PUSH" if abs(net) < 1e-8 else ("GANADA" if net > 0 else "PERDIDA")
    return "PENDIENTE"


def settle_record(record, completed_matches):
    if str(record.get("Estado", "")).upper() != "PENDIENTE":
        return None
    stake, odd = float(record["Monto_MXN"]), float(record["Cuota_Tomada"])
    vector([stake], "stake", 0)
    vector([odd], "odds", 1, strict_lower=True)
    legs = json.loads(record.get("Patas_JSON") or "[]")
    if legs:
        returns = []
        for leg in legs:
            key = leg.get("Event_ID") or leg["Match"]
            if key not in completed_matches:
                return None
            result = settle_selection(leg["Match"], leg["Market"], leg["Selection"], *completed_matches[key])
            if result == "PENDIENTE":
                return None
            returns.append(float(leg["Odds"]) if result == "GANADA" else (1. if result == "PUSH" else 0.))
        gross = float(np.prod(returns))
        if gross > 1:
            gross = 1+(gross-1)*(1+float(record.get("Profit_Boost") or 0))
        status = "GANADA" if gross > 1 else ("PUSH" if gross == 1 else "PERDIDA")
        return status, round(stake*(gross-1), 2)
    key = record.get("Evento_ID") or record["Partido"]
    if key not in completed_matches:
        return None
    status = settle_selection(record["Partido"], record["Mercado"], record["Selección"], *completed_matches[key])
    if status == "PENDIENTE":
        return None
    pnl = Decimal(str(stake))*(Decimal(str(odd))-1)
    return status, money(pnl) if status == "GANADA" else (0. if status == "PUSH" else -money(stake))


class QuantPersistenceSync:
    def __init__(self, db_path="state/ledger.sqlite", external_enabled=False):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.sheet_id = os.getenv("GOOGLE_SHEET_ID")
        self.gcp_creds_raw = os.getenv("GCP_SA_CREDENTIALS_JSON")
        self.notion_token, self.notion_db = os.getenv("NOTION_TOKEN"), os.getenv("NOTION_DATABASE_ID")
        self.notion_source = os.getenv("NOTION_DATA_SOURCE_ID")
        self.external_enabled = external_enabled
        with self._db() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS bets (bet_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def _db(self):
        return sqlite3.connect(self.db_path)

    def _get_gspread_client(self):
        import gspread
        from google.oauth2.service_account import Credentials
        if not self.sheet_id or not self.gcp_creds_raw:
            raise ValueError("Google Sheets credentials are not configured")
        creds = Credentials.from_service_account_info(json.loads(self.gcp_creds_raw), scopes=["https://www.googleapis.com/auth/spreadsheets"])
        return gspread.authorize(creds)

    def ledger_dataframe(self):
        with self._db() as conn:
            records = [json.loads(row[0]) for row in conn.execute("SELECT payload FROM bets ORDER BY rowid")]
        records = [{**dict.fromkeys(LEDGER_COLUMNS, ""), **rec} for rec in records]
        return pd.DataFrame(records, columns=LEDGER_COLUMNS)

    def _store(self, rec):
        with self._db() as conn:
            conn.execute("INSERT INTO bets VALUES (?, ?) ON CONFLICT(bet_id) DO UPDATE SET payload=excluded.payload",
                         (rec["Bet_ID"], json.dumps(rec, ensure_ascii=False, allow_nan=False)))

    def bankroll_state(self, initial_bankroll=25000., paper=False):
        vector([initial_bankroll], "bankroll", 0, strict_lower=True)
        df = self.ledger_dataframe()
        if df.empty:
            return initial_bankroll, initial_bankroll, 0.
        df = df[df.Paper == paper]
        settlement_order = df.Liquidada_UTC.where(df.Liquidada_UTC != "", df.Fecha_UTC)
        df = df.assign(_settlement_order=settlement_order).sort_values("_settlement_order")
        pnl = pd.to_numeric(df.PnL_MXN, errors="coerce").fillna(0)
        equity = initial_bankroll+pnl.cumsum()
        current = initial_bankroll+float(pnl.sum())
        peak = max(initial_bankroll, float(equity.max())) if len(equity) else initial_bankroll
        pending = float(df.loc[df.Estado == "PENDIENTE", "Monto_MXN"].sum())
        return current, peak, pending

    def fetch_current_bankroll_from_sheets(self, default_bankroll=25000.):
        if self.external_enabled and self.sheet_id and self.gcp_creds_raw:
            ws = self._get_gspread_client().open_by_key(self.sheet_id).worksheet("Config")
            value = float(ws.acell("B2").value)
            vector([value], "bankroll", 0, strict_lower=True)
            return value
        return self.bankroll_state(default_bankroll)[0]

    def append_ticket(self, ticket_df, paper=True, parlays=False):
        """Record paper fills or explicitly confirmed manual fills; no bookmaker API."""
        existing = set(self.ledger_dataframe().Bet_ID)
        inserted = 0
        for row in ticket_df.to_dict("records"):
            if row["Monto_MXN"] <= 0:
                continue
            if parlays:
                signature = "|".join(sorted(row["Leg_IDs"]))
                match, market, selection = "Parlay", "Parlay", row["Selecciones_Combinadas"]
                league, sport, event = "Multi", "Multi", signature
                prob, odds, legs = row["Prob_Conjunta_%"]/100, row["Cuota_Combinada"], row["Legs"]
            else:
                signature = row["Selection_ID"]
                match, market, selection = row["Match"], row["Market"], row["Selection"]
                league, sport, event = row["League"], row["Sport"], row.get("Event_ID", "")
                prob, odds, legs = row["Prob_Model"], row["Odds"], []
            bet_id = hashlib.sha256(f"{paper}|{signature}".encode()).hexdigest()[:24]
            if bet_id in existing:
                continue
            rec = dict.fromkeys(LEDGER_COLUMNS, "")
            rec.update(Bet_ID=bet_id, Fecha_UTC=datetime.now(timezone.utc).isoformat(), Deporte=sport, Liga=league,
                       Partido=match, Mercado=market, Prob_Modelo=prob, Cuota_Tomada=odds,
                       Monto_MXN=float(row["Monto_MXN"]), Evento_ID=event, Inicio_UTC=row.get("Commence_Time", ""),
                       Casa=row.get("Bookmaker", ""), Estado="PENDIENTE", PnL_MXN=0., Paper=bool(paper),
                       Patas_JSON=json.dumps(legs, ensure_ascii=False), Profit_Boost=row.get("Profit_Boost", 0.))
            rec["Selección"] = selection
            self._store(rec)
            existing.add(bet_id)
            inserted += 1
        return inserted

    def append_ticket_to_sheets(self, ticket_df, paper=True):
        self.append_ticket(ticket_df, paper=paper)
        return self.sync_ledger_to_sheets()

    def sync_ledger_to_sheets(self):
        if not self.external_enabled or not self.sheet_id or not self.gcp_creds_raw:
            return 0
        import gspread
        book = self._get_gspread_client().open_by_key(self.sheet_id)
        try:
            ws = book.worksheet("Ledger_Apuestas")
        except gspread.WorksheetNotFound:
            ws = book.add_worksheet(title="Ledger_Apuestas", rows=2000, cols=len(LEDGER_COLUMNS))
        existing = ws.get_all_records()
        headers = ws.row_values(1)
        if not headers:
            ws.append_row(LEDGER_COLUMNS, value_input_option="RAW")
            headers = LEDGER_COLUMNS
        if not set(LEDGER_COLUMNS).issubset(headers):
            raise ValueError("Ledger_Apuestas headers differ from the documented schema")
        by_id = {str(r["Bet_ID"]): i for i, r in enumerate(existing, 2)}
        updates, additions = [], []
        for record in self.ledger_dataframe().to_dict("records"):
            values = [record.get(key, "") for key in headers]
            if record["Bet_ID"] in by_id:
                row_num = by_id[record["Bet_ID"]]
                updates.append({"range": f"A{row_num}", "values": [values]})
            else:
                additions.append(values)
        if updates:
            ws.batch_update(updates, value_input_option="RAW")
        if additions:
            ws.append_rows(additions, value_input_option="RAW")
        return len(updates)+len(additions)

    def auto_settle_pending_bets(self, completed_matches):
        if self.external_enabled and self.sheet_id and self.gcp_creds_raw:
            records = self._get_gspread_client().open_by_key(self.sheet_id).worksheet("Ledger_Apuestas").get_all_records()
            known = set(self.ledger_dataframe().Bet_ID)
            for rec in records:
                if rec.get("Bet_ID") and rec["Bet_ID"] not in known:
                    rec["Paper"] = str(rec.get("Paper", "True")).lower() == "true"
                    self._store(rec)
        changed = 0
        for rec in self.ledger_dataframe().to_dict("records"):
            result = settle_record(rec, completed_matches)
            if result is not None:
                rec["Estado"], rec["PnL_MXN"] = result
                rec["Liquidada_UTC"] = datetime.now(timezone.utc).isoformat()
                self._store(rec)
                changed += 1
        self.sync_ledger_to_sheets()
        return changed

    def audit_clv(self, closing_probabilities):
        changed = 0
        for rec in self.ledger_dataframe().to_dict("records"):
            key = (rec["Evento_ID"], rec["Mercado"], rec["Selección"])
            if key in closing_probabilities and rec["Prob_Cierre"] == "":
                p = float(closing_probabilities[key])
                vector([p], "closing probability", 0, 1)
                rec["Prob_Cierre"] = p
                rec["CLV_%"] = (p*float(rec["Cuota_Tomada"])-1)*100
                self._store(rec)
                changed += 1
        self.sync_ledger_to_sheets()
        return changed

    def publish_ticket_to_notion(self, ticket_df):
        if not self.external_enabled or not self.notion_token or not (self.notion_db or self.notion_source):
            return 0
        headers = {"Authorization": f"Bearer {self.notion_token}", "Notion-Version": "2025-09-03"}
        source = self.notion_source
        if not source:
            response = requests.get(f"https://api.notion.com/v1/databases/{self.notion_db}", headers=headers, timeout=20)
            response.raise_for_status()
            sources = response.json().get("data_sources", [])
            if len(sources) != 1:
                raise ValueError("Set NOTION_DATA_SOURCE_ID when a database has zero or multiple sources")
            source = sources[0]["id"]
        count = 0
        for row in ticket_df.to_dict("records"):
            identifier = str(row["Selection_ID"])
            query = requests.post(f"https://api.notion.com/v1/data_sources/{source}/query", headers=headers,
                                  json={"filter": {"property": "Bet_ID", "rich_text": {"equals": identifier}}}, timeout=20)
            query.raise_for_status()
            if query.json().get("results"):
                continue
            payload = {"parent": {"data_source_id": source}, "properties": {
                "Name": {"title": [{"text": {"content": f"{row['Match']} | {row['Selection']}"}}]},
                "Bet_ID": {"rich_text": [{"text": {"content": identifier}}]},
                "EV": {"number": float(row["EV_%"])}, "Monto_MXN": {"number": float(row["Monto_MXN"])}}}
            response = requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            count += 1
        return count
