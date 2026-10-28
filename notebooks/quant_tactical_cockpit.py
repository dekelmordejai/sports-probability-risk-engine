# %%
"""Run in Jupyter/Colab from the repository root; imports have no network side effects."""
from pathlib import Path
import sys
import json

project = Path.cwd()
if not (project/"src").is_dir() and (project.parent/"src").is_dir():
    project = project.parent
if str(project) not in sys.path:
    sys.path.insert(0, str(project))


def load_colab_secrets():
    """Explicitly load configured secrets without printing their values."""
    import os
    from google.colab import userdata
    loaded = []
    for key in ["ODDS_API_KEY", "GOOGLE_SHEET_ID", "GCP_SA_CREDENTIALS_JSON",
                "NOTION_TOKEN", "NOTION_DATABASE_ID", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
        try:
            value = userdata.get(key)
        except userdata.SecretNotFoundError:
            continue
        if value:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def build_cockpit():
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from src.advanced_alpha_engines import InPlayStochasticEngine, PlayerPropsEngine, AmbiguityAverseKelly
    sport = widgets.Dropdown(options=["NBA", "NFL"], description="Deporte")
    pre = widgets.FloatText(value=-6.5, description="Spread previo")
    home = widgets.IntText(value=48, description="Puntos local")
    away = widgets.IntText(value=55, description="Puntos visita")
    elapsed = widgets.FloatSlider(value=.5, min=0, max=1, step=.01, description="Jugado")
    line = widgets.FloatText(value=2.5, description="Spread vivo")
    odds = widgets.FloatText(value=1.91, description="Cuota")
    sigma = widgets.FloatSlider(value=.035, min=0, max=.15, step=.005, description="Incertidumbre")
    minutes = widgets.FloatText(value=32, description="Minutos")
    rate = widgets.FloatText(value=.24, description="Tasa/minuto")
    prop_line = widgets.FloatText(value=7.5, description="Línea prop")
    output = widgets.Output()
    evaluate = widgets.Button(description="Evaluar en vivo", button_style="primary")
    prop = widgets.Button(description="Evaluar prop")
    snapshot_path = widgets.Text(value=str(project/"state/live/quant_engines_snapshot.json"), description="Estado diario")
    load_state = widgets.Button(description="Inspeccionar modelos")

    def show_snapshot(_):
        from src.state_registry import ProductionStateRegistry
        with output:
            clear_output()
            try:
                state = ProductionStateRegistry.load_daily_snapshot(snapshot_path.value)
                print(f"Entrenado: {state['timestamp_utc']} | Banca: {state['bankroll_mxn']:.2f} MXN")
                print("Fútbol:", ", ".join(state["soccer_engines"]))
                print("NFL:", ", ".join(getattr(state["nfl_engine"], "teams", [])))
                print("NBA:", ", ".join(getattr(state["nba_engine"], "team_states", {})))
            except (ValueError, FileNotFoundError) as error:
                print(str(error))

    def show_live(_):
        with output:
            clear_output()
            try:
                result = InPlayStochasticEngine.price_live_diffusion_nba_nfl(pre.value,
                    13.2 if sport.value == "NFL" else 11.85, home.value, away.value, elapsed.value, line.value)
                result.update(AmbiguityAverseKelly.compute_robust_fraction(result["P_Live_Home_Cover"], sigma.value, odds.value))
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except ValueError as error:
                print(str(error))

    def show_prop(_):
        with output:
            clear_output()
            try:
                print(json.dumps(PlayerPropsEngine.price_discrete_prop_nbinom(minutes.value, rate.value,
                      bookie_line=prop_line.value), indent=2))
            except ValueError as error:
                print(str(error))

    evaluate.on_click(show_live)
    prop.on_click(show_prop)
    load_state.on_click(show_snapshot)
    tabs = widgets.Tab(children=[widgets.VBox([sport, pre, home, away, elapsed, line, odds, sigma, evaluate]),
                                widgets.VBox([minutes, rate, prop_line, prop]), widgets.VBox([snapshot_path, load_state])])
    tabs.set_title(0, "En vivo")
    tabs.set_title(1, "Player props")
    tabs.set_title(2, "Producción")
    panel = widgets.VBox([widgets.HTML("<h3>Quant Sports — laboratorio táctico</h3><p>Escenarios de modelo; no ejecuta apuestas.</p>"), tabs, output])
    display(panel)
    return panel


# %%
if __name__ == "__main__":
    build_cockpit()
