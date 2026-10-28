# Uso y adaptadores opcionales

La demostración pública usa datos sintéticos. Los adaptadores con datos reales son experimentales y requieren credenciales propias. No hay ejecución de apuestas mediante API.

## Instalación y primera ejecución

Requiere Python 3.11 o 3.12. Desde la raíz del repositorio:

```bash
python -m venv .venv
```

Activa el entorno con `.venv\Scripts\Activate.ps1` en PowerShell o `source .venv/bin/activate` en Linux/macOS. Después:

```bash
python -m pip install -r requirements.txt
pytest tests/test_quant_invariants.py -v --tb=short
pytest -q
python run_daily_pipeline.py --demo --mode morning_scan
python run_daily_pipeline.py --demo --mode tactical_steam
python run_daily_pipeline.py --demo --mode auto_settle
python run_daily_pipeline.py --demo --mode clv_audit
python run_daily_pipeline.py --demo --mode monthly_tune --trials 2
```

La primera demostración crea tres simples y tres parlays. Guarda CSV y JSON en `reports/demo/`, y modelos y libro mayor en `state/demo/`. Las ejecuciones repetidas no duplican apuestas; el saldo pendiente reduce el presupuesto disponible. Para repetir una demostración aislada, utiliza una nueva carpeta con `--state-dir` y `--report-dir`.

`requirements.txt` declara las dependencias de esta edición. `nfl_data_py` se fija en 0.3.2 porque 0.3.3 restringe pandas a versiones anteriores a 2. La carga NFL evita el antiguo origen HTTP de calendarios de ese paquete y utiliza el calendario HTTPS de nflverse. `requirements-lock-py312.txt` registra las versiones verificadas en Python 3.12; Python 3.11 resuelve desde `requirements.txt`.

## Qué incluye

| Componente | Implementación |
|---|---|
| Microestructura | Shin estable numéricamente, desmarginalización multiplicativa, Shannon, EV, middles con colas asimétricas |
| Portafolio | Kelly simultáneo exacto hasta 12 selecciones binarias u 8 con devolución; muestreo reproducible para carteras mayores; límites conjuntos y redondeo hacia abajo |
| Fútbol | Dixon–Coles, decaimiento temporal, mezcla de goles/xG, altitud y matriz de marcadores; Pi-Ratings; meta-modelo LightGBM con calibración temporal |
| NFL | EPA de jugadas, filtro Kalman, covarianza estabilizada, regresión entre temporadas, PMF con números clave y probabilidad de push |
| NBA | Posesiones y eficiencia EWMA, ritmo normalizado por tiempo extra, descanso, altitud e impacto parametrizado de bajas |
| Módulos avanzados | Kelly con intervalo de ambigüedad, props binomial negativa/Gamma, modelos en vivo y cópula gaussiana |
| Parlays | Combinaciones de 2–3 eventos distintos, clasificación por régimen, profit boost y techo monetario |
| Control | CUSUM negativo sobre CLV, límite de drawdown, exposición pendiente y modo paper |
| Operación | Cuotas y resultados, caché Parquet, ledger SQLite, Sheets, Notion, Telegram y snapshots JSON |
| Investigación | Backtest cronológico por día y Optuna con tres ventanas temporales expansivas |

El pipeline diario usa Dixon–Coles, Kalman y posesiones. Pi-Ratings, meta-modelo, cópulas, props y ajustes en vivo también se exponen como componentes de investigación y por la consola táctica; no se aplican automáticamente a todos los mercados. Los pesos de contexto, números clave, lesiones y pago anticipado son hipótesis configurables, pendientes de calibración empírica.

## Configuración y modos

`config/settings.yaml` define banca inicial de 25,000 MXN, quarter-Kelly, exposición total máxima del 15%, parlays hasta el 3%, EV mínimo del 2.5%, paso monetario de 50 MXN y cortafuegos al 20% de drawdown. El límite de parlays está contenido dentro del límite total. La exposición pendiente se descuenta antes de asignar nuevas cantidades.

| Modo | Acción |
|---|---|
| `morning_scan` | Liquida pendientes, consulta eventos próximos, entrena, valora mercados, genera tickets y guarda modelos |
| `tactical_steam` | Carga modelos recientes y compara observaciones de cuotas del mismo evento, mercado y selección |
| `clv_audit` | Calcula CLV únicamente con muestras guardadas de los 15 minutos anteriores al inicio |
| `auto_settle` | Procesa resultados finalizados y liquida simples/parlays sin repetir PnL |
| `monthly_tune` | Ajusta `xi_decay` y `xg_weight` por liga; guarda parámetros para el próximo entrenamiento |

Variables de entorno: usa los nombres de `.env.example`. El programa lee el entorno del proceso; no carga automáticamente archivos `.env`.

- **Cuotas:** `ODDS_API_KEY`, requerida en el escaneo con datos reales. Se consultan eventos antes de gastar créditos en cuotas. El coste de cuotas/resultados depende del proveedor, mercados y regiones.
- **Google Sheets:** `GOOGLE_SHEET_ID` y `GCP_SA_CREDENTIALS_JSON` con la cuenta de servicio que tenga acceso a esa hoja. Activa escritura con `--sync`. La pestaña `Ledger_Apuestas` se crea con los encabezados definidos en `src/persistence_sync.py`; los campos se identifican por nombre. La hoja `Config`, celda B2, puede consultarse mediante el método público de lectura de banca; el pipeline usa como fuente de banca el ledger y el valor inicial del YAML.
- **Notion:** `NOTION_TOKEN` y `NOTION_DATABASE_ID`, o `NOTION_DATA_SOURCE_ID` si la base tiene varias fuentes. La fuente debe tener `Name` (título), `Bet_ID` (texto), `EV` y `Monto_MXN` (número). Se usa la API `2025-09-03` y se evitan duplicados por identificador. Requiere `--sync`.
- **Telegram:** `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`. Solo `--notify` activa el envío durante el radar táctico.

```bash
python run_daily_pipeline.py --mode morning_scan
python run_daily_pipeline.py --mode morning_scan --sync
python run_daily_pipeline.py --mode tactical_steam --notify
```

Los datos reales se guardan en `state/live/` y `reports/live/`. `--demo` rechaza las opciones de sincronización, notificación y registro real. `--record-live` es una opción explícita para registrar como reales los importes de un ticket que el operador haya confirmado manualmente; no ejecuta apuestas y el cortafuegos la convierte en registro paper si se activa. No debe utilizarse para registrar pronósticos no ejecutados.

## Consola táctica y notebook

```bash
python run_tactical_cli.py --sport NBA --match "LAL @ DEN" --pre_spread -6.5 --h_score 48 --a_score 55 --elapsed 0.5 --live_line 2.5 --live_odds 1.91
python run_tactical_cli.py --mode discrete_prop --minutes 32 --rate 0.24 --line 7.5
python run_tactical_cli.py --mode gamma_prop --mean-yards 65 --line 64.5
python run_tactical_cli.py --mode soccer --minutes 60 --h_score 1 --a_score 1 --reds-away 1
python run_tactical_cli.py --mode sgp --p1 .55 --p2 .55 --rho .45 --live_odds 3.2
```

Si omites `--pre_spread`, la consola consulta el modelo del snapshot vigente. Los estados de más de 36 horas se rechazan. `--commit --event-id ID` registra una evaluación en paper; añade `--sync` para reflejarla en Sheets. Para obtener un identificador válido usa el `Event_ID` del reporte diario.

Ejecuta `notebooks/quant_tactical_cockpit.py` en Jupyter o Colab desde la raíz clonada. Contiene controles de escenarios en vivo, props y lectura de modelos diarios. `load_colab_secrets()` es una llamada explícita; no imprime valores. Para compartir el estado con Colab, descarga el artefacto de Actions y coloca su contenido en `state/live/`.

## Reglas de cálculo y límites de cobertura

- Un evento aporta como máximo una selección a simples. Esto reduce la dependencia dentro del mismo encuentro, pero no demuestra independencia entre partidos.
- La simulación conjunta reutiliza el resultado de cada pata en simples y todos los parlays que la contienen. `VaR_95_MXN` usa la siguiente convención: percentil 5 del PnL **con signo**, no una pérdida positiva. Las métricas Monte Carlo dependen de las probabilidades estimadas.
- El redondeo siempre baja y respeta tanto el importe calculado como el máximo permitido. No se afirma que este redondeo evite que una casa identifique patrones.
- Los mercados con devolución se calculan con tres estados: victoria, pérdida y push. El pipeline diario incluye spreads y totales enteros con esas probabilidades; NBA aproxima la masa entera mediante intervalos de medio punto. Los parlays automáticos y las alertas tácticas usan patas sin push. La consola de difusión en vivo muestra líneas enteras, pero no les asigna stake antes del final. El pipeline NFL selecciona spreads; no dispone de un modelo entrenado de totales NFL.
- La captura de datos históricos usa estadísticas de tiros para un **proxy** de xG cuando no hay xG real. El meta-modelo requiere features de Dixon–Coles generadas fuera de muestra; suministrar predicciones entrenadas sobre el mismo partido invalida su evaluación.
- El ajuste de pago anticipado, los coeficientes de clima y las regresiones en vivo son aproximaciones de escenario. No se incorporan como ventajas verificadas en los tickets automáticos. El clima desconocido se marca como desconocido, sin inventar observaciones.
- La auditoría nocturna no puede reconstruir una cuota de cierre que nunca se capturó. Ejecuta el radar antes del inicio para guardar muestras. Si no hay una dentro de los 15 minutos previos, el CLV permanece vacío y el informe cuenta las muestras ausentes. Los horarios base de Actions no garantizan cubrir todos los cierres.
- `/scores` entrega hasta tres días recientes; eventos no cubiertos o sin marcador permanecen pendientes. El sistema no infiere resultados de partidos cancelados, aplazados, props o contratos de pago anticipado.
- Los parlays son combinaciones teóricas de cuotas de simples. Su disponibilidad, cuota final y condiciones del bono deben verificarse en la casa antes de una ejecución manual. No se presume que sean ofertas ejecutables de una misma casa.
- El caché se separa por liga y temporada y reutiliza temporadas cerradas. La latencia depende de descargas, entrenamiento y entorno; las pruebas offline no acreditan tiempos de producción.
