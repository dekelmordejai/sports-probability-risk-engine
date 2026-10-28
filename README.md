# Motor probabilístico de analítica deportiva y evaluación de riesgo

**Proyecto personal · Python · investigación reproducible**

[![CI](https://github.com/dekelmordejai/sports-probability-risk-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/dekelmordejai/sports-probability-risk-engine/actions/workflows/ci.yml)

Este proyecto estudia cómo transformar resultados deportivos y cuotas en probabilidades, comparar pronósticos fuera de muestra y simular decisiones bajo incertidumbre. Su aplicación incluye **apuestas deportivas**, mediante valoración de mercados y asignación hipotética de capital. Las ejecuciones predeterminadas son de **paper trading** y el software no coloca apuestas en casas.

El valor del proyecto está en la modelación, el diseño temporal de las evaluaciones, la integración de fuentes y la automatización verificable. **No se presenta como una estrategia rentable ni como un sistema predictivo validado con dinero real.** Los resultados publicados aquí son sintéticos y demuestran funcionamiento del software.

## Recorrido para revisar el proyecto

1. Ejecutar la demostración sin claves ni descargas de datos.
2. Consultar la [arquitectura](docs/ARCHITECTURE.md) y la [metodología](docs/METHODOLOGY.md).
3. Revisar las [capacidades y límites de los modelos](docs/MODEL_CARD.md).
4. Examinar las [pruebas](tests/) y el [informe de revisión pública](docs/PUBLIC_REVIEW.md).

## Ejecutar en local

Python 3.11 o 3.12. Desde la carpeta clonada:

```bash
python -m venv .venv
```

Activar con `.venv\Scripts\Activate.ps1` en PowerShell o `source .venv/bin/activate` en Linux/macOS. Después:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_daily_pipeline.py --demo --mode morning_scan
python -m examples.offline_evaluation --output reports/evaluation
```

La instalación descarga dependencias; las demostraciones posteriores usan fixtures sintéticos locales. El primer comando de demostración produce tickets hipotéticos multideporte y escenarios de riesgo. El segundo ejecuta un recorrido temporal de fútbol y compara RPS, Brier multiclase y log loss con tres referencias: probabilidades uniformes, frecuencias históricas expansivas y cuotas sintéticas desmarginalizadas por Shin. Valores menores de esas puntuaciones son mejores; superarlos en fixtures no acredita ventaja sobre mercados reales.

Los reportes se guardan en `reports/` y los estados en `state/`; no se versionan. Hay un [resultado sintético de ejemplo](examples/output/synthetic_evaluation.json) para inspección, con semilla y convención de métricas explícitas.

## Componentes

| Área | Implementación y alcance |
|---|---|
| Fútbol | Modelo de goles inspirado en Dixon–Coles, decaimiento temporal y ajuste opcional con proxy de xG; variantes de Pi-Ratings y meta-modelo LightGBM |
| NFL | Agregación de EPA, filtro Kalman y distribución discreta para spreads; coeficientes de escenario pendientes de calibración |
| NBA | Eficiencia y posesiones EWMA, contexto de descanso y aproximación de spreads y totales con masa de devolución |
| Cuotas | Normalización de entidades, eliminación de margen por Shin, EV y comparación de líneas equivalentes |
| Riesgo | Kelly fraccional con límites conjuntos, simulación Monte Carlo y dependencia de patas compartidas en parlays |
| Evaluación | Recorrido por días, ajuste expansivo de hiperparámetros y comparación con baselines; sin benchmark histórico real publicado |
| Ingeniería | Ledger SQLite idempotente, caché por temporada, snapshots JSON y pruebas de integración sin red |

El flujo diario usa los modelos base de fútbol, NFL y NBA. El meta-modelo, props, cópulas y escenarios en vivo son componentes de investigación; su presencia no implica que estén entrenados o integrados en todos los mercados.

## Validación y reproducibilidad

Las pruebas distinguen contratos matemáticos, consistencia de riesgo, liquidación idempotente y disponibilidad temporal de información. Las regresiones temporales comprueban que modificar resultados futuros no cambie pronósticos anteriores y que los partidos de un mismo día no incorporen resultados de ese día en sus features. Véanse [resultados de verificación](docs/VERIFICATION.md).

La integración continua se ejecuta en Python 3.11 y 3.12, con permisos de lectura y demostraciones sintéticas. Esta edición pública no contiene tareas programadas con credenciales ni publica estados o reportes de operación. Las dependencias mínimas están en `requirements.txt`; `requirements-lock-py312.txt` registra el entorno Windows/Python 3.12 utilizado en la revisión inicial y no es un lock universal.

## Datos, uso y límites

La banca y los importes de ejemplo son parámetros sintéticos de configuración. Los datos reales, sus licencias y el acceso a proveedores requieren revisión por quien los utilice. No se redistribuyen históricos de proveedores, modelos entrenados con datos privados, credenciales ni libros de operaciones.

El [manual de uso](docs/OPERATIONS.md) documenta los cinco modos, la consola y adaptadores opcionales para cuotas, Sheets, Notion y Telegram. `.env.example` contiene únicamente nombres vacíos; la aplicación lee el entorno del proceso y no carga automáticamente archivos `.env`.

Esta edición se desarrolló de forma iterativa con asistencia de IA. Las decisiones, supuestos y pruebas son inspeccionables; superar las pruebas verifica el comportamiento cubierto, no sustituye una evaluación estadística con datos reales. [Descripción breve para portafolio](docs/PORTFOLIO.md).

## Referencias

- Dixon y Coles (1997), [Modelling Association Football Scores and Inefficiencies in the Football Betting Market](https://doi.org/10.1111/1467-9876.00065). La implementación incorpora modificaciones y no es una réplica del estudio.
- Constantinou y Fenton, [pi-ratings: manuscrito de los autores](https://citeseerx.ist.psu.edu/document?doi=7f87b4c176d5a727c0f59caab0dcd83b732680f7&repid=rep1&type=pdf). Aquí se añade mezcla con xG y procesamiento por día.
- [Fuentes de datos y documentación de integraciones](docs/DATA_SOURCES.md).
