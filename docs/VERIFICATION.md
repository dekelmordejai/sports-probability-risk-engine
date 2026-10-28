# Verificación reproducible

La edición pública conserva las 72 pruebas de invariantes e integración de su base y añade cinco regresiones temporales y de scoring. El 28 de octubre de 2026 se ejecutó la suite completa en Windows/Python 3.12: **77 pruebas aprobadas en 13.54 segundos**, sin pruebas omitidas. La evidencia de automatización pública está disponible en [GitHub Actions](https://github.com/dekelmordejai/sports-probability-risk-engine/actions/workflows/ci.yml).

## Entorno y alcance

Python 3.11 y 3.12 son las versiones objetivo. La comprobación local usa Windows y el entorno aislado de Python 3.12; CI ejecuta la matriz en Ubuntu 24.04. `requirements-lock-py312.txt` describe el entorno Windows de referencia, mientras CI instala los mínimos declarados.

```bash
python -m pytest -q
python run_daily_pipeline.py --demo --mode morning_scan
python -m examples.offline_evaluation --output reports/evaluation
```

La suite verifica contratos de software y matemáticas, modelos con datos sintéticos, adaptadores con respuestas simuladas y los cinco modos offline. No se usaron credenciales reales ni se enviaron alertas. Las métricas del ejemplo temporal se basan en datos generados, no son un benchmark de deporte real ni un registro de rentabilidad.

## Resultado del ejemplo sintético

64 partidos generados, 32 de calentamiento y 32 de evaluación. Clase: local/empate/visita. Resultados redondeados solo para mostrar; los valores completos están en `examples/output/synthetic_evaluation.json`.

| Modelo o referencia | RPS | Brier multiclase | Log loss |
|---|---:|---:|---:|
| Dixon–Coles + Pi | 0.21879 | 0.72558 | 1.19537 |
| Uniforme | 0.21007 | 0.66667 | 1.09861 |
| Frecuencias expansivas | 0.21426 | 0.70473 | 1.16221 |
| Cuotas sintéticas Shin | 0.20908 | 0.67678 | 1.10862 |

Menor es mejor. En este fixture el modelo no mejora al baseline uniforme en ninguna de las tres métricas. Se conserva ese resultado: la demostración acredita el flujo de evaluación y hace visible una comparación desfavorable. No permite inferir capacidad predictiva o económica sobre deporte real. Diferencias numéricas pequeñas entre plataformas son posibles por optimización y librerías.
