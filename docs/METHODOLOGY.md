# Metodología de evaluación

## Objetivo y niveles de evidencia

Se distinguen tres preguntas: si el software respeta sus contratos, si las probabilidades predicen y se calibran fuera de muestra, y si una decisión sería ejecutable con valor económico. La suite y los ejemplos responden a la primera. El repositorio no publica un estudio de campo que resuelva las otras dos.

## Información disponible y cortes temporales

`WalkForwardBacktester` ordena partidos y predice por días. El entrenamiento de cada bloque utiliza fechas estrictamente anteriores a la medianoche que lo inicia. Los ratings y las features de xG se calculan antes de aplicar resultados del mismo día. El reentrenamiento es periódico y usa historia expansiva.

Se supone que los resultados de días previos ya estaban disponibles. Las fechas del dataset no equivalen necesariamente a timestamps de publicación: correcciones posteriores de estadísticas, partidos aplazados y latencias de cuotas requieren un dataset con información de disponibilidad para una evaluación real. No se afirma una garantía general de ausencia de leakage.

Las cuotas de cierre se utilizan para auditoría posterior de CLV y no como entrada a la selección del backtester. Las cuotas tomadas deben ser observaciones disponibles al decidir. El cargador distingue columnas de proveedores, pero por sí solo no acredita que una columna sea una cuota de apertura ejecutable.

El meta-modelo separa entrenamiento y calibración isotónica por fechas completas. Es un holdout temporal, no una validación cruzada completa. Sus probabilidades base deben producirse fuera de muestra antes de entrenarlo. El ajuste Optuna usa tres bloques cronológicos expansivos; su mejor puntuación es de selección de hiperparámetros, no una estimación final imparcial. Hace falta una prueba final posterior y no utilizada en selección.

## Ejemplo reproducible

```bash
python -m examples.offline_evaluation --output reports/evaluation
```

Usa 64 partidos generados con semilla 2026, 32 de calentamiento y reentrenamiento cada 14 días. Reporta todas las predicciones conocidas de las ventanas evaluadas, no solo selecciones que superan EV. Los equipos, marcadores, xG y cuotas del ejemplo son sintéticos.

Baselines: uniforme 1/3; frecuencias de resultados con suavizado Laplace calculadas solo sobre días previos; y probabilidades Shin de cuotas sintéticas. Se evalúan sobre las mismas filas que el modelo.

## Métricas

- **RPS:** media del error cuadrático acumulado sobre los dos umbrales de las clases ordenadas local/empate/visita, dividida por dos. Rango [0, 1].
- **Brier multiclase:** media de la suma de los tres errores cuadrados respecto a one-hot. Rango [0, 2], sin dividir por el número de clases.
- **Log loss:** logaritmo natural de la probabilidad de la clase observada, con la estabilización numérica de scikit-learn. Menor es mejor.
- **CLV:** comparación con probabilidad desmarginalizada de una muestra previa al inicio. Si falta una muestra en la ventana configurada, no se inventa un cierre.
- **VaR_95_MXN:** percentil 5 del PnL simulado con signo; es un escenario condicionado a las probabilidades de entrada, no una estimación validada de pérdidas futuras.

Los ejemplos no establecen intervalos de confianza ni mejoras significativas. Para un estudio real se requieren baseline temporal y de mercado, curvas de calibración, Brier/log loss/RPS, cobertura y abstenciones, intervalos por bloques temporales, sensibilidad a costes y límites y un registro completo de experimentos. El retorno económico se analiza solo después de esa validación, con cuotas y condiciones ejecutables.
