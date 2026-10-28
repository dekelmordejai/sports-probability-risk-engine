# Ficha de modelos y límites

**Uso previsto:** proyecto personal de investigación en probabilidades deportivas y simulación de decisiones. Aplicación explícita a valoración de apuestas, sin API de colocación de apuestas. Estado: prototipo con pruebas de software y ejemplos sintéticos.

| Componente | Salida | Supuestos y límites |
|---|---|---|
| Dixon–Coles adaptado | Matriz de goles, 1X2, totales y hándicaps | Corrección de marcadores bajos; mezcla de goles/xG y altitud modifica el modelo original; truncamiento y normalización de rejilla |
| Pi-Ratings adaptado | Diferencia esperada y features | Actualización por día, mezcla 50/50 de goles y xG; no réplica exacta de la publicación |
| Meta-modelo | Mezcla LightGBM/isotónica con modelo base | Requiere features base fuera de muestra y posterior prueba independiente; módulo de investigación |
| NFL | Distribución discreta de diferencia de puntos | Kalman sobre EPA agregado; pesos de números clave y dispersión son configurables; sin modelo entrenado de totales NFL |
| NBA | Probabilidades de spread y total con push | Eficiencia/posesiones EWMA y aproximaciones distribucionales; bajas y contexto dependen de entradas correctas |
| Props, cópula y modelos en vivo | Escenarios parametrizados | Parámetros suministrados por el operador; sin validación histórica publicada |

## Datos y cobertura

Los adaptadores contemplan EPL, La Liga, Liga MX, NFL y NBA; cobertura efectiva depende de la fuente y la disponibilidad de equipos/mercados. Football-Data puede aportar un proxy de xG basado en tiros y goles, no xG por evento. Cuando falta, se usan goles como fallback. Los coeficientes de altitud, descanso, clima y otras heurísticas no se presentan como efectos causales estimados.

## Riesgo y ejecución

Kelly y EV dependen de probabilidades estimadas. Un EV calculado positivo puede deberse a error de modelo, observaciones desfasadas o condiciones no ejecutables. La simulación comparte las patas reutilizadas entre simples y parlays, pero supone independencia entre eventos distintos; no modela todas las correlaciones reales. La cópula de investigación no corrige automáticamente esa suposición del flujo diario.

Los límites de exposición y drawdown reducen importes según reglas, pero no garantizan límites de pérdida en la realidad. Los parlays generados combinan cuotas de simples y pueden no estar disponibles en una misma casa. El software no garantiza beneficios, calibración, latencia o continuidad de proveedores.

## Evidencia disponible y siguiente evaluación

Hay pruebas de normalización, masas de push, restricciones monetarias, liquidación, persistencia, errores de proveedores simulados y separación temporal. Hay un ejemplo de scoring sintético con baselines. Quedan pendientes benchmark histórico real con holdout final, calibración por liga/temporada, incertidumbre estadística, análisis de sensibilidad y verificación operativa de integraciones con credenciales del usuario.
