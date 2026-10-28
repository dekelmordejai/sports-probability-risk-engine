# Revisión de la edición pública

Fecha de revisión: 28 de octubre de 2026.

## Alcance de la publicación

La edición pública conserva los módulos de modelación, valoración, evaluación y persistencia, y añade un ejemplo temporal de scoring con baselines. Se publica como proyecto personal con asistencia de IA. Los métodos implementados se diferencian de una reproducción académica exacta y de una validación empírica con datos reales.

Se exportaron archivos de código y documentación a un repositorio con historia inicial nueva. No se trasladó el historial Git previo, las configuraciones remotas, credenciales, conversaciones, rutas de usuario, artefactos de ejecución, cachés o estados operativos. La configuración de banca e importes publicada es de ejemplo sintético.

## Cambios de revisión

- Presentación orientada a probabilidades, evaluación y riesgo, manteniendo explícita la aplicación a apuestas.
- Procesamiento por día en Pi-Ratings y features del meta-modelo; inicio del backtest en medianoche para excluir resultados intradía del entrenamiento.
- Pruebas contrafactuales de resultados futuros y de resultados del mismo día.
- Distinción entre holdout de calibración, selección de hiperparámetros y prueba final pendiente.
- Eliminación de afirmaciones de anti-perfilado, implementación académica exacta y ausencia universal de leakage.
- CI de pruebas y ejemplos sintéticos, sin tareas programadas, secretos de operación o publicación de artefactos con estados.
- Documentación de supuestos, fuentes, límites de reproducción y evidencia pendiente.

## Revisión de archivos

Se revisaron los nombres de archivos y contenido antes del primer commit. La comprobación estática busca firmas habituales de tokens, claves privadas, JSON de cuenta de servicio, asignaciones de credenciales y rutas locales. También se revisa que `data/` y `state/` contengan únicamente marcadores y que `.env.example` no tenga valores. Es una revisión acotada de publicación, no una auditoría de seguridad independiente ni una garantía de detección exhaustiva.

## Cómo revisar o extender

Reproducir la suite y el ejemplo del README, inspeccionar los casos temporales y ejecutar los adaptadores solo con datos y credenciales propios. Para proponer una mejora predictiva, documentar el corte temporal, las versiones de datos, el baseline, la cobertura, la calibración y los experimentos que condujeron a su selección.
