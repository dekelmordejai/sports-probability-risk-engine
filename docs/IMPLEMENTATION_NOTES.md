# Decisiones de ingeniería

1. **Importes:** redondeo siempre hacia abajo, límites conjuntos para simples y parlays y descuento de exposición pendiente.
2. **Probabilidades:** validación de dimensiones, finitud, suma y cuotas; soluciones con devolución se tratan como tres estados.
3. **Dependencia:** los parlays y sus simples comparten resultados de patas en la simulación. La independencia entre eventos distintos sigue siendo un supuesto.
4. **Persistencia:** identificadores estables, ledger SQLite y liquidación idempotente; snapshots JSON versionados con control de antigüedad.
5. **Auditoría:** CLV solo con muestras previas al inicio y ausencias visibles; CUSUM acumulativo y control de drawdown.
6. **Tiempo:** folds expansivos para tuning; predicciones por día con entrenamiento anterior; features publicadas antes de resultados del día. Se asume disponibilidad de los datos previos.
7. **Operación:** cinco modos con demostración local y red bloqueada en sus pruebas; sincronización y alertas requieren opciones explícitas.
8. **Compatibilidad:** `nfl_data_py` 0.3.2 evita la restricción de pandas<2 de 0.3.3; el calendario se carga por HTTPS desde nflverse. Las variantes futuras de proveedores pueden requerir mantenimiento.

El diseño es un prototipo modular, no un servicio desplegado con SLA. Los nombres `src`, `advanced_alpha_engines` y `humanize_stake_mxn` se conservan por compatibilidad; no implican alfa demostrado ni propiedades de ocultación del redondeo.
