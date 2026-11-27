# Fase E · Resultados de test de V1 en el dominio en L

En la resolución zero-shot N=64, la red entrenada con el estimador reduce la mediana del error relativo en energía por factores de **13,0 y 67,5** frente a la malla uniforme, para p=2 y p=3. Con la pérdida de Ritz, los factores son **16,6 y 75,9**. Los órdenes 32→64 del estimador son **1,72 y 2,32**. En los 16 pares de comparación, el Algoritmo 1 con el estimador alcanza órdenes **2,03 y 2,94** sobre la misma caja de logits y el mismo suelo físico.

Cada una de las 16 redes usó 280 pares de entrenamiento, 60 de validación, T=5, suelo físico de 10⁻⁷ y cuatro niveles de entrenamiento (4, 8, 16 y 32). La entrada dependiente del nivel es log(d+1/N). La selección de V1 usó la validación de la fase D, incluido N=64. En el test la red usa el checkpoint congelado de N=32 para predecir los nudos en N=64, seguido de una solución de Galerkin. El Algoritmo 1 optimiza sus propios logits para cada par y nivel.

## Error e índice de efectividad

Medianas sobre 60 pares de test, con cuatro semillas por red (240 observaciones por grado, nivel y pérdida). La uniforme se calcula una vez por par (60 observaciones). El factor de mejora es el cociente de medianas. El índice divide η por el error **absoluto** en norma de energía.

| p | N | Error uniforme | Error red η² | Error red Ritz | Factor η² | Factor Ritz | Ieff uniforme | Ieff η² | Ieff Ritz |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 0,09778 | 0,09537 | 0,09664 | 1,03 | 1,01 | 8,84 | 9,01 | 10,3 |
| 2 | 8 | 0,04608 | 0,03369 | 0,0321 | 1,37 | 1,44 | 7,14 | 9,06 | 9,87 |
| 2 | 16 | 0,02602 | 0,009143 | 0,008764 | 2,85 | 2,97 | 6,09 | 9,57 | 10,3 |
| 2 | 32 | 0,01566 | 0,00243 | 0,002212 | 6,44 | 7,08 | 5,84 | 9,25 | 10,6 |
| 2 | 64 | 0,009581 | 0,0007367 | 0,000576 | 13 | 16,6 | 5,79 | 8,54 | 10,6 |
| 3 | 4 | 0,04869 | 0,06772 | 0,04938 | 0,719 | 0,986 | 11,1 | 13,2 | 13,9 |
| 3 | 8 | 0,02914 | 0,0212 | 0,01933 | 1,37 | 1,51 | 9,26 | 17,9 | 17,9 |
| 3 | 16 | 0,01771 | 0,003869 | 0,003471 | 4,58 | 5,1 | 8,95 | 13,4 | 14,2 |
| 3 | 32 | 0,01086 | 0,0004934 | 0,0004759 | 22 | 22,8 | 8,92 | 8,77 | 11 |
| 3 | 64 | 0,006671 | 9,888e−5 | 8,787e−5 | 67,5 | 75,9 | 8,92 | 9,16 | 10,9 |

Los órdenes de extremo a extremo usan log₂(error₄/error₆₄)/4. El ajuste de la malla uniforme sobre N=16, 32 y 64 da 0,721 y 0,704. Los percentiles 25 y 75 de cada error están en la tabla de datos. [Errores y efectividad](tables/energy_effectivity.csv) · [Órdenes](tables/orders.csv)

| p | Método | Orden 4→64 | Orden 32→64 |
|---|---|---|---|
| 2 | Red η² | 1,75 | 1,72 |
| 2 | Red Ritz | 1,85 | 1,94 |
| 2 | Uniforme | 0,838 | 0,709 |
| 3 | Red η² | 2,35 | 2,32 |
| 3 | Red Ritz | 2,28 | 2,44 |
| 3 | Uniforme | 0,717 | 0,703 |

## Comparación emparejada con el Algoritmo 1

Medianas sobre los mismos 16 pares de test. Cada red aporta cuatro semillas por par. Cada optimización por instancia aporta un resultado por pérdida. Los cambios de población entre esta tabla y la anterior explican sus distintas medianas uniformes.

| p | N | Uniforme | Alg. 1 η² | Red η² | Red / Alg. 1 η² | Alg. 1 Ritz | Red Ritz | Red / Alg. 1 Ritz |
|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 0,09746 | 0,08962 | 0,09666 | 1,079 | 0,08052 | 0,09951 | 1,236 |
| 2 | 8 | 0,05064 | 0,03346 | 0,03453 | 1,032 | 0,02808 | 0,03307 | 1,178 |
| 2 | 16 | 0,02975 | 0,009812 | 0,009387 | 0,9566 | 0,008486 | 0,00907 | 1,069 |
| 2 | 32 | 0,01814 | 0,002471 | 0,002505 | 1,014 | 0,002249 | 0,002285 | 1,016 |
| 2 | 64 | 0,01116 | 0,0006067 | 0,0007539 | 1,243 | 0,0005735 | 0,0005974 | 1,042 |
| 3 | 4 | 0,05433 | 0,04612 | 0,06772 | 1,468 | 0,04123 | 0,04883 | 1,184 |
| 3 | 8 | 0,03345 | 0,01642 | 0,02166 | 1,319 | 0,01386 | 0,01969 | 1,421 |
| 3 | 16 | 0,02053 | 0,003478 | 0,004044 | 1,163 | 0,003066 | 0,003727 | 1,216 |
| 3 | 32 | 0,01264 | 0,0005219 | 0,000526 | 1,008 | 0,0004811 | 0,0005035 | 1,047 |
| 3 | 64 | 0,007796 | 6,806e−5 | 0,0001059 | 1,556 | 0,0001168 | 9,344e−5 | 0,8001 |

En N=32, el error de la red del estimador queda a un 1,4 % y un 0,8 % del error del Algoritmo 1 para p=2 y p=3. En N=64 la diferencia crece al 24 % y 56 %. Para p=3, la red Ritz alcanza un error menor que el iterado Ritz obtenido por L-BFGS-B en N=64. En ese nivel, la caja está activa en 11 de 16 optimizaciones cúbicas con el estimador y 10 de 16 con Ritz. Los 320 resultados corresponden a iterados alcanzados. Ninguna optimización llegó al máximo de 1000 iteraciones. [Comparación emparejada](tables/alg1_comparison.csv) · [Criterios de parada](tables/alg1_stops.csv)

## Regularidad y condicionamiento

Resultados de la red del estimador. La relación de aspecto toma únicamente los elementos físicos de la L. κ₂ se mide en el sistema de grados de libertad libres. La columna Jacobi usa escalado diagonal simétrico. Los demás métodos y todos los niveles figuran en el archivo de datos.

| p | N | γloc mediana | γloc máximo | Aspecto máximo | h mínimo realizado | κ₂ mediana | κ₂ Jacobi mediana |
|---|---|---|---|---|---|---|---|
| 2 | 16 | 19,96 | 183,8 | 283,8 | 0,0005031 | 1398 | 193,8 |
| 2 | 32 | 22,31 | 46,61 | 807,9 | 9,358e−5 | 1,631e+4 | 1264 |
| 2 | 64 | 8,429 | 12,73 | 1367 | 2,787e−5 | 1,391e+5 | 6501 |
| 3 | 16 | 188,4 | 3222 | 3210 | 7,181e−5 | 5,151e+4 | 4329 |
| 3 | 32 | 174,8 | 277 | 7539 | 1,463e−5 | 3,005e+5 | 2,497e+4 |
| 3 | 64 | 30,74 | 43,93 | 9561 | 5,544e−6 | 1,837e+6 | 6,943e+4 |

Dentro de cada semieje la cota es exp(10) ≈ 22 026. Entre semiejes normalizados por separado se aplica la cota de dos bloques demostrada en la fase C. El archivo da ambas por nivel y el máximo observado de cada número de condición. [Geometría y condicionamiento](tables/mesh_conditioning.csv)

## Coste en Xeon Platinum 8362

Cuatro núcleos por tarea. El tiempo offline y la memoria son medianas sobre las cuatro semillas. El coste online incluye una predicción de malla y la solución de Galerkin. El tiempo por instancia del Algoritmo 1 acumula las optimizaciones hasta N tras el calentamiento. Ambos tiempos online se miden sobre los mismos 16 pares. Las consultas de amortización son el cociente entre el tiempo de entrenamiento y el ahorro por consulta.

| p | Pérdida | N | Entrenamiento (h) | RSS pico mediana (GiB) | Online (ms) | Alg. 1 (s) | Consultas de amortización |
|---|---|---|---|---|---|---|---|
| 2 | η² | 32 | 0,694 | 2,93 | 47,3 | 4,88 | 516 |
| 2 | Ritz | 32 | 3,89 | 2,93 | 47,5 | 2,98 | 4776 |
| 2 | η² | 64 | 0,694 | 2,93 | 354 | 35,1 | 72 |
| 2 | Ritz | 64 | 3,89 | 2,93 | 353 | 21,4 | 667 |
| 3 | η² | 32 | 2,2 | 3,51 | 55,4 | 13,2 | 603 |
| 3 | Ritz | 32 | 4,98 | 3,47 | 56 | 4,2 | 4327 |
| 3 | η² | 64 | 2,2 | 3,51 | 396 | 79,1 | 100 |
| 3 | Ritz | 64 | 4,98 | 3,47 | 398 | 13,4 | 1379 |

Las 32 etapas de entrenamiento Ritz (2 grados × 4 semillas × 4 niveles) alcanzaron el límite de 400 épocas. Las 32 del estimador terminaron antes por validación. [Tiempos](tables/cost.csv) · [Semillas y checkpoints](tables/training.csv)

## Precisión fija

El objetivo de cada grado es la mediana de la malla uniforme en N=64. El nivel adaptado se interpola en escala log–log y el factor nominal sigue la convención tensorial del artículo. La última columna usa los grados de libertad libres efectivos, contados en la L en el primer nivel ensayado que alcanza el objetivo.

| p | Pérdida | Error objetivo | N adaptado interpolado | Factor DOF nominal | Primer N medido | DOF libres uniforme/adaptada | Factor DOF libres medido |
|---|---|---|---|---|---|---|---|
| 2 | η² | 0,009581 | 15,61 | 14,05 | 16 | 3136 / 208 | 15,08 |
| 2 | Ritz | 0,009581 | 15,26 | 14,63 | 16 | 3136 / 208 | 15,08 |
| 3 | η² | 0,006671 | 12,81 | 17,95 | 16 | 3333 / 261 | 12,77 |
| 3 | Ritz | 0,006671 | 12,29 | 19,2 | 16 | 3333 / 261 | 12,77 |

[Precisión fija](tables/fixed_accuracy.csv)

## Comprobaciones numéricas y datos de figuras

Los 5400 registros de las redes y la malla uniforme, y los 320 del Algoritmo 1 emparejado, quedan por encima de la resolución nominal de la referencia (10⁻¹¹ en error relativo cuadrático). La máxima diferencia entre el funcional de Ritz de producción y la cuadratura independiente, medida como |2J − a + 2F|/Fref, es 1,19·10⁻¹⁴. El mayor defecto |a−F|/Fref es 3,03·10⁻¹¹. El error publicado usa el campo calculado y la identidad (Fref+a−2F)/Fref. Perturbar Fref en ±10⁻¹¹ deja el factor de mejora del estimador para p=3, N=64 entre 67,43 y 67,50. [Sensibilidad y precisión](tables/energy_precision.csv)

La auditoría independiente de elementos en doble precisión cubre la rigidez, la carga y los términos de las dos pérdidas. Frente a integrales polinómicas exactas, los mayores errores relativos de rigidez y carga fueron 1,60·10⁻¹⁵ y 1,30·10⁻¹⁵. Las discrepancias máximas del estimador y de Ritz en el campo de prueba fueron 7,48·10⁻¹⁶ y 5,23·10⁻¹⁶. [Auditoría en Ice Lake](../lshape_pilot/integration_audit_icelake.csv) · [Auditoría local](../lshape_pilot/integration_audit.csv)

Las curvas de pérdida proceden de la semilla 0 en N=8. Las mallas, soluciones y el zoom se regeneraron en N=16 con los checkpoints publicados. Los campos uniforme y adaptado se han verificado por separado. El manifiesto identifica los pares de conductividades de cada panel. [Pérdidas](figures/00_losses/) · [Figuras de L-shape](figures/2D_P4/) · [Parámetros](figures/2D_P4/params_used.json)

## Sensibilidad del Algoritmo 1 al límite T en test

El barrido suplementario utiliza los mismos 16 pares reservados para la comparación por instancia. Para cada grado y cada T se optimizó el estimador por continuación en N=4, 8, 16, 32, 64 y 128. La parametrización de este diagnóstico conserva el suelo físico cero de la fase B. La comparación principal de V1 impone un suelo de 10⁻⁷ tanto a las redes como al Algoritmo 1. El barrido utiliza cuatro núcleos Platinum 8362 por tarea y comprende 768 observaciones adaptadas y 768 controles uniformes.

| p | T | Error mediano en N=32 | Error mediano en N=64 | Orden 32→64 | Error mediano en N=128 |
|---|---|---|---|---|---|
| 2 | 5 | 0,002471036 | 0,0006067203 | 2,026012 | 0,0001494186 |
| 2 | 8 | 0,002471036 | 0,0006067203 | 2,026012 | 0,0001494186 |
| 2 | 10 | 0,002471036 | 0,0006067203 | 2,026012 | 0,0001494186 |
| 2 | 15 | 0,002471036 | 0,0006067203 | 2,026012 | 0,0001494186 |
| 3 | 5 | 0,0005218795 | 6,805787e-05 | 2,938883 | 1,157887e-05 |
| 3 | 8 | 0,0005218795 | 6,805785e-05 | 2,938883 | 8,58479e-06 |
| 3 | 10 | 0,0005218795 | 6,805785e-05 | 2,938883 | 8,584689e-06 |
| 3 | 15 | 0,0005218795 | 6,805785e-05 | 2,938883 | 8,584689e-06 |

En N=32 las cuatro cotas dan la misma mediana en ambos grados. Ninguno de los 16 casos cuadráticos cambió un 2 %. Entre los cúbicos, un solo par (0,1 y 0,206914) mejoró un 3,67 % al pasar de T=5 a cualquiera de las cotas mayores.

En N=64 las medianas coinciden a cuatro cifras significativas. Un caso cuadrático y cuatro cúbicos cambian más de un 2 %. Las mayores reducciones individuales del error son del 30,8 % y del 76,0 %, respectivamente.

En N=128 la mediana cúbica mejora un 25,9 % de T=5 a T=8. Cuatro observaciones del mismo par cúbico (10 y 1,43845), una por cota, tienen error cuadrático relativo positivo por debajo de 10⁻¹¹. Al conservar su posición y sustituir el valor por el intervalo de resolución, los dos extremos de las 48 medianas coinciden. [Medianas y resolución](../lshape_alg1_Tscan/test_medians.csv) · [Órdenes](../lshape_alg1_Tscan/test_orders.csv) · [Contabilidad de Slurm](../lshape_alg1_Tscan/test_jobs.psv)
