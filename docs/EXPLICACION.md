# Qué hicimos, explicado para entenderlo (no para presentarlo)

Pensalo como un pipeline de siete pasos. Cada paso es un script en `src/`. Acá va qué hace cada uno, por qué, y qué pregunta responde. Si entendés esto, podés defender cualquier slide.

## 0. El dato: una lista de 3333 números

`data/SN_m_tot_V2.0.csv` tiene una fila por mes desde 1749. La columna que usamos es el "número de manchas solares" del mes. Nada más. Es una serie temporal: valores ordenados en el tiempo a intervalo fijo (un mes).

`src/common.py` la carga en un objeto de pandas con índice de fechas. Todo lo demás parte de ahí.

## 1. Mirar la serie en el tiempo (`01_exploratorio.py`, primera mitad)

Antes de modelar, la miramos. Tres cosas:

- **Media, desvío, histograma.** El histograma mostró que hay muchísimos meses con pocas manchas y pocos con muchas. Eso se llama distribución asimétrica. Importa porque varios modelos asumen que los datos se parecen a una campana de Gauss, y estos no.
- **Media móvil de 13 meses (MA13).** Para cada mes, promediás ese mes con los 6 anteriores y los 6 siguientes. Borra el ruido mes a mes y deja la forma lenta, el ciclo. Es el filtro más simple que existe y es el que usa el observatorio belga oficialmente. En el código es una línea: `x.rolling(13, center=True).mean()`.
- **Dónde está el pico de cada ciclo.** Con la MA13 suavizada, el máximo de cada joroba es el pico del ciclo.

## 2. Mirar la serie en frecuencia (`01_exploratorio.py`, segunda mitad)

Acá está la parte "de señales" de la materia. La idea de Fourier: cualquier serie se puede descomponer en una suma de senoides de distintas frecuencias. La FFT te dice cuánto pesa cada frecuencia.

Hicimos `np.fft.rfft(x - x.mean())`. Restar la media saca la componente de frecuencia cero, que si no tapa todo. El resultado es un espectro: en el eje x la frecuencia (o su inverso, el periodo) y en el eje y cuánta energía hay. El pico más alto cayó en un periodo de 11,1 años.

La **autocorrelación** es otra forma de ver lo mismo: comparás la serie consigo misma desplazada k meses y medís cuánto se parecen. Si hay un ciclo de 11 años, al desplazar ~132 meses la serie se vuelve a parecer a sí misma. Nos dio un máximo en el lag 128, o sea 10,7 años. Dos métodos independientes, mismo resultado.

**Por qué importa para la presentación:** demuestra que el "ciclo de 11 años" no es algo que asumimos sino algo que sale de los datos.

## 2b. Filtrar (`02_filtros.py`)

Una media móvil es un filtro pasa-bajos: deja pasar lo lento (el ciclo) y atenúa lo rápido (el ruido mes a mes). Cuanto más larga la ventana, más filtra, pero si te pasás empezás a aplastar el propio ciclo: la de 37 meses ya le saca un 12 % al ciclo de 11 años, la de 13 solo un 2 %. Eso se ve en la "respuesta en frecuencia", que es la FFT de los pesos del filtro.

También filtramos por el otro camino, el de la clase 4: FFT, poner en cero todas las frecuencias con periodo menor a 3 años, FFT inversa. Da casi lo mismo que la MA13 (correlación 0,995). Ojo con un detalle que se ve al final del gráfico: el filtro por FFT hace un rulo raro en 2026, porque Fourier asume que la serie se repite y trata de empalmar el final con el principio.

## 3. ¿Es estacionaria? (`03_estacionariedad.py`)

Estacionaria significa que la media y la varianza no cambian con el tiempo. Importa porque los modelos clásicos (AR, ARIMA) lo necesitan.

Hay dos tests estadísticos que vimos en clase, ADF y KPSS. Se usan juntos porque tienen hipótesis opuestas:

- ADF: hipótesis nula "no es estacionaria". Si p < 0,05, rechazás y decís que sí lo es.
- KPSS: hipótesis nula "sí es estacionaria". Si p ≥ 0,05, no rechazás y decís que sí lo es.

Los dos dijeron estacionaria. Bien en media. Pero mirando la altura de cada ciclo vimos que varía de 69 a 270: la varianza sí cambia. Para emparejarla aplicamos raíz cuadrada a toda la serie. Es un truco estándar: comprime los valores grandes más que los chicos. Todos los modelos trabajan sobre `sqrt(SN)` y al final elevamos el pronóstico al cuadrado para volver a manchas.

## 4. El diseño del experimento (la idea más importante)

Si pronosticás el futuro, no podés saber si acertaste. Entonces hicimos un **backtest**: fingimos que estamos en diciembre de 2019, entrenamos los modelos solo con datos hasta ahí, y les pedimos que pronostiquen 156 meses. Los primeros 81 meses de ese pronóstico (2020 a 2026) ya pasaron, así que comparamos contra la realidad y medimos el error.

El error lo medimos con RMSE (raíz del error cuadrático medio, castiga más los errores grandes) y MAE (error absoluto medio). Lo calculamos contra la serie cruda y contra la MA13; la segunda es la que importa porque mide si acertaste la forma del ciclo y no el ruido de cada mes.

La constante `CORTE = "2019-12-01"` en `common.py` es esa fecha.

## 5. Los modelos clásicos (`04_clasicos.py`)

- **Ciclo promedio.** No es un modelo de la materia, es el baseline de sentido común: tomás los 15 ciclos anteriores, los alineás en su mínimo y los promediás. Si un modelo sofisticado no le gana a esto, no sirve.
- **AR(p).** Autorregresivo: el valor de hoy es una combinación lineal de los p valores anteriores más ruido. `y[t] = a1·y[t-1] + a2·y[t-2] + ... + ap·y[t-p]`. El p lo elige el criterio de Akaike (AIC), que premia ajustar bien y castiga usar muchos parámetros. Salió p = 27.
- **ARIMA + Fourier.** ARIMA es AR más una parte de media móvil del ruido (MA) y opcionalmente diferenciación (I). Le agregamos como variables externas dos senoides de periodo 132 meses, para que el ARIMA solo tenga que modelar lo que queda después de sacar el ciclo. Fue el peor porque fija el periodo en 11 años exactos y el ciclo real varía.

## 6. Prophet (`06_prophet.py`)

Herramienta de Facebook. Modela la serie como tendencia + estacionalidades + feriados. Le dijimos: tendencia plana, nada de estacionalidad anual ni semanal, y una estacionalidad custom de 11 años. Internamente la estacionalidad son senoides de Fourier, parecido al modelo anterior pero con otro ajuste. Anduvo mejor que los clásicos.

## 7. LSTM (`05_lstm.py`)

Una red neuronal recurrente. La idea sin matemática:

1. Cortás la serie en ventanas de 132 meses (un ciclo). Cada ventana es un ejemplo de entrenamiento y la respuesta correcta es el mes 133.
2. La red ve miles de esas ventanas y aprende "dado cómo vinieron los últimos 11 años, qué viene el mes que sigue".
3. Para pronosticar: le das los últimos 132 meses reales, predice uno, lo agregás al final de la ventana, sacás el primero, y repetís 156 veces. Se llama pronóstico recursivo.

Detalles: escalamos los datos entre 0 y 1 (las redes lo necesitan), 32 neuronas LSTM, 40 pasadas por los datos (épocas), semilla fija para que siempre dé lo mismo.

Por qué le ganó a los demás: los modelos lineales convierten todo en senoides, y el ciclo solar no es una senoide, sube rápido y baja lento. La LSTM no asume ninguna forma, aprende la que hay.

## 8. Comparar (`07_comparar.py`)

Carga los cinco pronósticos, calcula los errores contra lo real y dibuja todo junto. La tabla de `out/metricas.csv` es el resultado del TP.

## Lo que tenés que poder responder

- **¿Por qué raíz cuadrada y no logaritmo?** Hay meses con cero manchas y log(0) no existe. La raíz sí.
- **¿Por qué 132 meses de ventana?** Un ciclo completo. Si la ventana es más corta, la red no ve dónde está en el ciclo.
- **¿Por qué cortar en 2019?** Es el mínimo oficial que separa el ciclo 24 del 25. Así el test es un ciclo entero.
- **¿La LSTM no está haciendo trampa?** No: nunca vio un dato posterior a 2019. El pico de 2024 lo adivinó.
- **¿Entonces le ganamos a la NASA?** En este ciclo, sí. Con un solo ciclo de test no se puede generalizar. Es una de las limitaciones que decimos.
