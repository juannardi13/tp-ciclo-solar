# Predicción del ciclo solar con manchas solares

Trabajo práctico de **Procesamiento de Imágenes y Señales**, UTN FRBA (2026).

## Contexto

El Sol tiene un ciclo de actividad de unos 11 años que se mide contando manchas solares. Su máximo afecta satélites, GPS, comunicaciones de radio y redes eléctricas, así que anticiparlo importa. El TP aplica las herramientas de la materia (filtrado, análisis en frecuencia, estacionariedad y modelos de series temporales) a esa señal real.

## Qué hicimos

1. **Datos:** serie mensual oficial de manchas solares de [SILSO](https://www.sidc.be/SILSO/) (Real Observatorio de Bélgica), de 1749 a 2026: 3333 meses.
2. **Exploración:** histograma, media móvil de 13 meses (el suavizado oficial), FFT y autocorrelación. Las dos últimas encuentran el ciclo en los propios datos: 11,1 y 10,7 años.
3. **Filtrado:** medias móviles de 3, 13 y 37 meses y un pasa-bajos por FFT, comparando su respuesta en frecuencia.
4. **Estacionariedad:** tests ADF y KPSS. La serie es estable en media pero no en amplitud (picos entre 69 y 270), así que se modela la raíz cuadrada.
5. **Experimento:** entrenamos con datos hasta diciembre de 2019 y pronosticamos el ciclo 25, que ya transcurrió en gran parte, para medir el error contra lo que pasó de verdad.
6. **Modelos:** ciclo promedio (baseline), AR(27), ARIMA con términos de Fourier, Prophet y una red LSTM.

## Resultados

| Modelo | Error (RMSE vs MA13) | Pico previsto |
| --- | --- | --- |
| LSTM | 11,8 | 158 (may-2024) |
| Prophet | 20,5 | 131 |
| AR(27) | 23,6 | 119 |
| Ciclo promedio | 30,7 | 179 |
| ARIMA + Fourier | 43,5 | 97 |
| **Real** | — | **159 (oct-2024)** |

La LSTM acertó la altura del pico casi exacto. El panel oficial NASA/NOAA había previsto 115. El próximo mínimo se espera entre mediados de 2030 y mediados de 2031.

## Cómo correrlo

Requiere Python 3.11 o superior.

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./run.sh
```

Tarda unos 3 minutos. Las figuras, los pronósticos y la tabla de métricas quedan en `out/`. `docs/EXPLICACION.md` explica cada paso en detalle.

## Integrantes

- Candela Bergé
- Augusto Adrian Gil Tolentino
- Juan Ignacio Nardi
- Camila Lara Vinciguerra

Datos: SILSO, Royal Observatory of Belgium, Brussels (licencia CC BY-NC 4.0).
