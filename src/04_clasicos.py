"""Baseline (ciclo promedio), AR(p) y ARIMA+Fourier sobre sqrt(SN). Entrena hasta CORTE, pronostica 156 meses."""
from common import *
from statsmodels.tsa.ar_model import ar_select_order, AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings; warnings.filterwarnings("ignore")

x = load(); y = np.sqrt(x)
tr = y[:CORTE]; H = 156
fut = pd.date_range(tr.index[-1] + pd.offsets.MonthBegin(), periods=H, freq="MS")

# 1) Baseline: ciclo promedio alineado en el mínimo oficial de cada ciclo (SILSO)
minimos = ["1855-12","1867-03","1878-12","1890-03","1902-01","1913-07","1923-08","1933-09",
           "1944-02","1954-04","1964-10","1976-03","1986-09","1996-08","2008-12"]
perfiles = np.array([x[m:].values[:H] for m in minimos])
fc_base = pd.Series(perfiles.mean(0), index=pd.date_range("2019-12-01", periods=H, freq="MS"))
fc_base = fc_base.reindex(fut)
fc_base.to_csv("out/fc_baseline.csv", header=["sn"])
print(f"baseline: promedio de {len(minimos)} ciclos, pico={fc_base.max():.0f} en {fc_base.idxmax():%Y-%m}")

# 2) AR(p) puro sobre sqrt(SN), orden por AIC hasta 36
sel = ar_select_order(tr, maxlag=36, ic="aic", trend="c")
p = max(sel.ar_lags); ar = AutoReg(tr, lags=p, trend="c").fit()
fc_ar = ar.predict(start=len(tr), end=len(tr) + H - 1).clip(lower=0) ** 2
fc_ar.index = fut; fc_ar.to_csv("out/fc_ar.csv", header=["sn"])
print(f"AR({p}): AIC={ar.aic:.0f}, pico={fc_ar.max():.0f} en {fc_ar.idxmax():%Y-%m}")

# 3) ARIMA(p,0,q) + términos de Fourier (periodo 132 meses, K armónicos): regresión armónica con errores ARMA
def fourier(idx, K, P=132):
    t = np.arange(len(idx)) if isinstance(idx, int) else idx
    return np.column_stack([f(2 * np.pi * k * t / P) for k in range(1, K + 1) for f in (np.sin, np.cos)])
t_tr = np.arange(len(tr)); t_fut = np.arange(len(tr), len(tr) + H)
best = None
for K in (2, 3):
    for order in [(1,0,0),(2,0,0),(3,0,0),(2,0,1),(3,0,1)]:
        m = SARIMAX(tr, exog=fourier(t_tr, K), order=order, trend="c").fit(disp=False)
        if best is None or m.aic < best[0]: best = (m.aic, K, order, m)
aic, K, order, m = best
fc_f = m.get_forecast(H, exog=fourier(t_fut, K)).predicted_mean.clip(lower=0) ** 2
fc_f.index = fut; fc_f.to_csv("out/fc_arima_fourier.csv", header=["sn"])
print(f"ARIMA{order}+Fourier K={K}: AIC={aic:.0f}, pico={fc_f.max():.0f} en {fc_f.idxmax():%Y-%m}")
print(m.summary().tables[1])
