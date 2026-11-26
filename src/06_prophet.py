"""Prophet con estacionalidad custom de 11 años, tendencia plana, sobre sqrt(SN)."""
from common import *
from prophet import Prophet
import logging; logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

x = load(); y = np.sqrt(x); tr = y[:CORTE]; H = 156
df = pd.DataFrame({"ds": tr.index, "y": tr.values})
m = Prophet(growth="flat", yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
m.add_seasonality("solar", period=11 * 365.25, fourier_order=5)
m.fit(df)
fut = m.make_future_dataframe(H, freq="MS", include_history=False)
p = m.predict(fut)
fc = pd.Series(p.yhat.clip(lower=0).values ** 2, index=pd.DatetimeIndex(p.ds)); fc.to_csv("out/fc_prophet.csv", header=["sn"])
print(f"Prophet: pico={fc.max():.0f} en {fc.idxmax():%Y-%m}")
