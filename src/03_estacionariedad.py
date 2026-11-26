from common import *
from statsmodels.tsa.stattools import adfuller, kpss
import warnings; warnings.filterwarnings("ignore")

x = load()
def test(s, name):
    a = adfuller(s.dropna(), autolag="AIC")[1]; k = kpss(s.dropna(), regression="c", nlags="auto")[1]
    dec = "estacionaria" if (a < .05 and k >= .05) else "NO estacionaria" if (a >= .05 and k < .05) else "indefinida"
    print(f"{name:28s} ADF p={a:.3f}  KPSS p={k:.3f}  -> {dec}")
    return s

test(x, "SN cruda")
test(np.sqrt(x), "sqrt(SN)")
test(x.diff(), "diff(SN)")
test(np.sqrt(x).diff(), "diff(sqrt(SN))")
ma13 = x.rolling(13, center=True).mean()
test(x - ma13, "SN - MA13 (residuo)")
# varianza por ciclo: amplitud de picos
picos = ma13[(ma13.shift(1) < ma13) & (ma13.shift(-1) < ma13) & (ma13 > 60)]
picos = picos[picos.index.to_series().diff().dt.days.fillna(1e9) > 365*6]
print("picos suavizados por ciclo:", [(f"{d:%Y}", round(v)) for d, v in picos.items()])
print(f"amplitud de pico: min={picos.min():.0f} max={picos.max():.0f} -> varianza NO constante")

fig, ax = plt.subplots(3, 1, figsize=(12, 8))
ax[0].plot(x, lw=0.5); ax[0].set_title("SN cruda: media estable, varianza cambia por ciclo")
ax[1].plot(np.sqrt(x), lw=0.5); ax[1].set_title("sqrt(SN): comprime los picos altos")
ax[2].plot(np.sqrt(x).diff(), lw=0.5); ax[2].set_title("diff(sqrt(SN)): estacionaria por ADF y KPSS")
plt.tight_layout(); plt.savefig("out/03_estacionariedad.png", dpi=130)
