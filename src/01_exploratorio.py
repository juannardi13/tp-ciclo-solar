from common import *
from statsmodels.tsa.stattools import acf

x = load(); N = len(x)
ma13 = x.rolling(13, center=True).mean()
print(f"N={N}  {x.index[0]:%Y-%m} -> {x.index[-1]:%Y-%m}")
print(f"media={x.mean():.1f} mediana={x.median():.1f} DE={x.std():.1f} min={x.min()} max={x.max()} ({x.idxmax():%Y-%m})")
print(f"asimetria={x.skew():.2f}  % meses con SN<20: {(x<20).mean()*100:.0f}%")

# FFT
X = np.abs(np.fft.rfft(x - x.mean())); f = np.fft.rfftfreq(N, d=1/12)
top = np.argsort(X[1:])[::-1][:5] + 1
print("picos FFT (periodo años, amplitud rel):", [(round(1/f[i], 2), round(X[i]/X[top[0]], 2)) for i in top])

# ACF
r = acf(x, nlags=400)
lag1 = np.argmax(r[60:200]) + 60
print(f"ACF: primer maximo secundario en lag={lag1} meses = {lag1/12:.1f} años, r={r[lag1]:.2f}")

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0,0].hist(x, bins=40, color="C0"); ax[0,0].set_title("Histograma SN mensual (asimétrica, no gaussiana)")
ax[0,1].plot(x, lw=0.4, color="gray"); ax[0,1].plot(ma13, color="C1"); ax[0,1].set_title("Serie + MA13")
ax[1,0].plot(1/f[1:], X[1:]/X[1:].max()); ax[1,0].set_xlim(2, 40); ax[1,0].set_xlabel("periodo (años)")
ax[1,0].axvline(1/f[top[0]], ls="--", color="C3"); ax[1,0].set_title(f"Espectro: pico {1/f[top[0]]:.1f} años")
ax[1,1].plot(np.arange(401)/12, r); ax[1,1].axhline(0, color="k", lw=0.5); ax[1,1].set_xlabel("lag (años)")
ax[1,1].set_title(f"Autocorrelación: máximo secundario a {lag1/12:.1f} años")
plt.tight_layout(); plt.savefig("out/01_exploratorio.png", dpi=130)
