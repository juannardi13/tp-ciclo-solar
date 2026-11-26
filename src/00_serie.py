"""Lámina de presentación: serie completa + MA13, últimos 5 ciclos, espectro."""
from common import *
x = load(); N = len(x); ma13 = x.rolling(13, center=True).mean()
X = np.abs(np.fft.rfft(x - x.mean())); f = np.fft.rfftfreq(N, d=1/12); peak = np.argmax(X[1:]) + 1
fig, ax = plt.subplots(3, 1, figsize=(11, 10))
ax[0].plot(x, lw=0.4, color="gray", label="SN mensual"); ax[0].plot(ma13, lw=1.2, color="C1", label="MA13")
ax[0].set_title(f"Número de manchas solares, SILSO 1749–2026 (N={N})"); ax[0].legend()
m = x["1975":]; ax[1].plot(m, lw=0.6, color="gray"); ax[1].plot(ma13["1975":], lw=1.5, color="C1")
for c, d in [(21, "1976-03"), (22, "1986-09"), (23, "1996-08"), (24, "2008-12"), (25, "2019-12")]:
    ax[1].axvline(pd.Timestamp(d), ls="--", lw=0.8, color="k"); ax[1].text(pd.Timestamp(d), 250, f" c{c}")
ax[1].set_title("Últimos 5 ciclos (inicio oficial de cada ciclo en línea punteada)")
ax[2].plot(1/f[1:], X[1:]/X[1:].max(), lw=1); ax[2].set_xlim(2, 60); ax[2].axvline(1/f[peak], ls="--", color="C3")
ax[2].set_title(f"Espectro |FFT| vs. periodo: pico en {1/f[peak]:.1f} años"); ax[2].set_xlabel("periodo (años)")
plt.tight_layout(); plt.savefig("out/00_serie_silso.png", dpi=130)
