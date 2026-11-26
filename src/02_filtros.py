"""Filtrado: medias móviles de distinto largo (pasa-bajos en el tiempo) vs. pasa-bajos ideal en frecuencia."""
from common import *
x = load(); N = len(x)
mas = {k: x.rolling(k, center=True).mean() for k in (3, 13, 37)}

# Respuesta en frecuencia de cada MA: |FFT| del kernel
f = np.fft.rfftfreq(4096, d=1/12)                       # ciclos por año
H = {k: np.abs(np.fft.rfft(np.ones(k) / k, 4096)) for k in mas}
i11 = np.argmin(np.abs(f - 1/11))
for k in mas:
    corte = f[np.argmax(H[k] < 0.707)]
    print(f"MA{k}: ganancia en el ciclo de 11 años={H[k][i11]:.2f}, corte (-3 dB) en periodo={12/corte/12:.1f} años")

# Pasa-bajos en frecuencia: FFT -> anular periodos menores a 3 años -> FFT inversa
X = np.fft.rfft(x.values); fx = np.fft.rfftfreq(N, d=1/12)
X[fx > 1/3] = 0
lp = pd.Series(np.fft.irfft(X, N), index=x.index)
d = (lp - mas[13]).dropna()
print(f"pasa-bajos FFT (corte 3 años) vs MA13: correlacion={lp.corr(mas[13]):.3f}, diferencia RMS={np.sqrt((d**2).mean()):.1f}")
print(f"desvio del ruido removido: MA13={np.std(x - mas[13]):.1f}  FFT={np.std(x - lp):.1f}")

z = slice("1975", None)
fig, ax = plt.subplots(3, 1, figsize=(12, 10))
ax[0].plot(x[z], lw=0.5, color="gray", label="original")
for k, c in zip(mas, ("C0", "C1", "C3")): ax[0].plot(mas[k][z], lw=1.4, color=c, label=f"MA{k}")
ax[0].set_title("Medias móviles de 3, 13 y 37 meses: a más muestras, más suave (y más se achata el pico)"); ax[0].legend(ncol=4)
for k, c in zip(mas, ("C0", "C1", "C3")): ax[1].plot(f, H[k], color=c, label=f"MA{k}")
ax[1].axvline(1/11, ls="--", color="k"); ax[1].text(1/11, 1.02, " ciclo 11 años", fontsize=9)
ax[1].axhline(0.707, ls=":", color="gray"); ax[1].set_xlim(0, 3); ax[1].set_xlabel("frecuencia (ciclos por año)")
ax[1].set_title("Respuesta en frecuencia: la media móvil es un filtro pasa-bajos"); ax[1].legend()
ax[2].plot(x[z], lw=0.5, color="gray", label="original"); ax[2].plot(mas[13][z], color="C1", lw=1.6, label="MA13 (tiempo)")
ax[2].plot(lp[z], color="C2", lw=1.2, label="pasa-bajos FFT, corte 3 años (frecuencia)")
ax[2].set_title("Mismo filtrado por dos caminos: en el tiempo y en la frecuencia"); ax[2].legend(ncol=3)
plt.tight_layout(); plt.savefig("out/02_filtros.png", dpi=130)
