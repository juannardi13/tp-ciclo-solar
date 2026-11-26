"""Figuras para las slides: una por slide, tema oscuro, misma paleta que docs/slides.html. Requiere haber corrido 04–06."""
from common import *
from statsmodels.tsa.stattools import acf
import sys
# Tema: por defecto el de docs/slides.html; con "canva" usa la paleta negra/roja de la plantilla de Canva.
CANVA = len(sys.argv) > 1 and sys.argv[1] == "canva"
OUT = "out/canva" if CANVA else "out/slides"
if CANVA:
    BG, PANEL, FG, MUTED, ACC = "#000000", "#000000", "#ffffff", "#8a8a8a", "#ff8a1f"
    COL = ["#ff5a36", "#ff8a1f", "#ff2a2a", "#9ad1ff", "#c792ea"]; GRID, EDGE = "#1c1c1c", "#3a3a3a"
else:
    BG, PANEL, FG, MUTED, ACC = "#0b1020", "#141b33", "#e8ecf5", "#9aa5c0", "#f5a623"
    COL = ["#5ab0ff", "#f5a623", "#ff6b6b", "#7ee787", "#c792ea"]; GRID, EDGE = "#222b4a", "#2a3458"
__import__("os").makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG, "text.color": FG,
    "axes.labelcolor": FG, "axes.edgecolor": EDGE, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": FG,
    "font.size": 13, "axes.titlesize": 15, "axes.spines.top": False, "axes.spines.right": False,
    "legend.facecolor": PANEL, "legend.edgecolor": EDGE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
def save(name): plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150); plt.close()

x = load(); N = len(x); ma13 = x.rolling(13, center=True).mean()

# Datos: serie completa
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(x, lw=0.5, color=MUTED, label="mensual"); ax.plot(ma13, lw=1.6, color=ACC, label="media móvil 13 meses")
ax.set_title("Manchas solares por mes, 1749–2026"); ax.legend(); save("serie")

# Tiempo: histograma
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(x, bins=40, color=COL[0]); ax.axvline(x.mean(), color=ACC, ls="--"); ax.text(x.mean() + 4, ax.get_ylim()[1] * .9, f"media {x.mean():.0f}", color=ACC)
ax.set_title("Histograma: muchos meses tranquilos, pocos muy activos"); ax.set_xlabel("manchas por mes"); ax.set_ylabel("meses"); save("histograma")

# Frecuencia: FFT + ACF
X = np.abs(np.fft.rfft(x - x.mean())); f = np.fft.rfftfreq(N, d=1/12); pk = np.argmax(X[1:]) + 1; r = acf(x, nlags=400)
fig, ax = plt.subplots(2, 1, figsize=(9, 8))
ax[0].plot(1/f[1:], X[1:]/X[1:].max(), color=COL[0]); ax[0].set_xlim(2, 40); ax[0].axvline(1/f[pk], ls="--", color=ACC)
ax[0].text(1/f[pk] + .6, .92, f"{1/f[pk]:.1f} años", color=ACC, fontsize=15); ax[0].set_title("Espectro (FFT)"); ax[0].set_xlabel("periodo (años)")
ax[1].plot(np.arange(401)/12, r, color=COL[0]); ax[1].axhline(0, color=MUTED, lw=.6); l = np.argmax(r[60:200]) + 60
ax[1].plot(l/12, r[l], "o", color=ACC); ax[1].text(l/12 + .6, r[l], f"{l/12:.1f} años", color=ACC, fontsize=15)
ax[1].set_title("Autocorrelación"); ax[1].set_xlabel("desplazamiento (años)"); save("frecuencia")

# Filtrado: MAs, respuesta, pasa-bajos FFT
mas = {k: x.rolling(k, center=True).mean() for k in (3, 13, 37)}; z = slice("1975", None)
ff = np.fft.rfftfreq(4096, d=1/12); Xl = np.fft.rfft(x.values); Xl[np.fft.rfftfreq(N, d=1/12) > 1/3] = 0
lp = pd.Series(np.fft.irfft(Xl, N), index=x.index)
fig, ax = plt.subplots(3, 1, figsize=(9, 10))
ax[0].plot(x[z], lw=.5, color=MUTED)
for (k, m), c in zip(mas.items(), (COL[0], ACC, COL[2])): ax[0].plot(m[z], lw=1.5, color=c, label=f"MA{k}")
ax[0].set_title("Medias móviles de 3, 13 y 37 meses"); ax[0].legend(ncol=3)
for k, c in zip(mas, (COL[0], ACC, COL[2])): ax[1].plot(ff, np.abs(np.fft.rfft(np.ones(k)/k, 4096)), color=c, label=f"MA{k}")
ax[1].axvline(1/11, ls="--", color=FG); ax[1].text(1/11 + .03, .93, "ciclo de 11 años", fontsize=11); ax[1].set_xlim(0, 3)
ax[1].set_title("Respuesta en frecuencia: son filtros pasa-bajos"); ax[1].set_xlabel("ciclos por año"); ax[1].legend()
ax[2].plot(x[z], lw=.5, color=MUTED); ax[2].plot(mas[13][z], color=ACC, lw=1.8, label="MA13 (tiempo)"); ax[2].plot(lp[z], color=COL[3], lw=1.3, label="pasa-bajos FFT (frecuencia)")
ax[2].set_title("Mismo filtrado por dos caminos"); ax[2].legend(ncol=2); save("filtros")

# Estacionariedad: cruda vs raíz
fig, ax = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
ax[0].plot(x, lw=.5, color=COL[0]); ax[0].set_title("Serie original: picos entre 69 y 270")
ax[1].plot(np.sqrt(x), lw=.5, color=ACC); ax[1].set_title("Raíz cuadrada: amplitudes más parejas"); save("estacionariedad")

# Resultados
names = {"baseline": "Ciclo promedio", "ar": "AR(27)", "arima_fourier": "ARIMA + Fourier", "prophet": "Prophet", "lstm": "LSTM"}
fcs = {n: pd.read_csv(f"out/fc_{k}.csv", index_col=0, parse_dates=True).sn for k, n in names.items()}
def comp(desde, hasta, titulo, name, size):
    fig, ax = plt.subplots(figsize=size)
    ax.plot(x[desde:], lw=.5, color=MUTED); ax.plot(ma13[desde:], lw=3, color=FG, label="Real (suavizado)")
    for (n, fc), c in zip(fcs.items(), (COL[0], COL[3], COL[4], "#ffd166", COL[2])):
        ax.plot(fc, lw=3 if n == "LSTM" else 1.4, color=c, label=n, alpha=1 if n == "LSTM" else .85)
    ax.axvline(pd.Timestamp(CORTE), ls="--", color=MUTED); ax.text(pd.Timestamp(CORTE), 245, " fin del entrenamiento", color=MUTED, fontsize=11)
    ax.axvspan(x.index[-1], pd.Timestamp(hasta), color=ACC, alpha=.06); ax.text(x.index[-1], 225, " futuro", color=ACC, fontsize=11)
    ax.set_xlim(pd.Timestamp(desde), pd.Timestamp(hasta)); ax.set_ylim(0, 265); ax.set_title(titulo); ax.legend(loc="upper left", ncol=2); save(name)
comp("2008-01-01", "2033-01-01", "Ciclo 25: lo que predijo cada modelo vs. lo que pasó", "comparacion", (13, 6.5))
comp("2022-01-01", "2033-01-01", "La bajada del ciclo 25 y el próximo mínimo", "pronostico", (9, 6.5))
# Diseño del experimento: qué vio el modelo y qué no
fig, ax = plt.subplots(figsize=(8.28, 6.5)); c = pd.Timestamp(CORTE); fin = x.index[-1]
ax.plot(ma13["1990":CORTE], color=MUTED, lw=2.2); ax.plot(ma13[CORTE:], color=FG, lw=2.8)
ax.axvspan(pd.Timestamp("1990-01-01"), c, color=MUTED, alpha=.12); ax.axvspan(c, fin, color=COL[2], alpha=.22); ax.axvspan(fin, pd.Timestamp("2033-01-01"), color=ACC, alpha=.22)
for t0, t1, lab, col in [("1990-01-01", CORTE, "ENTRENAMIENTO\n1749 a dic 2019", MUTED), (CORTE, fin, "TEST\n81 meses", COL[2]), (fin, "2033-01-01", "FUTURO", ACC)]:
    m = pd.Timestamp(t0) + (pd.Timestamp(t1) - pd.Timestamp(t0)) / 2; ax.text(m, 250, lab, ha="center", va="top", color=col, fontsize=13, fontweight="bold")
ax.set_xlim(pd.Timestamp("1990-01-01"), pd.Timestamp("2033-01-01")); ax.set_ylim(0, 265); ax.set_title("Qué ve el modelo y contra qué se lo compara"); save("experimento")
# Esquema de la LSTM: ventana de 132 meses -> mes siguiente
fig, ax = plt.subplots(figsize=(9, 6)); seg = ma13["1996":"2019"]; w0, w1 = pd.Timestamp("2008-12-01"), pd.Timestamp("2019-12-01")
ax.plot(seg, color=MUTED, lw=2); ax.plot(seg[w0:w1], color=FG, lw=3); ax.axvspan(w0, w1, color=ACC, alpha=.18)
ax.text(w0 + (w1 - w0) / 2, 235, "VENTANA: 132 MESES", ha="center", color=ACC, fontsize=14, fontweight="bold")
nxt = w1 + pd.offsets.MonthBegin(8); ax.plot([nxt], [14], "o", color=COL[2], ms=13)
ax.annotate("predice el\nmes siguiente", xy=(nxt, 14), xytext=(pd.Timestamp("2021-06-01"), 110), color=COL[2], fontsize=13, ha="center", arrowprops=dict(arrowstyle="->", color=COL[2], lw=2))
ax.text(pd.Timestamp("2010-06-01"), 185, "corre la ventana un mes y repite 156 veces", color=FG, fontsize=12, ha="left")
ax.set_xlim(pd.Timestamp("1996-01-01"), pd.Timestamp("2024-06-01")); ax.set_ylim(0, 260); ax.set_title("Cómo pronostica la LSTM"); save("lstm")
print(f"figuras en {OUT}/:", ", ".join(sorted(p.name for p in __import__("pathlib").Path(OUT).glob("*.png"))))
