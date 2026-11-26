from common import *
x = load(); ma13 = x.rolling(13, center=True).mean()
test = x["2020-01":]; test13 = ma13["2020-01":].dropna()
names = {"baseline": "Ciclo promedio", "ar": "AR(27)", "arima_fourier": "ARIMA(3,0,1)+Fourier", "lstm": "LSTM", "prophet": "Prophet"}
fcs = {n: pd.read_csv(f"out/fc_{k}.csv", index_col=0, parse_dates=True).sn for k, n in names.items()}
rows = []
for n, fc in fcs.items():
    e = fc.reindex(test.index) - test; e13 = fc.reindex(test13.index) - test13
    rows.append(dict(modelo=n, RMSE=np.sqrt((e**2).mean()), MAE=e.abs().mean(), RMSE_MA13=np.sqrt((e13**2).mean()),
                     pico=fc.max(), fecha_pico=f"{fc.idxmax():%Y-%m}", min_c26=f"{fc['2027':].idxmin():%Y-%m}"))
real = dict(modelo="REAL (hasta 2026-09)", RMSE=0, MAE=0, RMSE_MA13=0, pico=ma13["2020":].max(), fecha_pico=f"{ma13["2020":].idxmax():%Y-%m}", min_c26="?")
tab = pd.DataFrame(rows + [real]).round(1); print(tab.to_string(index=False)); tab.to_csv("out/metricas.csv", index=False)

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(x["2008":], lw=0.5, color="gray", label="SN mensual real")
ax.plot(ma13["2008":], lw=2, color="k", label="MA13 real")
for n, fc in fcs.items(): ax.plot(fc, lw=1.3, label=n)
ax.axvline(pd.Timestamp(CORTE), ls="--", color="k"); ax.text(pd.Timestamp(CORTE), 230, " corte entrenamiento", fontsize=9)
ax.axvspan(pd.Timestamp("2020-01-01"), x.index[-1], alpha=0.08, color="green"); ax.text(pd.Timestamp("2021-06-01"), 250, "backtest", color="green")
ax.set_xlim(pd.Timestamp("2008-01-01"), pd.Timestamp("2033-01-01")); ax.set_ylim(0, 270)
ax.set_title("Ciclo 25: pronósticos entrenados hasta dic-2019 vs. realidad, y extrapolación a 2032"); ax.legend(loc="upper left", ncol=2)
plt.tight_layout(); plt.savefig("out/07_comparacion.png", dpi=130)
