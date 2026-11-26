"""LSTM univariada sobre sqrt(SN) escalada: ventana de 132 meses -> mes siguiente, pronóstico recursivo 156 meses."""
from common import *
import torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

x = load(); y = np.sqrt(x); tr = y[:CORTE]; H = 156; W = 132
scale = tr.max(); s = (tr / scale).values.astype(np.float32)
X = np.stack([s[i:i+W] for i in range(len(s) - W)]); Y = s[W:]
X = torch.tensor(X)[:, :, None]; Y = torch.tensor(Y)[:, None]

class Net(nn.Module):
    def __init__(s): super().__init__(); s.l = nn.LSTM(1, 32, batch_first=True); s.o = nn.Linear(32, 1)
    def forward(s, x): return s.o(s.l(x)[0][:, -1])
net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3); loss = nn.MSELoss()
for ep in range(40):
    perm = torch.randperm(len(X)); tot = 0
    for i in range(0, len(X), 64):
        b = perm[i:i+64]; opt.zero_grad(); l = loss(net(X[b]), Y[b]); l.backward(); opt.step(); tot += l.item() * len(b)
    if ep % 10 == 9: print(f"epoch {ep+1}: mse={tot/len(X):.5f}")

net.eval(); win = list(s[-W:]); out = []
with torch.no_grad():
    for _ in range(H):
        p = net(torch.tensor(win[-W:], dtype=torch.float32)[None, :, None]).item()
        p = max(p, 0); out.append(p); win.append(p)
fut = pd.date_range(tr.index[-1] + pd.offsets.MonthBegin(), periods=H, freq="MS")
fc = pd.Series((np.array(out) * scale) ** 2, index=fut); fc.to_csv("out/fc_lstm.csv", header=["sn"])
print(f"LSTM: pico={fc.max():.0f} en {fc.idxmax():%Y-%m}")
