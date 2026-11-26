import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load():
    df = pd.read_csv("data/SN_m_tot_V2.0.csv", sep=";", header=None,
                     usecols=[0, 1, 3], names=["y", "m", "sn"])
    df.index = pd.to_datetime(dict(year=df.y, month=df.m, day=1))
    return df.sn.rename("sn").asfreq("MS")

CORTE = "2019-12-01"   # fin del set de entrenamiento (inicio ciclo 25)
