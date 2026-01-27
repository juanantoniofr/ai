import numpy as np
import pandas as pd

def psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets))
    psi_value = 0

    for i in range(len(breakpoints) - 1):
        exp_pct = ((expected >= breakpoints[i]) & (expected < breakpoints[i+1])).mean()
        act_pct = ((actual >= breakpoints[i]) & (actual < breakpoints[i+1])).mean()

        if act_pct > 0 and exp_pct > 0:
            psi_value += (act_pct - exp_pct) * np.log(act_pct / exp_pct)

    return psi_value

train_preds = pd.read_csv("data/train.csv")["target"]
new_preds = pd.read_csv("data/predictions.csv")["probability"]

print("PSI:", psi(train_preds, new_preds))
