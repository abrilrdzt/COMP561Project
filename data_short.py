import pandas as pd
import generate_data as S

d = S.generate(n_sweeps=1, pre_s=0.2e-3, pulse_s=0.2e-3, post_s=0.6e-3, seed=1)

df = pd.DataFrame({
    "sweep_id": d["sweep_id"],
    "time_ms": (d["time_s"] * 1000).round(2),
    "state_name": [S.STATES[i] for i in d["states"]],
    "state_index": d["states"],
    "current_pA": d["current_pA"].round(2) + 0.0,
})
print(df)
