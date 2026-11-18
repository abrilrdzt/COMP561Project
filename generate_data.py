"""
Synthetic single-channel AMPA receptor data from the Robert & Howe 12-state model

generate() returns one value per time sample of a patch-clamp-style recording:
    states      hidden state index 0-11 (names in STATES)  <- ground truth for the HMM
    current_pA  noise-free current reading
    sweep_id    which sweep the sample belongs to
    time_s      time within the sweep
No noise is added. Needs numpy and scipy.  Run:  python ampa_synthetic.py
"""
import bisect
import numpy as np
from scipy.linalg import expm

STATES = ["R0", "R1", "R2", "R3", "R4", "O2", "O3", "O4", "D1", "D2", "D3", "D4"]

# Table 1, GluR1
ALPHA, BETA = 3100.0, 8000.0          # closing, opening (1/s)
K1, KM1 = 2e7, 9000.0                 # glutamate binding (1/(M s)), unbinding (1/s)
DELTA1, GAMMA1 = 1800.0, 7.6          # entry into / recovery from desensitization (1/s)
CONDUCTANCE_PS = {"O2": 9.0, "O3": 15.0, "O4": 21.0}   # the other nine states carry no current


def build_Q(glu):
    """12x12 rate matrix (1/s) at glutamate concentration glu (molar), as labelled in Fig. 3a."""
    k = K1 * glu
    edges = {
        # closed chain (R = resting, number = glutamates bound)
        ("R0", "R1"): 4 * k, ("R1", "R0"): 1 * KM1,
        ("R1", "R2"): 3 * k, ("R2", "R1"): 2 * KM1,
        ("R2", "R3"): 2 * k, ("R3", "R2"): 3 * KM1,
        ("R3", "R4"): 1 * k, ("R4", "R3"): 4 * KM1,
        # opening / closing
        ("R2", "O2"): 2 * BETA, ("O2", "R2"): ALPHA,
        ("R3", "O3"): 3 * BETA, ("O3", "R3"): ALPHA,
        ("R4", "O4"): 4 * BETA, ("O4", "R4"): ALPHA,
        # entry into / recovery from desensitization
        ("R1", "D1"): 1 * DELTA1, ("D1", "R1"): GAMMA1,
        ("R2", "D2"): 2 * DELTA1, ("D2", "R2"): GAMMA1,
        ("R3", "D3"): 3 * DELTA1, ("D3", "R3"): GAMMA1,
        ("R4", "D4"): 4 * DELTA1, ("D4", "R4"): GAMMA1,
        # glutamate binding / unbinding while desensitized
        ("D1", "D2"): 3 * k, ("D2", "D1"): 1 * KM1,
        ("D2", "D3"): 2 * k, ("D3", "D2"): 2 * KM1,
        ("D3", "D4"): 1 * k, ("D4", "D3"): 3 * KM1,
    }
    Q = np.zeros((12, 12))
    for (a, b), rate in edges.items():
        Q[STATES.index(a), STATES.index(b)] = rate
    return Q - np.diag(Q.sum(axis=1))      # rows sum to zero


def transition_probs(glu, t):
    """Exact probability of going from each state to each state in time t: expm(Q t)."""
    P = np.clip(expm(build_Q(glu) * t), 0.0, None)
    return P / P.sum(axis=1, keepdims=True)


def generate(n_sweeps=400, pre_s=5e-3, pulse_s=1e-3, post_s=24e-3, gap_s=1.0, glu_M=1e-2,
             dt=20e-6, v_hold_mV=-60.0, seed=0):
    """
    Each sweep: pre_s without glutamate, a pulse of glu_M for pulse_s, then post_s without
    glutamate, sampled every dt. Between sweeps the channel recovers for gap_s without being
    recorded. The first sweep starts in R0 (all unbound). Reversal potential is taken as 0 mV.
    """
    rng = np.random.default_rng(seed)
    n_pre, n_pulse, n_post = (int(round(x / dt)) for x in (pre_s, pulse_s, post_s))
    pulse_on = np.r_[np.zeros(n_pre), np.ones(n_pulse), np.zeros(n_post)].astype(int)
    n = pulse_on.size

    cdfs = []                                     # [no glutamate, glutamate], as nested lists
    for glu in (0.0, glu_M):
        c = np.cumsum(transition_probs(glu, dt), axis=1)
        c[:, -1] = 1.0
        cdfs.append(c.tolist())
    P_gap = transition_probs(0.0, gap_s)

    states = np.empty(n_sweeps * n, dtype=np.int8)
    s = 0
    for k in range(n_sweeps):
        u = rng.random(n)
        sweep = states[k * n:(k + 1) * n]
        sweep[0] = s
        for t in range(1, n):
            s = bisect.bisect_right(cdfs[pulse_on[t]][s], u[t])   # next state
            sweep[t] = s
        s = int(rng.choice(12, p=P_gap[s]))                       # unrecorded recovery gap

    g_pS = np.zeros(12)
    for name, g in CONDUCTANCE_PS.items():
        g_pS[STATES.index(name)] = g
    return {
        "states": states,
        "current_pA": g_pS[states] * v_hold_mV * 1e-3,            # pS * mV = 1e-3 pA (inward < 0)
        "sweep_id": np.repeat(np.arange(n_sweeps), n),
        "time_s": np.tile(np.arange(n) * dt, n_sweeps),
    }


if __name__ == "__main__":
    d = generate()
    st = d["states"]
    opened = np.isin(st, [STATES.index(x) for x in CONDUCTANCE_PS])
    print(f"{st.size} samples in {d['sweep_id'].max() + 1} sweeps")
    print("current levels (pA):", sorted(set(np.round(d["current_pA"], 3).tolist())))
    print("sweeps with at least one opening:", len(set(d["sweep_id"][opened])))
    print("fraction of samples per state:",
          {name: round(float((st == i).mean()), 4) for i, name in enumerate(STATES)})