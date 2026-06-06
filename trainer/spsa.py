#!/usr/bin/env python3
# SPSA tuner for KINg's search params (the -DTUNE build exposes them as UCI spin
# options; see src/sparams.hpp). Fishtest-style SPSA: each iteration perturbs ALL
# params simultaneously by +/-c_k, plays a small paired match theta+ vs theta- at
# a moderate TC, and nudges theta along the estimated gradient. Resumable: theta is
# checkpointed to spsa_theta.json every iteration; re-launch to continue.
#
# Validate the resulting theta at the SLOW TC vs Tucano before adopting — SPSA can
# overfit the tuning TC. Run:  python3 spsa.py [iterations]
import json, os, re, subprocess, sys, math, random

K    = "/home/nikodem/king"
CC   = "/home/nikodem/cc/cutechess-cli/cutechess-cli"
ENG  = f"{K}/v-tune/engine"
BOOK = f"{K}/book.epd"
STATE = f"{K}/spsa_theta.json"
LOG   = f"{K}/spsa.log"

# Tuning match settings.
TC          = "20+0.2"     # moderate TC: depth-sensitive but enough iterations
GAME_PAIRS  = 8            # 16 games/iteration (reversed-colour pairs)
CONCURRENCY = 14
HASH        = 64

# SPSA schedule constants (Spall's defaults).
ALPHA = 0.602
GAMMA = 0.101

# Params: name -> [init, lo, hi, c_end].  c_end = final perturbation (param units).
# r = learning-rate ratio (a_end = R * c_end^2). Small-integer params get c_end>=1.
R = 0.04   # learning-rate ratio (a_end = R*c_end^2). 0.0025 was ~16x too low —
           # params didn't move in 548 iters. 0.04 gives meaningful early steps;
           # the a_k ~ 1/k^0.602 decay still anneals to fine-tuning later.
PARAMS = {
    "LmrBase":    [75,   20, 200,  8],
    "LmrDiv":     [225,  80, 500, 16],
    "LmrHistDiv": [8192,1024,32768,1200],
    "NmpDiv":     [3,    1,   8,   1],
    "RfpMargin":  [75,   30, 160,  8],
    "FutBase":    [100,  20, 250, 15],
    "FutMult":    [80,   30, 200, 10],
    "SeeMargin":  [20,   5,   60,  4],
    "SeMargin":   [64,   20, 160,  8],
    "AspDelta":   [20,   6,   60,  4],
}

def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

def load_theta():
    if os.path.exists(STATE):
        d = json.load(open(STATE))
        return d["theta"], d["k"]
    return {p: float(v[0]) for p, v in PARAMS.items()}, 0

def save_theta(theta, k):
    json.dump({"theta": theta, "k": k}, open(STATE, "w"), indent=1)

def clamp(p, x):
    return max(PARAMS[p][1], min(PARAMS[p][2], x))

def opt_args(vals):
    # cutechess per-engine options: option.Name=Value
    return sum(([f"option.{p}={int(round(v))}"] for p, v in vals.items()), [])

def play(plus, minus):
    # paired match plus vs minus; returns plus's score in [0,1] and (w,l,d).
    cmd = [CC,
        "-engine", f"cmd={ENG}", "name=plus", "proto=uci", *opt_args(plus),
        "-engine", f"cmd={ENG}", "name=minus", "proto=uci", *opt_args(minus),
        "-each", f"tc={TC}", f"option.Hash={HASH}", "option.Threads=1",
        "-openings", f"file={BOOK}", "format=epd", "order=random",
        "-repeat", "-games", "2", "-rounds", str(GAME_PAIRS),
        "-draw", "movenumber=40", "movecount=8", "score=10",
        "-resign", "movecount=4", "score=700",
        "-concurrency", str(CONCURRENCY), "-recover"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    # cutechess prints a running "Score of ..." after EVERY game; take the LAST
    # (the final cumulative tally over all GAME_PAIRS*2 games), not the first.
    ms = re.findall(r"Score of plus vs minus:\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)", out)
    if not ms:
        return None, (0, 0, 0)
    w, l, d = map(int, ms[-1])
    n = w + l + d
    if n == 0:
        return None, (0, 0, 0)
    return (w + 0.5 * d) / n, (w, l, d)

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    A = max(1, N // 10)
    theta, k0 = load_theta()
    log(f"[spsa] start k0={k0} N={N} TC={TC} pairs={GAME_PAIRS} params={list(PARAMS)}")
    for k in range(k0 + 1, N + 1):
        plus, minus, delta = {}, {}, {}
        for p, (init, lo, hi, c_end) in PARAMS.items():
            c_k = c_end / (k ** GAMMA)
            d = 1 if random.random() < 0.5 else -1
            delta[p] = (d, c_k)
            plus[p]  = clamp(p, theta[p] + c_k * d)
            minus[p] = clamp(p, theta[p] - c_k * d)
        score, (w, l, d_) = play(plus, minus)
        if score is None:
            log(f"[spsa] k={k} MATCH FAILED, skipping")
            continue
        # ascent: theta += a_k * (2r-1) * delta / c_k   (per param)
        for p, (init, lo, hi, c_end) in PARAMS.items():
            dsign, c_k = delta[p]
            a_end = R * c_end * c_end
            a_k = a_end * ((A + N) ** ALPHA) / ((A + k) ** ALPHA)
            grad = (2 * score - 1) * dsign / c_k
            theta[p] = clamp(p, theta[p] + a_k * grad)
        save_theta(theta, k)
        if k % 5 == 0 or k < 10:
            cur = " ".join(f"{p}={theta[p]:.0f}" for p in PARAMS)
            log(f"[spsa] k={k} plus={w}-{l}-{d_} r={score:.3f} | {cur}")
    log("[spsa] done")

if __name__ == "__main__":
    main()
