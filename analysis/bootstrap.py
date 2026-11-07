"""Prueba de significación por remuestreo bootstrap pareado (Efron y Tibshirani, 1993; Dror et al., 2018) en test_v2.

  python analysis/bootstrap.py <carpeta dataset v3>  -> results/bootstrap_test_v2.json

Se remuestrean los 395 mensajes con reemplazo (10.000 veces); en cada muestra se calcula la métrica de cada semilla de
los dos modelos y se compara la media de sus semillas. p = proporción de muestras en que el modelo A no supera a B
(prueba unilateral); IC 95 % de la diferencia por percentiles.
"""
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import extra_metrics as X  # noqa: E402

rows = common.load_jsonl(sys.argv[1] + "/test_v2.jsonl")
L = X.PII
P = X.PROBS
RUNS = {
    "mmBERT-small": [P / f"mmbert-small-lr3e-05-{s}__test_v2.pkl.gz" for s in ("s42", "s13-r2", "s2026-r2")],
    "XLM-R base": [P / f"xlmr-base-lr2e-05-s{s}__test_v2.pkl.gz" for s in (42, 13, 2026)],
    "DistilBERT multilingüe": [P / f"distilbert-mbert-lr8e-05-s{s}__test_v2.pkl.gz" for s in (42, 13, 2026)],
    "Multilingual MiniLM-L12": [P / f"minilm-l12-lr0.0001-s{s}__test_v2.pkl.gz" for s in (42, 13, 2026)],
    "Alumno destilado (A+B)": [P / f"mmbert-small-kdAB2-lr3e-05-s{s}__test_v2.pkl.gz" for s in (42, 13, 2026)],
}


def per_message(probs):
    # por mensaje: (tp, fp, fn) estrictos, (tp, fp, fn) por token, caracteres de datos y caracteres sin ocultar
    preds = X.SPANS_FROM_PROBS(rows, probs, common.ID2LABEL)
    out = np.zeros((len(rows), 8))
    for k, (r, Pk) in enumerate(zip(rows, preds)):
        g = {(e["start"], e["end"], e["label"]) for e in r["entities"] if e["label"] in L}
        q = {(e["start"], e["end"], e["label"]) for e in Pk if e["label"] in L}
        out[k, 0:3] = len(g & q), len(q - g), len(g - q)
        out[k, 3:6] = X.counts_token([r], [probs[k]])
        cov = np.zeros(len(r["text"]) + 1, bool)
        for e in Pk:
            if e["label"] in L:
                cov[e["start"]:e["end"]] = True
        for e in r["entities"]:
            if e["label"] in L:
                out[k, 6] += e["end"] - e["start"]
                out[k, 7] += (e["end"] - e["start"]) - cov[e["start"]:e["end"]].sum()
    return out


def metrics(M):          # M: [n, 8] sumado por muestra
    s = M.sum(0)
    f = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)  # noqa: E731
    return np.array([f(*s[0:3]), f(*s[3:6]), s[7] / max(s[6], 1)])


data = {m: [per_message(pickle.load(gzip.open(f))) for f in fs] for m, fs in RUNS.items()}
rng = np.random.default_rng(2026)
B = 10000
idx = rng.integers(0, len(rows), size=(B, len(rows)))
boot = {m: np.mean([np.array([metrics(M[i]) for i in idx]) for M in runs], 0) for m, runs in data.items()}
full = {m: np.mean([metrics(M) for M in runs], 0) for m, runs in data.items()}
res = {}
for b in [m for m in RUNS if m != "mmBERT-small"]:
    d = boot["mmBERT-small"] - boot[b]           # diferencia A − B en cada muestra
    obs = full["mmBERT-small"] - full[b]
    r = {}
    for j, name in enumerate(("strict_F1", "token_F1", "leak_rate")):
        better = d[:, j] < 0 if name == "leak_rate" else d[:, j] > 0
        r[name] = {"diff": float(obs[j]), "ci95": [float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))],
                   "p": float(1 - better.mean())}
    res[f"mmBERT-small vs {b}"] = r
    print(b, {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci95"]], round(v["p"], 4)) for k, v in r.items()}, flush=True)
json.dump(res, open(common.ROOT / "results" / "bootstrap_test_v2.json", "w"), indent=1, ensure_ascii=False)
