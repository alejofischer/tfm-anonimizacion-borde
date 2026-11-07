"""Intervalos de confianza al 95 % por tipo de entidad en test_v2 para mmBERT-small (bootstrap de mensajes).

  python analysis/per_label_ci.py <carpeta dataset v3>  -> results/per_label_ci_test_v2.json

Se remuestrean los 395 mensajes con reemplazo (10.000 veces, misma semilla que bootstrap.py); en cada muestra se calcula,
para cada semilla, el F1 estricto, el F1 por token y la fuga de cada tipo, y se promedian las tres semillas.
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
L = list(X.PII)
FILES = [X.PROBS / f"mmbert-small-lr3e-05-{s}__test_v2.pkl.gz" for s in ("s42", "s13-r2", "s2026-r2")]


def per_message(probs):
    # [n, tipo, 8]: tp/fp/fn estrictos, tp/fp/fn por token, caracteres del tipo, caracteres sin ocultar
    preds = X.SPANS_FROM_PROBS(rows, probs, common.ID2LABEL)
    out = np.zeros((len(rows), len(L), 8))
    for k, (r, P) in enumerate(zip(rows, preds)):
        g = {(e["start"], e["end"], e["label"]) for e in r["entities"] if e["label"] in L}
        q = {(e["start"], e["end"], e["label"]) for e in P if e["label"] in L}
        for j, l in enumerate(L):
            gl, ql = {x for x in g if x[2] == l}, {x for x in q if x[2] == l}
            out[k, j, 0:3] = len(gl & ql), len(ql - gl), len(gl - ql)
        gch = X.char_labels(r["text"], r["entities"])
        offs, pr = probs[k]
        for (a, b), kk in zip(offs, pr.astype(np.float32).argmax(-1)):
            ci = next((i for i in range(int(a), int(b)) if not r["text"][i].isspace()), None)
            if ci is None:
                continue
            gi = gch[ci] if gch[ci] in L else None
            lab = common.ID2LABEL[int(kk)]
            pi = lab.split("-", 1)[1] if lab != "O" else None
            pi = pi if pi in L else None
            if pi and pi == gi:
                out[k, L.index(pi), 3] += 1
            else:
                if pi:
                    out[k, L.index(pi), 4] += 1
                if gi:
                    out[k, L.index(gi), 5] += 1
        cov = np.zeros(len(r["text"]) + 1, bool)
        for e in P:
            if e["label"] in L:
                cov[e["start"]:e["end"]] = True
        for e in r["entities"]:
            if e["label"] in L:
                j = L.index(e["label"])
                out[k, j, 6] += e["end"] - e["start"]
                out[k, j, 7] += (e["end"] - e["start"]) - cov[e["start"]:e["end"]].sum()
    return out


def metrics(S):  # S: [..., tipo, 8] sumado
    f = lambda tp, fp, fn: 2 * tp / np.maximum(2 * tp + fp + fn, 1)  # noqa: E731
    return np.stack([f(S[..., 0], S[..., 1], S[..., 2]), f(S[..., 3], S[..., 4], S[..., 5]),
                     S[..., 7] / np.maximum(S[..., 6], 1)], -1)


runs = [per_message(pickle.load(gzip.open(f))) for f in FILES]
n = len(rows)
rng = np.random.default_rng(2026)
idx = rng.integers(0, n, size=(10000, n))
W = np.stack([np.bincount(i, minlength=n) for i in idx]).astype(np.float64)
boot = np.mean([metrics((W @ M.reshape(n, -1)).reshape(len(W), len(L), 8)) for M in runs], 0)
full = np.mean([metrics(M.sum(0)) for M in runs], 0)
ents = {l: sum(e["label"] == l for r in rows for e in r["entities"]) for l in L}
res = {}
for j, l in enumerate(L):
    res[l] = {"entities": ents[l]}
    for m, name in enumerate(("strict_F1", "token_F1", "leak_rate")):
        res[l][name] = {"value": float(full[j, m]), "ci95": [float(np.percentile(boot[:, j, m], 2.5)),
                                                           float(np.percentile(boot[:, j, m], 97.5))]}
    print(l, ents[l], {k: (round(v["value"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in res[l].items() if k != "entities"})
json.dump(res, open(common.ROOT / "results" / "per_label_ci_test_v2.json", "w"), indent=1, ensure_ascii=False)
