"""F1 estricto, F1 por token, F1 por carácter y fuga por tipo de entidad en test_v2 (media de semillas).
  python analysis/per_label.py <carpeta dataset v3>  -> results/per_label_test_v2.json"""
import collections
import gzip
import json
import pickle
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import extra_metrics as X  # noqa: E402

sys.path.insert(0, str(common.ROOT / "lib"))
import postprocess  # noqa: E402
from baselines import merge_overlaps  # noqa: E402

rows = common.load_jsonl(sys.argv[1] + "/test_v2.jsonl")
L = X.PII
T = common.ROOT / "results" / "colab" / "teachers"
MODELS = {
    "DistilBERT multilingüe": [X.PROBS / f"distilbert-mbert-lr8e-05-s{s}__test_v2.pkl.gz" for s in (42, 13, 2026)],
    "Multilingual MiniLM-L12": [X.PROBS / f"minilm-l12-lr0.0001-s{s}__test_v2.pkl.gz" for s in (42, 13, 2026)],
    "mmBERT-small": [X.PROBS / f"mmbert-small-lr3e-05-{s}__test_v2.pkl.gz" for s in ("s42", "s13-r2", "s2026-r2")],
    "XLM-R base": [X.PROBS / f"xlmr-base-lr2e-05-s{s}__test_v2.pkl.gz" for s in (42, 13, 2026)],
    "Alumno destilado (A+B)": [X.PROBS / f"mmbert-small-kdAB2-lr3e-05-s{s}__test_v2.pkl.gz" for s in (42, 13, 2026)],
    "XLM-R large": [T / f"xlmr-large-lr2e-05-s{s}__test_v2.pkl.gz" for s in (42, 13, 2026)],
    "Qwen2.5-7B": [T / "qwen2.5-7b-lr0.0001-s42__test_v2.pkl.gz"],
}


def perlabel(preds, probs=None):
    c = {l: [0, 0, 0] for l in L}
    t = {l: [0, 0, 0] for l in L}
    for k, (r, P) in enumerate(zip(rows, preds)):
        g, p = X.char_labels(r["text"], r["entities"]), X.char_labels(r["text"], P)
        for i, ch in enumerate(r["text"]):
            if ch.isspace():
                continue
            gi = g[i] if g[i] in L else None
            pi = p[i] if p[i] in L else None
            if pi and pi == gi:
                c[pi][0] += 1
            else:
                if pi: c[pi][1] += 1
                if gi: c[gi][2] += 1
        if probs is not None:
            offs, pr = probs[k]
            for (a, b), kk in zip(offs, pr.astype(np.float32).argmax(-1)):
                ci = next((i for i in range(int(a), int(b)) if not r["text"][i].isspace()), None)
                if ci is None:
                    continue
                gi = g[ci] if g[ci] in L else None
                lab = common.ID2LABEL[int(kk)]
                pi = lab.split("-", 1)[1] if lab != "O" else None
                pi = pi if pi in L else None
                if pi and pi == gi:
                    t[pi][0] += 1
                else:
                    if pi: t[pi][1] += 1
                    if gi: t[gi][2] += 1
    ev = common.evaluate(rows, preds)
    return {l: {"strict": ev[f"F1_{l}"], "token": X._f1(*t[l]) if probs is not None else None,
                "char": X._f1(*c[l]), "leak": ev[f"leak_{l}"]} for l in L}


out = {}
for name, files in MODELS.items():
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        pr = pickle.load(gzip.open(f))
        for l, v in perlabel(X.SPANS_FROM_PROBS(rows, pr, common.ID2LABEL), pr).items():
            for k, x in v.items():
                acc[l][k].append(x)
    out[name] = {l: {k: st.mean(v) for k, v in d.items()} for l, d in acc.items()}
preds = [[{"start": x["start"], "end": x["end"], "label": x["label"]} for x in merge_overlaps(postprocess.rule_spans(r["text"]))] for r in rows]
out["Solo reglas"] = perlabel(preds)
out["_n"] = {l: sum(e["label"] == l for r in rows for e in r["entities"]) for l in L}
json.dump(out, open(common.ROOT / "results" / "per_label_test_v2.json", "w"), indent=1, ensure_ascii=False)
for m in out:
    if m != "_n":
        print(m, {l: (round(v["token"], 3) if v["token"] is not None else None, round(100 * v["leak"], 1)) for l, v in out[m].items()})
