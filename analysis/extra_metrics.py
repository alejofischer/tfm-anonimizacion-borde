"""Métricas adicionales a partir de las probabilidades guardadas (sin GPU):

  python analysis/extra_metrics.py <carpeta dataset v3> [runs...]   -> results/extra_metrics.jsonl

Para cada ejecución y conjunto de test calcula, sobre las mismas predicciones (clase más probable por token):
  - F1 estricto y fuga (las de siempre, para comprobar que coinciden con results.jsonl);
  - F1 por token: cada token del modelo vale lo mismo; etiqueta = tipo (B-X e I-X juntos). Depende del tokenizador;
  - F1 por carácter: cada carácter no blanco vale lo mismo; comparable entre modelos y con las reglas;
  - F1 por solapamiento: un tramo es acierto si se solapa con uno anotado de la misma etiqueta (modo «parcial»);
y lo mismo tras unir los tramos de la misma etiqueta separados solo por espacios (regla de la guía de anotación:
las piezas contiguas de un mismo dato forman un único tramo). En las tres F1 nuevas solo cuentan las 7 etiquetas de
datos personales: PUB se trata como una clase más al comparar, pero no suma aciertos ni errores propios.
"""
import collections
import gzip
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ROOT = common.ROOT
PROBS = ROOT / "results" / "colab" / "probs"
OUT = ROOT / "results" / "extra_metrics.jsonl"
PII = [l for l in common.LABELS if l not in common.NON_PII]
SPANS_FROM_PROBS = common.NS["spans_from_probs"]
GAP = re.compile(r"^[ \t]+$")       # solo espacios: ni coma, ni «y», ni salto de línea


def merge_adjacent(text, spans):
    out = []
    for s in sorted(spans, key=lambda x: x["start"]):
        if out and out[-1]["label"] == s["label"] and GAP.match(text[out[-1]["end"]:s["start"]] or "x"):
            out[-1] = {**out[-1], "end": s["end"]}
        else:
            out.append(dict(s))
    return out


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def char_labels(text, spans):
    lab = [None] * len(text)
    for s in spans:
        for i in range(s["start"], min(s["end"], len(text))):
            lab[i] = s["label"]
    return lab


def counts_char(rows, preds):
    tp = fp = fn = 0
    for r, P in zip(rows, preds):
        g, p = char_labels(r["text"], r["entities"]), char_labels(r["text"], P)
        for i, ch in enumerate(r["text"]):
            if ch.isspace():
                continue
            gi = g[i] if g[i] in PII else None
            pi = p[i] if p[i] in PII else None
            if pi and pi == gi:
                tp += 1
            else:
                fp += pi is not None
                fn += gi is not None
    return tp, fp, fn


def counts_overlap(rows, preds):
    mp = npred = mg = ngold = 0
    for r, P in zip(rows, preds):
        G = [e for e in r["entities"] if e["label"] in PII]
        Q = [e for e in P if e["label"] in PII]
        hit = lambda a, b: a["label"] == b["label"] and a["start"] < b["end"] and a["end"] > b["start"]  # noqa: E731
        npred += len(Q); ngold += len(G)
        mp += sum(any(hit(q, g) for g in G) for q in Q)
        mg += sum(any(hit(g, q) for q in Q) for g in G)
    p = mp / npred if npred else 0.0
    rr = mg / ngold if ngold else 0.0
    return 2 * p * rr / (p + rr) if p + rr else 0.0


def counts_token(rows, probs):
    # gold de cada token: tipo de la entidad anotada que contiene su primer carácter no blanco
    tp = fp = fn = 0
    for r, (offs, pr) in zip(rows, probs):
        g = char_labels(r["text"], r["entities"])
        cls = pr.astype(np.float32).argmax(-1)
        for (a, b), k in zip(offs, cls):
            c = next((i for i in range(int(a), int(b)) if not r["text"][i].isspace()), None)
            if c is None:
                continue
            gi = g[c] if g[c] in PII else None
            lab = common.ID2LABEL[int(k)]
            pi = lab.split("-", 1)[1] if lab != "O" else None
            pi = pi if pi in PII else None
            if pi and pi == gi:
                tp += 1
            else:
                fp += pi is not None
                fn += gi is not None
    return tp, fp, fn


def evaluate_all(rows, preds, probs=None):
    base = common.evaluate(rows, preds)
    rec = {"strict_F1": base["strict_F1"], "leak_rate": base["leak_rate"], "false_alarm": base["false_alarm"],
           "char_F1": _f1(*counts_char(rows, preds)), "overlap_F1": counts_overlap(rows, preds)}
    if probs is not None:
        rec["token_F1"] = _f1(*counts_token(rows, probs))
    return rec


def main():
    data_dir = sys.argv[1]
    runs = sys.argv[2:]
    tests = common.load_tests(data_dir)
    out = []
    for run in runs:
        for s, rows in tests.items():
            f = PROBS / f"{run}__{s}.pkl.gz"
            if not f.exists():
                continue
            probs = pickle.load(gzip.open(f))
            preds = SPANS_FROM_PROBS(rows, probs, common.ID2LABEL)
            merged = [merge_adjacent(r["text"], P) for r, P in zip(rows, preds)]
            a = evaluate_all(rows, preds, probs)
            b = evaluate_all(rows, merged)
            out.append({"run": run, "set": s, **a, **{f"merged_{k}": v for k, v in b.items()}})
            print(run, s, {k: round(v, 3) for k, v in out[-1].items() if isinstance(v, float)}, flush=True)
    # línea base de reglas (sin probabilidades: sin F1 por token)
    sys.path.insert(0, str(ROOT / "lib"))
    import postprocess  # noqa: E402
    from baselines import merge_overlaps  # noqa: E402
    for s, rows in tests.items():
        preds = [[{"start": x["start"], "end": x["end"], "label": x["label"]} for x in merge_overlaps(postprocess.rule_spans(r["text"]))]
                 for r in rows]
        out.append({"run": "baseline-rules", "set": s, **evaluate_all(rows, preds)})
    with open(OUT, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("ok", len(out))


if __name__ == "__main__":
    main()
