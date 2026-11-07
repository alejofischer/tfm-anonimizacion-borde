"""Métricas adicionales de los profesores, sus conjuntos y los alumnos destilados (sin GPU).

  python analysis/extra_teachers.py <carpeta dataset v3>   -> results/extra_metrics_teachers.jsonl

Probabilidades de los profesores en results/colab/teachers/ (descargadas de Drive) y de los alumnos en
results/colab/probs/. Los conjuntos se evalúan con la media por carácter del cuaderno de destilación (sin F1 por token:
no hay un tokenizador común).
"""
import gzip
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import extra_metrics as X  # noqa: E402

ROOT = common.ROOT
TEACH = ROOT / "results" / "colab" / "teachers"
OUT = ROOT / "results" / "extra_metrics_teachers.jsonl"

# char_type_probs / char_spans / ensemble_eval tal cual están en el cuaderno de destilación
src = (ROOT / "notebook" / "build_distill_notebook.py").read_text(encoding="utf-8")
ns = dict(common.NS)
ns["np"] = np
ns["B_IDX"] = [2 * i + 1 for i in range(len(common.LABELS))]
ns["I_IDX"] = [2 * i + 2 for i in range(len(common.LABELS))]
for fn in ("char_type_probs", "char_spans"):
    m = re.search(rf"(def {fn}\(.*?)(?=\ndef |\nclass |\n[A-Z_]+ = |\Z)", src, re.S)
    exec(m.group(1), ns)

XL = [f"xlmr-large-lr2e-05-s{s}" for s in (42, 13, 2026)]
QW = "qwen2.5-7b-lr0.0001-s42"
STUDENTS = [f"mmbert-small-kd{t}-lr3e-05-s{s}" for t in ("A2", "AB2") for s in (42, 13, 2026)]


def load(path):
    return pickle.load(gzip.open(path))


def ens_preds(rows, prob_lists):
    preds = []
    for k, r in enumerate(rows):
        cs = [ns["char_type_probs"](r["text"], *pl[k]) for pl in prob_lists]
        preds.append(ns["char_spans"](r["text"], np.mean([x[0] for x in cs], 0), np.mean([x[1] for x in cs], 0)))
    return preds


def main():
    tests = common.load_tests(sys.argv[1])
    out = []
    for s, rows in tests.items():
        single = {n: load(TEACH / f"{n}__{s}.pkl.gz") for n in XL + [QW] if (TEACH / f"{n}__{s}.pkl.gz").exists()}
        for n, pr in single.items():
            preds = X.SPANS_FROM_PROBS(rows, pr, common.ID2LABEL)
            out.append({"run": n, "set": s, **X.evaluate_all(rows, preds, pr)})
        for name, members in (("ensemble-xlmr-large-x3", XL), ("ensemble-xlmr-large-x3+qwen7b", XL + [QW])):
            if all(m in single for m in members):
                out.append({"run": name, "set": s, **X.evaluate_all(rows, ens_preds(rows, [single[m] for m in members]))})
        for n in STUDENTS:
            f = X.PROBS / f"{n}__{s}.pkl.gz"
            if f.exists():
                pr = load(f)
                out.append({"run": n, "set": s, **X.evaluate_all(rows, X.SPANS_FROM_PROBS(rows, pr, common.ID2LABEL), pr)})
        for r in out[-12:]:
            if r["set"] == s:
                print(r["run"], s, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    with open(OUT, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
