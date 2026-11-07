"""Funciones compartidas con el cuaderno (métricas, ventanas, decodificación), para que el análisis local use
exactamente el mismo código que Colab. Se cargan ejecutando las celdas LIB del generador del cuaderno."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebook"))
sys.path.insert(0, str(ROOT / "lib"))
import build_tfm_notebook as _B  # noqa: E402

NS = {}
exec(_B.lib_source(("LIB",)).replace("/content/work", os.environ.get("TMPDIR", "/tmp")), NS)

load_jsonl = NS["load_jsonl"]
evaluate = NS["evaluate"]
row_lang = NS["row_lang"]
row_broken = NS["row_broken"]
LABELS = NS["LABELS"]
NON_PII = NS["NON_PII"]
ID2LABEL = NS["ID2LABEL"]
decode_spans = NS["decode_spans"]
dataset_stats = NS["dataset_stats"]

TEST_FILES = {"test": "test_v1.jsonl", "test_long": "test_long_v1.jsonl", "test_v2": "test_v2.jsonl", "test_v3": "test_v3.jsonl"}


def load_tests(data_dir):
    return {k: load_jsonl(f"{data_dir}/{v}") for k, v in TEST_FILES.items()}


def evaluate_with_slices(name, set_name, rows, preds, extra=None):
    out = [{"run": name, "set": set_name, **(extra or {}), **evaluate(rows, preds)}]
    if set_name in ("test", "test_v2", "test_v3"):
        for lang in ("es", "en", "mixed"):
            idx = [i for i, r in enumerate(rows) if row_lang(r) == lang]
            if idx:
                out.append({"run": name, "set": f"{set_name}[{lang}]", **(extra or {}),
                            **evaluate([rows[i] for i in idx], [preds[i] for i in idx])})
        idx = [i for i, r in enumerate(rows) if row_broken(r)]
        if idx:
            out.append({"run": name, "set": f"{set_name}[broken]", **(extra or {}),
                        **evaluate([rows[i] for i in idx], [preds[i] for i in idx])})
    return out


def append_results(path, records):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
