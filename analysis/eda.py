"""Análisis exploratorio del corpus (sin GPU):  python analysis/eda.py <carpeta dataset v3>  -> results/eda.json"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

D = sys.argv[1]
OUT = common.ROOT / "results" / "eda.json"
sets = {"train": common.load_jsonl(f"{D}/train.jsonl"), "val": common.load_jsonl(f"{D}/val.jsonl"),
        **common.load_tests(D)}
res = {"length_by_label": {}, "broken_share": {}, "sources": {}, "tokens_per_word": {}}

# longitud de las entidades por tipo (caracteres y palabras), en train y en test_v2
for s in ("train", "test_v2"):
    by = collections.defaultdict(list)
    for r in sets[s]:
        for e in r["entities"]:
            by[e["label"]].append((len(e["text"]), len(e["text"].split())))
    res["length_by_label"][s] = {l: {"n": len(v), "chars_median": st.median(x[0] for x in v),
                                     "chars_mean": round(st.mean(x[0] for x in v), 1),
                                     "words_mean": round(st.mean(x[1] for x in v), 2)} for l, v in by.items()}

# proporción de mensajes con valores rotos y sin datos personales
for s, rows in sets.items():
    res["broken_share"][s] = {"broken": round(sum(common.row_broken(r) for r in rows) / len(rows), 3),
                              "no_pii": round(sum(not any(e["label"] not in common.NON_PII for e in r["entities"]) for r in rows) / len(rows), 3),
                              "mixed": round(sum(common.row_lang(r) == "mixed" for r in rows) / len(rows), 3)}

# composición del entrenamiento por fuente: mensajes, idioma y entidades por tipo
src = collections.defaultdict(lambda: {"rows": 0, "es": 0, "en": 0, "mixed": 0, "labels": collections.Counter()})
for r in sets["train"]:
    s = src[r.get("source", "?")]
    s["rows"] += 1
    lang = common.row_lang(r)
    if lang in ("es", "en", "mixed"):
        s[lang] += 1
    s["labels"].update(e["label"] for e in r["entities"])
res["sources"] = {k: {**{kk: vv for kk, vv in v.items() if kk != "labels"}, "labels": dict(v["labels"])} for k, v in src.items()}

# fragmentación de cada tokenizador: tokens por palabra en test_v2, por idioma
from transformers import AutoTokenizer  # noqa: E402
TOKS = {"DistilBERT multilingüe": "distilbert/distilbert-base-multilingual-cased",
        "MiniLM-L12 / XLM-R": "FacebookAI/xlm-roberta-base", "mmBERT-small": "jhu-clsp/mmBERT-small",
        "Qwen2.5": "Qwen/Qwen2.5-7B"}
for name, path in TOKS.items():
    tok = AutoTokenizer.from_pretrained(path)
    out = {}
    for lang in ("es", "en", "mixed"):
        rows = [r for r in sets["test_v2"] if common.row_lang(r) == lang]
        words = sum(len(r["text"].split()) for r in rows)
        toks = sum(len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows)
        ent_w = sum(len(e["text"].split()) for r in rows for e in r["entities"])
        ent_t = sum(len(tok(e["text"], add_special_tokens=False)["input_ids"]) for r in rows for e in r["entities"])
        out[lang] = {"tokens_per_word": round(toks / words, 2), "entity_tokens_per_word": round(ent_t / ent_w, 2)}
    res["tokens_per_word"][name] = out
    print(name, out, flush=True)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "sources"}, ensure_ascii=False, indent=1)[:4000])
