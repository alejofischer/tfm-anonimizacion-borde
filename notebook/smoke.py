"""Prueba local (CPU) de las celdas del cuaderno antes de gastar GPU.

  python notebook/smoke.py <carpeta del dataset v3 descomprimido>

1. Oráculo: un predictor que devuelve las etiquetas correctas debe dar F1 = 1, fuga = 0 y falsas alarmas = 0
   con los tokenizadores de los cuatro candidatos (comprueba ventanas, alineación, decodificación y métricas).
2. Entrenamiento + exportación con un XLM-R diminuto y aleatorio: train_and_eval, export FP32/FP16/INT8,
   eficiencia (latencia y RAM) y volcado de probabilidades.
"""
import json
import os
import sys
import tempfile

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import build_tfm_notebook as B

DATA = sys.argv[1]
work, runs = tempfile.mkdtemp(), tempfile.mkdtemp()
for sub in ("", "/models", "/logs", "/probs", "/onnx"):
    os.makedirs(runs + sub, exist_ok=True)
src = B.lib_source(("LIB", "TRAIN_LIB", "EXPORT_LIB"))
src = (src.replace("/content/work", work).replace("bf16=True", "bf16=False, use_cpu=True")
          .replace("dataloader_num_workers=2", "dataloader_num_workers=0").replace(".eval().cuda()", ".eval()")
          .replace("torch.tensor(ids).cuda()", "torch.tensor(ids)").replace("torch.tensor(mask).cuda()", "torch.tensor(mask)")
          .replace('self.torch.autocast("cuda", dtype=self.torch.bfloat16)', 'self.torch.autocast("cpu", enabled=False)'))
ns = {}
exec(src, ns)
ns["RUNS_DIR"] = runs
load = ns["load_jsonl"]
ns.update(train_rows=load(f"{DATA}/train.jsonl")[:300], val_rows=load(f"{DATA}/val.jsonl")[:60],
          test_rows=load(f"{DATA}/test_v1.jsonl")[:80], long_rows=load(f"{DATA}/test_long_v1.jsonl")[:4],
          test2_rows=load(f"{DATA}/test_v2.jsonl")[:80], test3_rows=load(f"{DATA}/test_v3.jsonl")[:60])
print(ns["dataset_stats"]("test_v2", load(f"{DATA}/test_v2.jsonl")))


class Oracle:
    def __init__(self, tok_name, rows):
        self.tok = AutoTokenizer.from_pretrained(tok_name)
        self.id2label = ns["ID2LABEL"]
        self.table = {}
        for r in rows:
            ents = sorted(r["entities"], key=lambda e: e["start"])
            for ids, mask, offs in ns["windows_for_text"](self.tok, r["text"]):
                self.table[tuple(ids)] = ns["align_labels"](offs, ents)

    def logits(self, ids, mask):
        out = np.full((len(ids), len(ids[0]), len(self.id2label)), -10.0, dtype=np.float32)
        for b, seq in enumerate(ids):
            labels = self.table[tuple(t for t, m in zip(seq, mask[b]) if m)]
            for t, lab in enumerate(labels):
                out[b, t, max(lab, 0)] = 10.0
            out[b, len(labels):, 0] = 10.0
        return out


ok = True
for tok_name in {c["tokenizer"] for c in ns["CANDIDATES"].values()}:
    for s in ("test", "test_long", "test_v2", "test_v3"):
        rows = ns["rows_for"](s)
        res = ns["evaluate"](rows, ns["predict_rows"](rows, Oracle(tok_name, rows)))
        line = f"{tok_name.split('/')[1]:38s} oráculo {s:9s} F1={res['strict_F1']:.3f} macroF1={res['macro_F1']:.3f} fuga={res['leak_rate']:.4f} FA={res['false_alarm']:.3f}"
        print(line)
        # el oráculo no llega a 1 cuando un valor empieza o acaba dentro de un token (techo de cada tokenizador)
        ok &= res["strict_F1"] > 0.95 and res["leak_rate"] < 0.01 and res["macro_F1"] <= 1
print("ORÁCULO OK" if ok else "ORÁCULO FALLA")

# --- entrenamiento + exportación con un modelo diminuto
from transformers import XLMRobertaConfig, XLMRobertaForTokenClassification
tiny = f"{work}/tiny-xlmr"
cfg = XLMRobertaConfig(vocab_size=250002, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=64, max_position_embeddings=514, num_labels=len(ns["ID2LABEL"]))
XLMRobertaForTokenClassification(cfg).save_pretrained(tiny)
AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base").save_pretrained(tiny)
ns["CANDIDATES"]["tiny"] = dict(model=tiny, tokenizer=tiny, lr_grid=[5e-4])
ns["EPOCHS"] = 1
name, final = ns["train_and_eval"]("tiny", 5e-4, 42)
print("entrenado:", name, sorted(os.listdir(final)))
assert ns["run_done"](name)
print("log_history:", len(json.load(open(f"{runs}/logs/{name}.json"))), "entradas; probs:", sorted(os.listdir(f"{runs}/probs"))[:3])
exp = f"{work}/export"
os.makedirs(exp, exist_ok=True)
fp32 = ns["export_fp32"](final, exp)
paths = {"fp32": fp32, "fp16": ns["export_fp16"](fp32, exp), "int8-matmul": ns["export_int8"](fp32, exp, "matmul"),
         "int8-all": ns["export_int8"](fp32, exp, "all")}
ns["LAT_CHARS"] = (200, 1000)
for fmt, p in paths.items():
    eff = ns["efficiency"](p, exp, ns["test2_rows"])
    recs = ns["evaluate_all"](ns["OnnxPredictor"](p, exp, threads=2), f"{name}:onnx-{fmt}", {"fmt": f"onnx-{fmt}", **eff},
                              sets=("test_v2",), dump_probs=False)
    print(fmt, eff, "F1", round(recs[0]["strict_F1"], 3))
r = [x for x in ns["load_results"]() if x["run"] == name and x["set"] == "test_v2"][0]
print("registro de entrenamiento:", {k: r[k] for k in ("model", "lr", "seed", "params_m", "emb_params_m", "vocab", "best_epoch", "train_min")})
print("SMOKE OK")
