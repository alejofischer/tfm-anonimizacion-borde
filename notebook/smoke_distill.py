"""Prueba local (CPU) del cuaderno de destilación con modelos diminutos y aleatorios.

  python notebook/smoke_distill.py <carpeta del dataset v3 descomprimido>     (necesita peft)

Comprueba de punta a punta: profesor A (XLM-R diminuto, 2 learning rates + semillas, lote acumulado), profesor B
(Qwen2 diminuto + LoRA + atención bidireccional), probabilidades de los profesores, conjunto por caracteres,
selección en validación y alumnos destilados. Los números no significan nada (modelos aleatorios, 1 época).
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import build_tfm_notebook as B
import build_distill_notebook as D

DATA = sys.argv[1]
work, runs, teach = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
for sub in ("", "/models", "/logs", "/probs", "/onnx"):
    os.makedirs(runs + sub, exist_ok=True)


def patch(src):
    return (src.replace("/content/work", work).replace("bf16=True", "bf16=False, use_cpu=True")
               .replace("dataloader_num_workers=2", "dataloader_num_workers=0").replace(".eval().cuda()", ".eval()")
               .replace("torch.tensor(ids).cuda()", "torch.tensor(ids)").replace("torch.tensor(mask).cuda()", "torch.tensor(mask)")
               .replace('self.torch.autocast("cuda", dtype=self.torch.bfloat16)', 'self.torch.autocast("cpu", enabled=False)')
               .replace("dtype=torch.bfloat16", "dtype=torch.float32")
               .replace("torch.cuda.empty_cache()", "None").replace("display(", "print("))


ns = {}
exec(patch(B.lib_source(("LIB", "TRAIN_LIB"))), ns)
ns["RUNS_DIR"] = runs
exec("TEACH_DIR, KD_T, KD_ALPHA, STUDENT_KEY, STUDENT_LR, DUMP_TEACHER_PROBS_TO_DRIVE = "
     f"{teach!r}, 2.0, 0.5, 'mmbert-small', 5e-4, False", ns)
exec(patch(D.lib_source(("DISTILL_LIB",))), ns)
load = ns["load_jsonl"]
ns.update(train_rows=load(f"{DATA}/train.jsonl")[:96], val_rows=load(f"{DATA}/val.jsonl")[:40],
          test_rows=load(f"{DATA}/test_v1.jsonl")[:20], long_rows=load(f"{DATA}/test_long_v1.jsonl")[:2],
          test2_rows=load(f"{DATA}/test_v2.jsonl")[:20], test3_rows=load(f"{DATA}/test_v3.jsonl")[:20])
ns["EPOCHS"] = 1
ns["SEEDS"] = [42, 13]                                 # 2 semillas para ir más rápido

# modelos diminutos: XLM-R (profesor A y alumno) y Qwen2 (profesor B), con el tokenizador de XLM-R
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, XLMRobertaConfig, XLMRobertaForTokenClassification
tok_x = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
tiny_x = f"{work}/tiny-xlmr"
XLMRobertaForTokenClassification(XLMRobertaConfig(vocab_size=250002, hidden_size=32, num_hidden_layers=2,
                                                  num_attention_heads=2, intermediate_size=64,
                                                  max_position_embeddings=514, num_labels=17)).save_pretrained(tiny_x)
tok_x.save_pretrained(tiny_x)
tiny_q = f"{work}/tiny-qwen"
Qwen2ForCausalLM(Qwen2Config(vocab_size=250002, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)).save_pretrained(tiny_q)
tok_x.save_pretrained(tiny_q)
C = ns["CANDIDATES"]
C["xlmr-large"].update(model=tiny_x, tokenizer=tiny_x, lr_grid=[5e-4, 1e-3], per_device=8)
C["qwen2.5-7b"].update(model=tiny_q, tokenizer=tiny_q, lr_grid=[1e-3], epochs=1, per_device=8, model_kwargs={}, ckpt_dir=f"{work}/ckpt", save_steps=2)
C["mmbert-small"].update(model=tiny_x, tokenizer=tiny_x)

# las celdas sin etiqueta de la sección 6 (6.2 a 6.6), tal cual
body = [c[1] for c in D.cells if c[0] == "code" and c[2] is None and ("TEACHER" in c[1] or "best_tag" in c[1])
        and "flush_and_unmount" not in c[1] and "pip" not in c[1]]
for src in body:
    print("=" * 30, src.splitlines()[0])
    exec(patch(src), ns)

res = ns["load_results"]()
kinds = sorted({(r.get("role"), r["run"]) for r in res if r["set"] == "test_v2"})
for k in kinds:
    print(k)
assert any(r.get("role") == "student" for r in res), "no hay alumnos"
assert any(r.get("role") == "teacher-ensemble" for r in res), "no hay conjuntos"
s = ns["SOFT"][0] if ns["SOFT"] else None
print("probabilidades objetivo de una ventana:", None if s is None else (s.shape, np.round(s.astype(np.float32).sum(1)[:6], 3)))
print("SMOKE DESTILACIÓN OK")

# --- continuar desde un punto de control (como si Colab hubiera cortado el entorno a mitad del LLM)
import shutil as _sh
_orig_rmtree = ns["shutil"].rmtree
ns["shutil"].rmtree = lambda p, ignore_errors=False: None if "ckpt" in str(p) else _orig_rmtree(p, ignore_errors=ignore_errors)
ns["train_model"]("qwen2.5-7b", 1e-3, 7, "resume-test")         # 1.ª vez: deja puntos de control
ns["shutil"].rmtree = _orig_rmtree
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns["train_model"]("qwen2.5-7b", 1e-3, 7, "resume-test")     # 2.ª vez: debe continuar desde el último
assert "se continúa desde" in buf.getvalue(), buf.getvalue()[-500:]
print("CONTINUAR DESDE PUNTO DE CONTROL OK")
