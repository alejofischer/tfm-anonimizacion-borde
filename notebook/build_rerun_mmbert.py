"""Genera notebook/tfm_rerun_mmbert.ipynb: repetición de las semillas 13 y 2026 de mmBERT-small.

  python notebook/build_rerun_mmbert.py

Por qué existe: en la ejecución del 4-5 de noviembre, mmBERT-small con las semillas 13 y 2026 dio F1 ≈ 0,20 en los
tests, pero durante el entrenamiento esos mismos modelos tenían 0,978 de macro F1 por token en validación. El fallo
está en la evaluación posterior (el modelo recargado predecía «O» casi siempre), no en el entrenamiento.

Este cuaderno reutiliza las celdas 0-5 del cuaderno principal (mismas funciones, mismo protocolo) y añade:
1. mmBERT se carga con `reference_compile=False` (sin torch.compile interno de ModernBERT, primer sospechoso: su
   caché de funciones compiladas sobrevive de un modelo a otro dentro del mismo proceso).
2. Tras entrenar, se evalúa la validación con el modelo recargado y se compara con la validación del propio
   entrenamiento. Si no cuadra, se repite la evaluación en fp32 sin autocast y se avisa.
3. Los resultados se guardan con el sufijo «-r2» (no se borra nada de la ejecución original; el análisis local usa
   la versión r2 para estas semillas y explica el cambio en la memoria).
4. Control: se reevalúa con el mismo código el mejor modelo de la semilla 42 que ya está en Drive.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_tfm_notebook as B  # noqa: E402

# Celdas 0-5 del cuaderno principal: instalación, configuración, datos, tokenización, métricas y entrenamiento.
# Se corta justo antes de la sección 5a (búsqueda de learning rate).
cells = []
for c in B.CELLS:
    if c[0] == "markdown" and c[1].startswith("### 5a."):
        break
    cells.append(c)


def md(s):
    cells.append(("markdown", s.strip("\n")))


def code(s, tag=None):
    cells.append(("code", s.strip("\n"), tag))


cells.insert(0, ("markdown", """# TFM — Repetición de las semillas 13 y 2026 de mmBERT-small

Las celdas 0-5 son idénticas a las del cuaderno principal (`tfm_comparativa.ipynb`). La sección 6 es nueva: vuelve a
entrenar mmBERT-small (learning rate 3e-5, el elegido en validación) con las semillas 13 y 2026, comprueba que la
evaluación tras recargar el modelo coincide con la validación del entrenamiento, y guarda los resultados con el
sufijo `-r2`. Duración: ~40 minutos en una L4. **Entorno → Ejecutar todo.**"""))

md("""## 6. Repetición con comprobaciones

En versiones antiguas de `transformers`, `ModernBERT` (la arquitectura de mmBERT) compilaba partes del modelo con
`torch.compile` (`reference_compile`); si la versión instalada aún tiene esa opción, se desactiva. Además se añade
una segunda ruta de evaluación en fp32 pura con atención "eager", y se compara con la validación del entrenamiento.""")
code(r"""
from transformers import AutoConfig

KEY, LR = "mmbert-small", 3e-5
SUFFIX = "-r2"

# 1) Entrenamiento sin compilación interna: se envuelve from_pretrained para que mmBERT reciba reference_compile=False.
if getattr(AutoModelForTokenClassification.from_pretrained, "__name__", "") == "_from_pretrained_no_compile":
    del AutoModelForTokenClassification.from_pretrained   # la celda ya se ejecutó antes: se quita el envoltorio viejo
_orig_from_pretrained = AutoModelForTokenClassification.from_pretrained


def _from_pretrained_no_compile(name, *args, **kwargs):
    # Solo si la versión de transformers aún tiene esa opción (la 5.x la quitó y ya no compila nada por su cuenta).
    cfg = AutoConfig.from_pretrained(name)
    if getattr(cfg, "model_type", "") == "modernbert" and hasattr(cfg, "reference_compile"):
        kwargs.setdefault("reference_compile", False)
    return _orig_from_pretrained(name, *args, **kwargs)


AutoModelForTokenClassification.from_pretrained = _from_pretrained_no_compile


class Fp32Predictor(TorchPredictor):
    # Igual que TorchPredictor pero en fp32 puro (sin autocast bf16) y con atención "eager": la ruta más simple.
    def __init__(self, model_dir):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = _from_pretrained_no_compile(model_dir, attn_implementation="eager").eval().cuda()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def logits(self, ids, mask):
        with self.torch.no_grad():
            out = self.model(input_ids=self.torch.tensor(ids).cuda(), attention_mask=self.torch.tensor(mask).cuda())
        return out.logits.float().cpu().numpy()


def train_val_f1(run_name):
    # Mejor macro F1 por token en validación según el propio Trainer (lo que se vio durante el entrenamiento).
    hist = json.load(open(f"{RUNS_DIR}/logs/{run_name}.json"))
    return max(h["eval_tok_macro_f1"] for h in hist if "eval_tok_macro_f1" in h)


def quick_val(predictor):
    # strict F1 en validación con el modelo recargado (la métrica final del TFM, no la del Trainer).
    return evaluate(val_rows, predict_rows(val_rows, predictor))["strict_F1"]
""")

md("""### 6a. Control: el mejor modelo de la semilla 42 (ya en Drive)

Se reevalúa en validación con las dos rutas. En la ejecución original dio strict F1 = 0,966: si aquí sale lo
mismo, la ruta de evaluación es correcta.""")
code(r"""
ctrl_dir = f"{RUNS_DIR}/models/mmbert-small-best"
if os.path.exists(ctrl_dir):
    p = TorchPredictor(ctrl_dir); print("control s42, bf16:", round(quick_val(p), 4)); del p
    p = Fp32Predictor(ctrl_dir);  print("control s42, fp32:", round(quick_val(p), 4)); del p
    gc.collect(); torch.cuda.empty_cache()
else:
    print("No está el modelo de control en Drive; se omite.")
""")

md("""### 6b. Semillas 13 y 2026

Para cada semilla: entrenar → evaluar validación con el modelo recargado en bf16 y en fp32 → si alguna ruta cuadra
con la validación del entrenamiento (strict F1 ≥ 0,90), evaluar todos los conjuntos con esa ruta y guardar como
`mmbert-small-lr3e-05-s<semilla>-r2`. Si ninguna cuadra, se guarda igualmente y se avisa (sería un resultado real
de inestabilidad, que habría que contar en la memoria).""")
code(r"""
for seed in (13, 2026):
    name = f"{KEY}-lr{LR:g}-s{seed}{SUFFIX}"
    if run_done(name):
        print("ya hecho:", name); continue
    final, info = train_model(KEY, LR, seed, name)
    tv = train_val_f1(name)
    p16 = TorchPredictor(final); v16 = quick_val(p16)
    p32 = Fp32Predictor(final);  v32 = quick_val(p32)
    print(f"[{name}] validación del Trainer (macro F1 por token) = {tv:.4f} | recargado: bf16 = {v16:.4f}, fp32 = {v32:.4f}")
    use, fmt = (p16, "torch-bf16") if v16 >= 0.90 else (p32, "torch-fp32")
    if max(v16, v32) < 0.90:
        print("⚠️ Ninguna ruta reproduce la validación del entrenamiento: se guarda igualmente, revisar.")
    evaluate_all(use, name, {**info, "fmt": fmt, "rerun": True, "val_bf16": round(v16, 4), "val_fp32": round(v32, 4),
                             "trainer_val_tok_macro_f1": round(tv, 4)})
    del p16, p32, use; gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree(Path(final).parent, ignore_errors=True)

display(show([r for r in load_results() if r["run"].endswith(SUFFIX) and r["set"] in ("val", "test_v2")]))
""")


def build():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for c in cells:
        if c[0] == "markdown":
            nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": c[1].splitlines(keepends=True)})
        else:
            nb["cells"].append({"cell_type": "code", "metadata": {"tags": [c[2]] if c[2] else []},
                                "execution_count": None, "outputs": [], "source": c[1].splitlines(keepends=True)})
    out = HERE / "tfm_rerun_mmbert.ipynb"
    json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
    print("escrito", out, "-", len(nb["cells"]), "celdas")


if __name__ == "__main__":
    build()
