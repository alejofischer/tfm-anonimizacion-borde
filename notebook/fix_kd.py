# Corrección del 5-nov (se ejecuta DENTRO del entorno de tfm_destilacion.ipynb, que ya tiene las probabilidades de los
# profesores en /content/teachers):  exec(open("/content/drive/MyDrive/tfm_pii/tfm/fix_kd.py").read())
#
# Fallo: mmBERT a veces separa el espacio de dentro de un valor ("ES74 0081 …", "634 118 902") en un token propio, y la
# versión anterior de soft_for_offsets le daba siempre la etiqueta O. El alumno aprendía a partir IBAN, teléfonos y
# direcciones en trozos (alumnos kdA: F1 0,71 y 0,62 en test_v2). Corregido: un token que solo es espacio es
# continuación (I-X) si el profesor dice que lo que viene después continúa la entidad. Comprobado en local: trasladar
# las probabilidades de XLM-R base a los tokens de mmBERT da F1 0,890 (el propio XLM-R: 0,889).
# Los alumnos nuevos se guardan con las etiquetas "A2" y "<mejor>2"; los "kdA" antiguos se quedan como registro del fallo.
#
# Segundo arreglo: la celda 6.4 (Qwen) falló porque peft rechaza la versión de torchao que trae Colab (0.10; pide >0.16).
# torchao es opcional (cuantización) y aquí no se usa: se desinstala y se le dice a peft que no está.
import subprocess, importlib, sys as _sys
subprocess.run([_sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
importlib.invalidate_caches()
for _name, _mod in list(_sys.modules.items()):
    if _name.startswith("peft") and hasattr(_mod, "is_torchao_available"):
        _mod.is_torchao_available = lambda: False
print("torchao desactivado para peft")


def soft_for_offsets(text, offs, tables):
    out = np.zeros((len(offs), len(ID2LABEL)), dtype=np.float32)
    for t, o in enumerate(offs):
        if o is None or o[0] == o[1]:
            continue
        a, b = o
        c = next((k for k in range(a, b) if not text[k].isspace()), None)
        acc = np.zeros(len(ID2LABEL), dtype=np.float32)
        for cp, start in tables:
            if c is None:
                k = next((j for j in range(b, len(text)) if not text[j].isspace()), None)
                q = np.zeros(len(ID2LABEL), dtype=np.float32)
                if k is None:
                    q[0] = 1.0
                else:
                    nxt = cp[k]
                    q[I_IDX] = nxt[I_IDX] if start[k] else nxt[I_IDX] + nxt[B_IDX]
                    q[0] = 1.0 - q[I_IDX].sum()
            else:
                q = cp[c].copy()
                if not start[c]:
                    q[I_IDX] += q[B_IDX]
                    q[B_IDX] = 0
            acc += q
        out[t] = acc / len(tables)
    return out


_KD_CACHE.clear()
print("profesores A disponibles:", [n for n in TEACHERS_A if teacher_ready(n)])

# 6.3 bis — alumno destilado del conjunto de 3 XLM-R large, con la corrección
for seed in SEEDS:
    train_student(TEACHERS_A, "A2", seed)

# 6.4 — profesor B (Qwen2.5-7B)
TEACHER_B_RUN = train_teacher(TEACHER_B, CANDIDATES[TEACHER_B]["lr_grid"][0], SEEDS[0], predictor_cls=LLMPredictor)

# 6.5 — elegir el conjunto de profesores en validación
sets_to_try = {"A": TEACHERS_A, "B": [TEACHER_B_RUN], "A+B": TEACHERS_A + [TEACHER_B_RUN]}
val_f1 = {tag: ensemble_eval(val_rows, [load_teacher(n, "val") for n in runs])["strict_F1"]
          for tag, runs in sets_to_try.items()}
print("strict F1 en validación por conjunto de profesores:", val_f1)
best_tag = max(val_f1, key=val_f1.get)
log_ensemble("ensemble-qwen7b", [TEACHER_B_RUN])
log_ensemble("ensemble-xlmr-large-x3+qwen7b", TEACHERS_A + [TEACHER_B_RUN])

# 6.6 — segundo alumno solo si el mejor conjunto no es A
if best_tag != "A":
    for seed in SEEDS:
        train_student(sets_to_try[best_tag], best_tag.replace("+", "") + "2", seed)

res = load_results()
display(show([r for r in res if r["set"] == "test_v2" and (r.get("role") in ("teacher", "student", "teacher-ensemble")
              or r["run"].startswith(f"{STUDENT_KEY}-lr{STUDENT_LR:g}"))]))
from google.colab import drive as _d
_d.flush_and_unmount()
