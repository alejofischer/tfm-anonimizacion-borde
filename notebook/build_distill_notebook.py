"""Genera notebook/tfm_destilacion.ipynb: profesores grandes y destilación hacia mmBERT-small.

  python notebook/build_distill_notebook.py

Experimento (capítulo 5 de la memoria):
  1. Profesor A: XLM-R large (550 M), mismo protocolo que el resto de candidatos (búsqueda corta de learning rate en
     validación y 3 semillas). Los tres modelos forman un conjunto (ensemble).
  2. Alumno: mmBERT-small (el modelo del escenario de aplicación) entrenado con las etiquetas del corpus Y las probabilidades del
     conjunto de profesores (destilación, Hinton et al., 2015). Mismo learning rate y épocas que su versión normal,
     así la diferencia solo se debe a la destilación. 3 semillas.
  3. Profesor B: un LLM de 7 B (Qwen2.5-7B) convertido en etiquetador de tokens: capa de clasificación + LoRA + atención
     bidireccional (sin máscara causal), como en Li et al. (2023) y Dukić y Šnajder (2024).
  4. Se elige en VALIDACIÓN qué conjunto de profesores es mejor (A, B o A+B); si B ayuda, segundo alumno destilado.

Reutiliza las celdas 0-5 del cuaderno principal (datos, métricas y train_model). Las celdas nuevas llevan la etiqueta
DISTILL_LIB para probarlas en local con notebook/smoke_distill.py.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_tfm_notebook as B  # noqa: E402

cells = []
for c in B.CELLS:
    if c[0] == "markdown" and c[1].startswith("### 5a."):
        break
    cells.append(c)


def md(s):
    cells.append(("markdown", s.strip("\n")))


def code(s, tag=None):
    cells.append(("code", s.strip("\n"), tag))


cells.insert(0, ("markdown", """# TFM — Profesores grandes y destilación hacia mmBERT-small

Las celdas 0-5 son las del cuaderno principal (`tfm_comparativa.ipynb`). Desde la sección 6: profesores grandes
(XLM-R large y un LLM de 7 B) y destilación de su conocimiento en mmBERT-small, el modelo pequeño del escenario de aplicación.

**Antes de ejecutar:** Entorno → Cambiar tipo → **A100** si está disponible (con L4 también funciona, más lento).
Duración aproximada en A100: XLM-R large ~2 h, alumno ~1 h, LLM ~2 h, segundo alumno ~1 h. Si la sesión se corta,
volver a ejecutar: lo terminado se salta (pero las probabilidades de los profesores están en el disco local de Colab:
si se pierde la sesión hay que repetir esos entrenamientos)."""))

md("""## 6. Destilación: configuración

**Qué es destilar.** Un modelo grande ya entrenado (el *profesor*) da, para cada token, una probabilidad por clase
("70 % salud, 25 % nada…"). Un modelo pequeño (el *alumno*) se entrena con dos objetivos a la vez:
- la pérdida normal con las etiquetas del corpus (entropía cruzada), y
- parecerse a las probabilidades del profesor (divergencia KL).

Esas probabilidades llevan más información que la etiqueta sola (qué casos son dudosos y con qué se confunden): es el
"conocimiento oscuro" de Hinton et al. (2015). La **temperatura** T suaviza ambas distribuciones para que el alumno vea
también las clases poco probables; la pérdida KL se multiplica por T² para que pese lo mismo con cualquier T.
Valores de la literatura: T = 2 y mitad y mitad entre las dos pérdidas (α = 0,5).""")
code(r"""
!pip -q install peft
!pip -q uninstall -y torchao    # peft rechaza el torchao 0.10 de Colab; es opcional y aquí no se usa
TEACH_DIR = f"{RUNS_DIR}/teachers"          # probabilidades de los profesores, EN DRIVE (~150 MB por profesor):
                                           # si Colab corta el entorno no se pierden (el 5-nov se perdieron así)
os.makedirs(TEACH_DIR, exist_ok=True)
KD_T, KD_ALPHA = 2.0, 0.5                  # temperatura y peso de la pérdida normal (1 - α = peso de la destilación)
STUDENT_KEY, STUDENT_LR = "mmbert-small", 3e-5   # alumno: el modelo del escenario de aplicación, con el learning rate que eligió en validación
DUMP_TEACHER_PROBS_TO_DRIVE = False        # Drive está casi lleno: las probabilidades de los profesores solo en local
""")

md("""### 6.1 Profesores grandes

**XLM-R large** (Conneau et al., 2020): 24 capas, 550 M parámetros. Es el mejor modelo en el trabajo más parecido a
este (Graehl, 2026: etiquetado de privacidad en 35 idiomas). Los modelos *large* son sensibles al learning rate
(Mosbach et al., 2021), así que se prueban 1e-5 y 2e-5. Lote por GPU de 16 con acumulación de 2 = mismo lote efectivo (32).

**Qwen2.5-7B como etiquetador de tokens.** Un LLM normal solo "mira a la izquierda" (máscara causal): para etiquetar el
token *i* no ve lo que viene después, y eso perjudica al NER. Li et al. (2023) y Dukić y Šnajder (2024) quitan esa
máscara al ajustar el modelo y mejoran mucho. Aquí se hace igual: capa de clasificación por token + LoRA (Hu et al.,
2022: solo se entrenan matrices pequeñas añadidas, ~0,5 % de los parámetros) + atención bidireccional. 2 épocas en vez
de 4 por coste (en una L4, cada época de un modelo de 7 B tarda ~2,7 h, lo que ~10 de mmBERT-small); con puntos de
control en Drive cada 200 pasos para poder continuar si Colab corta el entorno.""")
code(r"""
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"GPU: {GPU_GB:.0f} GB")


def bidirectional(model):
    # Quita la máscara causal del LLM: cada token ve también los tokens de su derecha.
    # Cómo: (1) se marca cada capa de atención como no causal y (2) antes de entrar al modelo se convierte la máscara
    # de relleno [lote, longitud] en una máscara 4D [lote, 1, longitud, longitud] que solo tapa el relleno.
    # transformers usa una máscara 4D tal cual, sin añadirle la parte causal.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = base.model                                  # el cuerpo del LLM (sin la capa de clasificación)
    for m in inner.modules():
        if hasattr(m, "is_causal"):
            m.is_causal = False

    def hook(module, args, kwargs):
        am = kwargs.get("attention_mask")
        if am is not None and am.dim() == 2:
            b, L = am.shape
            kwargs["attention_mask"] = am[:, None, None, :].bool().expand(b, 1, L, L)
        return args, kwargs

    if not getattr(inner, "_bidir_hook", False):
        inner.register_forward_pre_hook(hook, with_kwargs=True)
        inner._bidir_hook = True
    return model


def is_bidirectional(model, tok):
    # Comprobación: si cambio la ÚLTIMA palabra, ¿cambia la salida del PRIMER token? Solo pasa si ve a su derecha.
    dev = next(model.parameters()).device
    a = tok("Hola Ana, tu DNI es 12345678Z", return_tensors="pt").to(dev)
    b = tok("Hola Ana, tu DNI es 87654321X", return_tensors="pt").to(dev)
    with torch.no_grad():
        la = model(input_ids=a["input_ids"], attention_mask=a["attention_mask"]).logits[0, 0].float()
        lb = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits[0, 0].float()
    return not torch.allclose(la, lb, atol=1e-4)


def prepare_llm(model, tok):
    # LLM -> etiquetador de tokens bidireccional con LoRA.
    from peft import LoraConfig, get_peft_model
    model.config.pad_token_id = tok.pad_token_id
    if torch.cuda.is_available():
        model = model.cuda()
    model = bidirectional(model)
    assert is_bidirectional(model, tok), "la atención sigue siendo causal: revisar la versión de transformers"
    print("atención bidireccional: OK")
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="TOKEN_CLS",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)               # la capa de clasificación nueva se entrena entera (task_type)
    model.enable_input_require_grads()                # necesario con gradient checkpointing
    model.print_trainable_parameters()
    return model


class LLMPredictor(TorchPredictor):
    # Igual que TorchPredictor, pero carga el LLM en bf16 y le vuelve a quitar la máscara causal.
    def __init__(self, model_dir):
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir, dtype=torch.bfloat16).eval().cuda()
        bidirectional(self.model)
        assert is_bidirectional(self.model, self.tok)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}


TEACHER_A = "xlmr-large"
CANDIDATES[TEACHER_A] = dict(model="FacebookAI/xlm-roberta-large", tokenizer="FacebookAI/xlm-roberta-large",
                             lr_grid=[1e-5, 2e-5], per_device=16)
TEACHER_B = "qwen2.5-7b"
CANDIDATES[TEACHER_B] = dict(model="Qwen/Qwen2.5-7B", tokenizer="Qwen/Qwen2.5-7B", lr_grid=[1e-4], epochs=2,
                             ckpt_dir=f"{RUNS_DIR}/ckpt", save_steps=200,   # puntos de control en Drive (~14 min)
                             per_device=8 if GPU_GB >= 40 else 4, eval_batch=16, grad_ckpt=True,
                             model_kwargs={"dtype": torch.bfloat16}, prepare=prepare_llm)
""", tag="DISTILL_LIB")

md("""Funciones de los profesores: entrenar, evaluar (como un candidato más) y guardar sus probabilidades por token en
**train**, **val** y los tests (en el disco local). Las de train son las que aprende el alumno; las de val sirven para
elegir el conjunto de profesores sin mirar los tests.""")
code(r"""
TEACH_SETS = ("train", "val", "test", "test_long", "test_v2", "test_v3")


def rows_any(s):
    return train_rows if s == "train" else rows_for(s)


def teacher_file(name, s):
    return f"{TEACH_DIR}/{name}__{s}.pkl.gz"


def teacher_ready(name):
    return run_done(name) and all(os.path.exists(teacher_file(name, s)) for s in TEACH_SETS)


def train_teacher(key, lr, seed, predictor_cls=None):
    name = f"{key}-lr{lr:g}-s{seed}"
    if teacher_ready(name):
        print("ya hecho:", name)
        return name
    final, info = train_model(key, lr, seed, name)
    pred = (predictor_cls or TorchPredictor)(final)
    evaluate_all(pred, name, {**info, "fmt": "torch-bf16", "role": "teacher"}, dump_probs=DUMP_TEACHER_PROBS_TO_DRIVE)
    for s in TEACH_SETS:                               # probabilidades del profesor, guardadas en local
        t0 = time.time()
        probs = predict_probs(rows_any(s), pred, batch=CANDIDATES[key].get("eval_batch", 64))
        with gzip.open(teacher_file(name, s), "wb") as f:
            pickle.dump([(o, p.astype(np.float16)) for o, p in probs], f)
        print(f"  probabilidades de {name} en {s}: {time.time() - t0:.0f} s")
    del pred; gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree(Path(final).parent, ignore_errors=True)   # el modelo ya no hace falta: sus probabilidades sí
    return name


def load_teacher(name, s):
    return pickle.load(gzip.open(teacher_file(name, s)))


def val_score(name):                                   # igual que en el cuaderno principal (sección 5a)
    r = [x for x in load_results() if x["run"] == name and x["set"] == "val"]
    return r[-1]["strict_F1"] if r else None


def best_lr_for(key):
    # El learning rate de la rejilla con mejor strict F1 en validación (semilla 42).
    scores = {lr: val_score(f"{key}-lr{lr:g}-s{SEEDS[0]}") for lr in CANDIDATES[key]["lr_grid"]}
    scores = {k: v for k, v in scores.items() if v is not None}
    return max(scores, key=scores.get) if scores else None
""", tag="DISTILL_LIB")

md("""Probabilidades por **carácter**. Cada modelo trocea el texto a su manera (XLM-R, Qwen y mmBERT tienen tokenizadores
distintos), así que para combinar profesores y pasárselos al alumno se usa el carácter como unidad común:
- cada carácter recibe las 17 probabilidades del token del profesor que lo contiene;
- se apunta si ese carácter es el **inicio** de un token del profesor.

Para un token del alumno se toman las probabilidades de su primer carácter (sin contar espacios). Si ese carácter está
en mitad de un token del profesor (el alumno ha partido una palabra que el profesor tenía entera), la probabilidad de
"B-" (inicio de entidad) se pasa a "I-" (continuación): un trozo del medio de una palabra nunca empieza una entidad.
Con varios profesores se promedian sus probabilidades (conjunto, como en Hinton et al., 2015).""")
code(r"""
B_IDX = [2 * i + 1 for i in range(len(LABELS))]       # columnas B-X
I_IDX = [2 * i + 2 for i in range(len(LABELS))]       # columnas I-X


def char_table(text, offs, probs):
    cp = np.zeros((len(text) + 1, len(ID2LABEL)), dtype=np.float32)
    cp[:, 0] = 1.0                                      # por defecto "O"
    start = np.zeros(len(text) + 1, dtype=bool)
    for (a, b), p in zip(offs, probs):
        cp[a:b] = p
        start[a] = True
    return cp, start


def soft_for_offsets(text, offs, tables):
    # Probabilidades objetivo (17 por token) para una ventana del alumno; los tokens especiales quedan a cero.
    out = np.zeros((len(offs), len(ID2LABEL)), dtype=np.float32)
    for t, o in enumerate(offs):
        if o is None or o[0] == o[1]:
            continue
        a, b = o
        c = next((k for k in range(a, b) if not text[k].isspace()), None)
        acc = np.zeros(len(ID2LABEL), dtype=np.float32)
        for cp, start in tables:
            if c is None:
                # Token que solo es espacio (mmBERT a veces separa el espacio de dentro de un valor: "ES74 0081").
                # Pertenece a la entidad si el profesor dice que lo que viene después CONTINÚA esa entidad (I-X);
                # si lo que sigue empieza otra (B-X) o no es nada (O), el espacio es O. Corrección del 5-nov: antes
                # recibía siempre O y el alumno aprendía a partir IBAN, teléfonos y direcciones en trozos.
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
                if not start[c]:                       # mitad de un token del profesor: B- -> I-
                    q[I_IDX] += q[B_IDX]
                    q[B_IDX] = 0
            acc += q
        out[t] = acc / len(tables)
    return out


def char_type_probs(text, offs, probs):
    # Para combinar profesores al EVALUAR: por carácter, P(O) y P(tipo) = P(B-tipo) + P(I-tipo), y P(inicio).
    c = np.zeros((len(text), len(LABELS) + 1), np.float32)
    c[:, 0] = 1
    b = np.zeros(len(text), np.float32)
    q = probs.astype(np.float32)
    t = np.concatenate([q[:, :1], q[:, B_IDX] + q[:, I_IDX]], 1)
    for (a, e), v, qq in zip(offs, t, q):
        c[a:e] = v
        ent = qq[1:].sum()
        b[a] = qq[B_IDX].sum() / ent if ent > 0 else 0
    return c, b


def char_spans(text, c, b):
    # Tramos a partir de las probabilidades por carácter: tipo más probable por carácter; dos tramos del mismo tipo
    # separados solo por espacios se unen si el segundo es continuación (P(inicio) < 0,5).
    lab = c.argmax(1)
    runs, i = [], 0
    while i < len(text):
        if lab[i] == 0:
            i += 1
            continue
        j = i
        while j < len(text) and lab[j] == lab[i]:
            j += 1
        runs.append([i, j, lab[i]])
        i = j
    merged = []
    for r in runs:
        if merged and merged[-1][2] == r[2] and text[merged[-1][1]:r[0]].isspace() and b[r[0]] < 0.5:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    out = []
    for s, e, l in merged:
        s, e = trim_span(text, s, e, LABELS[l - 1])
        if e > s:
            out.append({"start": s, "end": e, "label": LABELS[l - 1]})
    return out


def ensemble_eval(rows, prob_lists):
    # Evalúa el conjunto (media por carácter) de varios modelos. prob_lists = [probs del modelo 1, del 2, …].
    preds = []
    for k, r in enumerate(rows):
        cs = [char_type_probs(r["text"], *pl[k]) for pl in prob_lists]
        preds.append(char_spans(r["text"], np.mean([x[0] for x in cs], 0), np.mean([x[1] for x in cs], 0)))
    return evaluate(rows, preds)
""", tag="DISTILL_LIB")

md("""El alumno: mismo `train_model` que el resto, con dos cambios.
- `kd_dataset` construye las ventanas de entrenamiento igual que siempre y guarda, para cada una, las probabilidades
  objetivo del conjunto de profesores (en la lista `SOFT`; el conjunto de datos solo lleva el índice `idx`).
- `KDTrainer` cambia la pérdida: α · entropía cruzada con las etiquetas + (1 − α) · T² · KL(profesor ‖ alumno).
  En la evaluación (validación) no hay `idx` y se usa la pérdida normal.""")
code(r"""
from transformers import Trainer
SOFT = []                                              # probabilidades objetivo por ventana del alumno
_KD_CACHE = {}                                         # mismas ventanas para las 3 semillas: se calculan una vez


def kd_dataset(teacher_runs):
    def hook(rows, tok):
        key = (tuple(teacher_runs), tok.name_or_path, len(rows))
        if key in _KD_CACHE:
            feats, soft = _KD_CACHE[key]
            SOFT[:] = soft
            return feats
        T = [load_teacher(n, "train") for n in teacher_runs]
        SOFT.clear()
        feats = {"input_ids": [], "attention_mask": [], "labels": [], "idx": []}
        for ri, r in enumerate(rows):
            tables = [char_table(r["text"], *t[ri]) for t in T]
            ents = sorted(r["entities"], key=lambda e: e["start"])
            for ids, mask, offs in windows_for_text(tok, r["text"]):
                feats["input_ids"].append(ids)
                feats["attention_mask"].append(mask)
                feats["labels"].append(align_labels(offs, ents))
                feats["idx"].append(len(SOFT))
                SOFT.append(soft_for_offsets(r["text"], offs, tables).astype(np.float16))
        _KD_CACHE.clear()                              # solo se guarda el último conjunto (memoria)
        _KD_CACHE[key] = (feats, list(SOFT))
        return feats
    return hook


class KDTrainer(Trainer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        base = self.data_collator

        def collate(features):                         # saca idx antes de rellenar y lo vuelve a añadir como tensor
            idx = [f.pop("idx") for f in features] if "idx" in features[0] else None
            batch = base(features)
            if idx is not None:
                batch["idx"] = torch.tensor(idx)
            return batch
        self.data_collator = collate

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        idx = inputs.pop("idx", None)
        out = model(**inputs)
        if idx is None:                                # evaluación: pérdida normal
            return (out.loss, out) if return_outputs else out.loss
        logits = out.logits.float()
        soft = torch.zeros_like(logits)
        for b, i in enumerate(idx.tolist()):          # relleno a la derecha: las posiciones coinciden
            s = torch.from_numpy(SOFT[i].astype(np.float32))
            soft[b, :len(s)] = s.to(logits.device)
        m = inputs["labels"] != -100                   # solo tokens reales (no especiales ni relleno)
        pt = soft[m].clamp_min(1e-8) ** (1 / KD_T)     # profesor con temperatura
        pt = pt / pt.sum(-1, keepdim=True)
        logq = torch.log_softmax(logits[m] / KD_T, -1)  # alumno con temperatura
        kd = (pt * (pt.log() - logq)).sum(-1).mean() * KD_T ** 2
        loss = KD_ALPHA * out.loss + (1 - KD_ALPHA) * kd
        return (loss, out) if return_outputs else loss


def train_student(teacher_runs, tag, seed):
    name = f"{STUDENT_KEY}-kd{tag}-lr{STUDENT_LR:g}-s{seed}"
    if run_done(name):
        print("ya hecho:", name)
        return name
    final, info = train_model(STUDENT_KEY, STUDENT_LR, seed, name, trainer_cls=KDTrainer, ds_hook=kd_dataset(teacher_runs))
    pred = TorchPredictor(final)
    evaluate_all(pred, name, {**info, "fmt": "torch-bf16", "role": "student", "teachers": teacher_runs,
                              "kd_T": KD_T, "kd_alpha": KD_ALPHA})
    del pred; gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree(Path(final).parent, ignore_errors=True)
    return name


def log_ensemble(name, teacher_runs, sets=("val", "test", "test_long", "test_v2", "test_v3")):
    # Guarda en results.jsonl el resultado del CONJUNTO de profesores (para la tabla de la memoria). Una sola vez.
    if any(r["run"] == name for r in load_results()):
        print("ya registrado:", name)
        return
    for s in sets:
        res = ensemble_eval(rows_for(s), [load_teacher(n, s) for n in teacher_runs])
        rec = {"run": name, "set": s, "fmt": "ensemble", "role": "teacher-ensemble", "members": teacher_runs, **res}
        with open(f"{RUNS_DIR}/results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{name} {s}: F1 {res['strict_F1']:.3f}  fuga {res['leak_rate']:.3f}")
""", tag="DISTILL_LIB")

md("""### 6.2 Profesor A: XLM-R large (2 learning rates con la semilla 42 → el mejor en validación → semillas 13 y 2026)""")
code(r"""
# Si la búsqueda de learning rate ya está en results.jsonl, no se repite: solo se reentrenan las 3 semillas con el
# learning rate elegido para volver a tener sus probabilidades (las de la primera ejecución estaban en el disco local).
grid_done = all(val_score(f"{TEACHER_A}-lr{lr:g}-s{SEEDS[0]}") is not None for lr in CANDIDATES[TEACHER_A]["lr_grid"])
if not grid_done:
    for lr in CANDIDATES[TEACHER_A]["lr_grid"]:
        train_teacher(TEACHER_A, lr, SEEDS[0])
lr_a = best_lr_for(TEACHER_A)
print("XLM-R large: learning rate elegido", lr_a)
for seed in SEEDS:
    train_teacher(TEACHER_A, lr_a, seed)
TEACHERS_A = [f"{TEACHER_A}-lr{lr_a:g}-s{s}" for s in SEEDS]
log_ensemble("ensemble-xlmr-large-x3", TEACHERS_A)
""")

md("""### 6.3 Alumno 1: mmBERT-small destilado del conjunto de 3 XLM-R large (3 semillas)""")
code(r"""
for seed in SEEDS:
    train_student(TEACHERS_A, "A2", seed)   # "A2": versión corregida (los "A" del 5-nov tenían el fallo de los espacios)
display(show([r for r in load_results() if r["set"] in ("val", "test_v2") and
              (r["run"].startswith(f"{STUDENT_KEY}-kd") or r["run"].startswith(f"{STUDENT_KEY}-lr{STUDENT_LR:g}"))]))
""")

md("""### 6.4 Profesor B: Qwen2.5-7B como etiquetador bidireccional (LoRA, 2 épocas, semilla 42)""")
code(r"""
TEACHER_B_RUN = train_teacher(TEACHER_B, CANDIDATES[TEACHER_B]["lr_grid"][0], SEEDS[0], predictor_cls=LLMPredictor)
""")

md("""### 6.5 ¿Qué conjunto de profesores es mejor? Se decide en **validación** (nunca con los tests)""")
code(r"""
sets_to_try = {"A": TEACHERS_A, "B": [TEACHER_B_RUN], "A+B": TEACHERS_A + [TEACHER_B_RUN]}
val_f1 = {}
for tag, runs in sets_to_try.items():
    val_f1[tag] = ensemble_eval(val_rows, [load_teacher(n, "val") for n in runs])["strict_F1"]
    print(f"profesores {tag}: strict F1 en validación = {val_f1[tag]:.4f}")
best_tag = max(val_f1, key=val_f1.get)
print("mejor conjunto:", best_tag)
log_ensemble("ensemble-qwen7b", [TEACHER_B_RUN])
log_ensemble("ensemble-xlmr-large-x3+qwen7b", TEACHERS_A + [TEACHER_B_RUN])
""")

md("""### 6.6 Alumno 2 (solo si el mejor conjunto no es A): mmBERT-small destilado del mejor conjunto""")
code(r"""
if best_tag != "A":
    for seed in SEEDS:
        train_student(sets_to_try[best_tag], best_tag.replace("+", "") + "2", seed)
else:
    print("El conjunto A sigue siendo el mejor en validación: no hace falta un segundo alumno.")
""")

md("""## 7. Resumen (test_v2): alumnos destilados frente a mmBERT-small normal y a los profesores""")
code(r"""
res = load_results()
rows = [r for r in res if r["set"] == "test_v2" and (r.get("role") in ("teacher", "student", "teacher-ensemble")
        or r["run"].startswith(f"{STUDENT_KEY}-lr{STUDENT_LR:g}"))]
display(show(rows))
from google.colab import drive as _d
_d.flush_and_unmount()                               # que todo llegue a Drive antes de cerrar
""")


def build():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for c in cells:
        if c[0] == "markdown":
            nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": c[1].splitlines(keepends=True)})
        else:
            nb["cells"].append({"cell_type": "code", "metadata": {"tags": [c[2]] if c[2] else []},
                                "execution_count": None, "outputs": [], "source": c[1].splitlines(keepends=True)})
    out = HERE / "tfm_destilacion.ipynb"
    json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
    print("escrito", out, "-", len(nb["cells"]), "celdas")


def lib_source(tags):
    return "\n\n".join(c[1] for c in cells if c[0] == "code" and c[2] in tags)


if __name__ == "__main__":
    build()
