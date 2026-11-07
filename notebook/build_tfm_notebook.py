"""Genera notebook/tfm_comparativa.ipynb (Google Colab, GPU A100).

  python notebook/build_tfm_notebook.py

¿Por qué un generador y no el cuaderno a mano? Porque así el código del cuaderno vive en un fichero .py normal,
que se puede probar en local: las celdas marcadas LIB / TRAIN_LIB / EXPORT_LIB se ejecutan también en el
portátil (CPU, con un modelo diminuto) con notebook/smoke.py, y analysis/common.py reutiliza exactamente las
mismas métricas. Si se cambia algo, se edita ESTE fichero y se vuelve a generar el .ipynb.

Contenido del cuaderno: comparativa del TFM. Cuatro codificadores multilingües ligeros entrenados con el MISMO
protocolo (misma división de datos, mismas épocas, mismo tamaño de lote, misma planificación del learning rate,
búsqueda de learning rate en validación y tres semillas), evaluados con las mismas métricas, y exportados a ONNX
en FP32 / FP16 / INT8 para medir tamaño, memoria y latencia en CPU.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = []


def md(s):
    """Añade una celda de texto (Markdown) al cuaderno."""
    CELLS.append(("markdown", s.strip("\n")))


def code(s, tag=None):
    """Añade una celda de código. `tag` marca las celdas que también se ejecutan en las pruebas locales."""
    CELLS.append(("code", s.strip("\n"), tag))


# ---------------------------------------------------------------------------
md(r"""
# TFM — Comparativa de arquitecturas ligeras para anonimización contextual en el borde

Este cuaderno entrena y evalúa, con un protocolo idéntico, cuatro codificadores Transformer multilingües para
reconocer datos personales (NER de privacidad) en texto en español, inglés y mixto:

| Clave | Modelo | Papel en la comparativa |
|---|---|---|
| `distilbert-mbert` | `distilbert/distilbert-base-multilingual-cased` | DistilBERT multilingüe (sustituye a DistilBERT/TinyBERT/MobileBERT de la propuesta, que solo existen en inglés) |
| `minilm-l12` | `microsoft/Multilingual-MiniLM-L12-H384` | Destilación profunda, el más pequeño |
| `mmbert-small` | `jhu-clsp/mmBERT-small` | Arquitectura moderna (ModernBERT multilingüe, 2025) |
| `xlmr-base` | `FacebookAI/xlm-roberta-base` | Referencia de techo (no es ligero) |

**Idea general.** Cada modelo lee un mensaje y decide, token a token, si es parte de un dato personal y de qué tipo
(clasificación de tokens con etiquetas BIO). Todos parten de un modelo ya preentrenado en muchos idiomas y se
ajustan (*fine-tuning*, aprendizaje por transferencia) con nuestro corpus.

**Protocolo** (sección 1): 4 épocas, lote 32, AdamW con decaimiento lineal y 6 % de calentamiento, *weight decay* 0,01;
se guarda la mejor época según validación. Búsqueda de *learning rate* en una rejilla de 3 valores por modelo con la
semilla 42, elegida **solo con validación** (nunca con los tests); después, dos semillas más con el learning rate
elegido. Resultado final: media ± desviación típica sobre 3 semillas.

**Antes de ejecutar**
1. Entorno de ejecución → Cambiar tipo → **GPU A100**.
2. En Google Drive, carpeta `MyDrive/tfm_pii/`: `corpus_pii_es_en_v3.zip`.
3. Ejecutar de arriba abajo. Todo se guarda en `MyDrive/tfm_pii/tfm/` conforme termina; si se corta la sesión,
   volver a ejecutar: lo ya hecho se salta.

**Etiquetas**: `PER` persona, `LOC` lugar, `ID` identificador (DNI, pasaporte…), `FIN` dato financiero, `DEMO` dato
demográfico o de categoría especial (salud, religión…), `CON` contraseña o secreto, `CONTACT` email o teléfono, y
`PUB` personaje público mencionado como tema (se etiqueta pero **no** es dato personal: nunca cuenta como ocultado
ni como fuga). En formato BIO: `B-X` = primer token de un valor de tipo X, `I-X` = tokens siguientes, `O` = fuera.

**Métricas** (sobre tramos de caracteres; idénticas para PyTorch y ONNX):
- *strict P / R / F1* — el tramo y la etiqueta tienen que ser exactos (equivale a seqeval en modo estricto, a nivel
  de entidad); también por etiqueta, y *macro F1* = media de las 7 etiquetas de datos personales.
- *overlap R* — el valor se encuentra al menos en parte con la etiqueta correcta.
- **tasa de fuga** — proporción de caracteres de datos personales que quedan sin ocultar (con cualquier etiqueta).
  Es la métrica que más importa para la privacidad.
- **falsas alarmas** — proporción de mensajes *sin* datos personales en los que el modelo oculta algo.
""")

md("""## 0. Instalación

Comprueba qué GPU ha asignado Colab e instala lo que falta: `onnx` y `onnxruntime` (exportación e inferencia en
CPU) y una versión de `transformers` con mmBERT. Colab ya trae `datasets`, `accelerate` y `pandas`; no se
actualizan para no romper los paquetes propios de Colab (que exigen su versión de pandas).""")
code(r"""
!nvidia-smi -L
!pip -q install "transformers>=4.48" onnx onnxruntime
""")

md("""## 1. Configuración

Todos los parámetros del experimento están aquí, en un solo sitio. Para repetir el experimento con otros valores
basta con cambiar esta celda.""")
code(r"""
import os, json, re, random, time, math, collections, zipfile, shutil, gc, gzip, pickle, subprocess, sys
from pathlib import Path
import numpy as np

# --- Rutas
DRIVE_DIR = "/content/drive/MyDrive/tfm_pii"
DATA_ZIP  = f"{DRIVE_DIR}/corpus_pii_es_en_v3.zip"   # corpus (train, val y los 4 conjuntos de test)
RUNS_DIR  = f"{DRIVE_DIR}/tfm"            # resultados, curvas y probabilidades (persisten en Drive)
WORK_DIR  = "/content/work"               # disco local de Colab: checkpoints temporales (se pierden al cerrar)
# Mejor modelo de cada candidato (para exportarlo a ONNX). En el disco local de Colab, NO en Drive: los 4 modelos
# ocupan ~2,7 GB y la cuenta gratuita de Drive tiene 15 GB. Si la sesión se corta y se pierden, la sección 6 los
# vuelve a entrenar (mismo learning rate y semilla 42 = mismo modelo).
BEST_DIR  = "/content/best"

# --- Etiquetas. El modelo predice 17 clases: O + (B-, I-) x 8 tipos.
LABELS = ["PER", "LOC", "ID", "FIN", "DEMO", "CON", "CONTACT", "PUB"]
NON_PII = {"PUB"}                         # se etiqueta, pero no se oculta ni cuenta como fuga
ID2LABEL = {0: "O"}                       # número de clase -> nombre ("O", "B-PER", "I-PER", ...)
for i, l in enumerate(LABELS):
    ID2LABEL[2 * i + 1] = f"B-{l}"
    ID2LABEL[2 * i + 2] = f"I-{l}"
LABEL2ID = {v: k for k, v in ID2LABEL.items()}   # y al revés

# --- Candidatos. `model` = pesos preentrenados en Hugging Face; `tokenizer` = cómo se trocea el texto
# (MiniLM usa el de XLM-R: mismo vocabulario sobre un cuerpo BERT). `lr_grid` = 3 learning rates a probar,
# centrados en el valor recomendado por sus autores o en el usado en la prueba previa.
CANDIDATES = {
    "distilbert-mbert": dict(model="distilbert/distilbert-base-multilingual-cased",
                             tokenizer="distilbert/distilbert-base-multilingual-cased", lr_grid=[3e-5, 5e-5, 8e-5]),
    "minilm-l12":       dict(model="microsoft/Multilingual-MiniLM-L12-H384",
                             tokenizer="FacebookAI/xlm-roberta-base", lr_grid=[4e-5, 6e-5, 1e-4]),
    "mmbert-small":     dict(model="jhu-clsp/mmBERT-small", tokenizer="jhu-clsp/mmBERT-small",
                             lr_grid=[3e-5, 5e-5, 8e-5]),
    "xlmr-base":        dict(model="FacebookAI/xlm-roberta-base", tokenizer="FacebookAI/xlm-roberta-base",
                             lr_grid=[2e-5, 3e-5, 5e-5]),
}
RUN_CANDIDATES = list(CANDIDATES)         # se puede reducir, p. ej. ["mmbert-small"], para pruebas rápidas

# --- Protocolo de entrenamiento: IGUAL para todos los candidatos (es lo que hace justa la comparación)
EPOCHS, BATCH, WARMUP, WEIGHT_DECAY = 4, 32, 0.06, 0.01
SEEDS = [42, 13, 2026]                    # la primera se usa en la búsqueda de learning rate
DO_LR_SEARCH = True                       # False: usa el valor central de la rejilla para todas las semillas
MAX_LEN, STRIDE = 512, 128                # ventana máxima del modelo (tokens) y solape entre ventanas

# --- Conjuntos que se evalúan, y en cuáles se guardan las probabilidades por token para el análisis local
EVAL_SETS = ("val", "test", "test_long", "test_v2", "test_v3")
PROB_SETS = ("test", "test_long", "test_v2", "test_v3")
os.makedirs(WORK_DIR, exist_ok=True)
""", tag="LIB")

md("""## 2. Datos

Monta Google Drive, crea las carpetas de resultados y descomprime el corpus. Conjuntos:
- `train` (75.098 mensajes) y `val` (4.345): entrenamiento y validación (elegir época y learning rate).
- `test_v1` (1.000), `test_long_v1` (150 textos de 1.500–3.400 caracteres), `test_v2` (395, escritos a mano y
  **nunca usados para ajustar nada**: es el test principal) y `test_v3` (102, personajes públicos, erratas,
  salud y dinero).""")
code(r"""
from google.colab import drive
drive.mount("/content/drive")
for sub in ("", "/models", "/logs", "/probs", "/onnx"):
    os.makedirs(RUNS_DIR + sub, exist_ok=True)
with zipfile.ZipFile(DATA_ZIP) as z:
    z.extractall("/content")
DATA_DIR = "/content/corpus_pii_es_en_v3"
""")

md("""Funciones para leer el corpus. Cada línea de los ficheros `.jsonl` es un mensaje:
`{"text": "...", "entities": [{"start": 10, "end": 21, "label": "PER", "text": "Lucía Martín"}, ...]}`,
con las posiciones de cada dato personal en caracteres. También se definen dos "cortes" para analizar resultados
por subgrupos: el idioma del mensaje y si contiene valores "rotos" (escritos de forma que una regla fija no los
reconocería).""")
code(r"""
def load_jsonl(path):
    # Lee un .jsonl y comprueba que las posiciones de cada entidad coinciden con su texto (detecta datos corruptos).
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in rows:
        for e in r["entities"]:
            assert r["text"][e["start"]:e["end"]] == e.get("text", r["text"][e["start"]:e["end"]]), (r.get("id"), e)
    return rows


def row_lang(r):
    # Idioma del mensaje: "es", "en" o "mixed" (según el campo lang o los metadatos de generación).
    if r.get("lang"):
        return {"mix": "mixed"}.get(r["lang"], r["lang"])
    axes = (r.get("meta") or {}).get("axes") or {}
    return {"mix": "mixed"}.get(axes.get("lang"), axes.get("lang", "?"))


def is_broken_value(e):
    # ¿Valor que una regla fija no detectaría? Letras en vez de dígitos (12O45), "arroba" en vez de @,
    # separadores raros, asteriscos… Sirve para medir la robustez frente a texto "sucio".
    t, lab = e["text"], e["label"]
    if lab == "CONTACT":
        if re.search(r"arroba|\bat\b|\(at\)|\[at\]|\bAT\b|\bpunto\b|\bdot\b", t, re.I):
            return True
        if "@" in t:
            return bool(re.search(r"\s|@@|,|@[\w-]+$|@$|\.con$|gmai\.|gmial|hotmial|yahooo", t)) or not re.search(r"@[\w-]+\.\w", t)
        digits = re.sub(r"\D", "", t)
        return bool(re.search(r"[A-Za-z]", t)) or len(digits) not in (9, 11, 12, 13) or bool(re.search(r"\d \d \d", t))
    if lab in ("ID", "FIN", "LOC"):
        return bool(re.search(r"\d[OoIlS?]|[OoIlS?]\d", t)) or "*" in t or "\n" in t or bool(re.search(r"\d \d \d \d", t))
    return False


def row_broken(r):
    return any(is_broken_value(e) for e in r["entities"])


def dataset_stats(name, rows):
    # Una fila de la tabla de estadísticas del corpus (capítulo 4 de la memoria).
    labs = collections.Counter(e["label"] for r in rows for e in r["entities"])
    return {"set": name, "rows": len(rows), "entities": sum(labs.values()),
            "no_pii_share": round(sum(not any(e["label"] not in NON_PII for e in r["entities"]) for r in rows) / max(len(rows), 1), 3),
            "chars_mean": round(float(np.mean([len(r["text"]) for r in rows])), 1) if rows else 0,
            **{f"n_{l}": labs.get(l, 0) for l in LABELS},
            **{f"lang_{k}": v for k, v in collections.Counter(row_lang(r) for r in rows).items()}}
""", tag="LIB")

md("Carga los seis conjuntos y muestra (y guarda en Drive) sus estadísticas: mensajes, entidades por tipo, idiomas y proporción de mensajes sin datos personales.")
code(r"""
train_rows = load_jsonl(f"{DATA_DIR}/train.jsonl")
val_rows   = load_jsonl(f"{DATA_DIR}/val.jsonl")
test_rows  = load_jsonl(f"{DATA_DIR}/test_v1.jsonl")
long_rows  = load_jsonl(f"{DATA_DIR}/test_long_v1.jsonl")
test2_rows = load_jsonl(f"{DATA_DIR}/test_v2.jsonl")
test3_rows = load_jsonl(f"{DATA_DIR}/test_v3.jsonl")
import pandas as pd
stats = pd.DataFrame([dataset_stats(n, r) for n, r in [("train", train_rows), ("val", val_rows), ("test_v1", test_rows),
                      ("test_long", long_rows), ("test_v2", test2_rows), ("test_v3", test3_rows)]]).fillna(0)
stats.to_csv(f"{RUNS_DIR}/dataset_stats.csv", index=False)
display(stats)
""")

md("""## 3. Tokenización y alineación de etiquetas

Los modelos no ven caracteres sino **tokens** (trozos de palabra: "Martín" puede ser "Mart" + "ín"). Hay que
traducir las etiquetas del corpus (en caracteres) a una etiqueta por token:
- un token recibe `B-X` si es el primero que solapa un valor de tipo X, `I-X` los siguientes, y `O` el resto;
- los tokens especiales (inicio/fin de secuencia) reciben -100, que el entrenamiento ignora.

Los modelos solo aceptan 512 tokens a la vez, así que los textos largos se dividen en **ventanas solapadas**
(512 tokens, solape de 128): no se trunca nada, y en la parte solapada se promedian las dos predicciones.""")
code(r"""
_SPECIALS = {}


def _special_ids(tok):
    # Averigua qué tokens especiales pone el tokenizador antes y después de una secuencia (p. ej. <s> y </s>),
    # tokenizando "a" con y sin ellos. Se guarda en caché por tokenizador.
    key = id(tok)
    if key not in _SPECIALS:
        with_sp = tok("a")["input_ids"]
        body = tok("a", add_special_tokens=False)["input_ids"]
        i = next(i for i in range(len(with_sp)) if with_sp[i:i + len(body)] == body)
        _SPECIALS[key] = (with_sp[:i], with_sp[i + len(body):])
    return _SPECIALS[key]


def windows_for_text(tok, text, max_len=MAX_LEN, stride=STRIDE):
    # Trocea un texto en ventanas de como mucho `max_len` tokens que se solapan `stride` tokens.
    # Devuelve [(ids, máscara de atención, offsets)]; offsets = posición (inicio, fin) en caracteres de cada token,
    # o None para los tokens especiales. Se hace a mano porque transformers 5 pierde texto con su opción automática.
    pre, suf = _special_ids(tok)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    ids, offs = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    body = max_len - len(pre) - len(suf)          # tokens de texto que caben en una ventana
    out, start = [], 0
    while True:
        w_ids = pre + ids[start:start + body] + suf
        w_offs = [None] * len(pre) + offs[start:start + body] + [None] * len(suf)
        out.append((w_ids, [1] * len(w_ids), w_offs))
        if start + body >= len(ids):
            return out
        start += body - stride                    # la siguiente ventana empieza `stride` tokens antes del final


def align_labels(offsets, entities):
    # Convierte las entidades (en caracteres) en una etiqueta numérica por token (B-/I-/O, o -100 si se ignora).
    labels, prev = [], None
    for o in offsets:
        if o is None or o[0] == o[1]:             # token especial o vacío: el entrenamiento lo ignora
            labels.append(-100)
            continue
        s, e = o
        hit = None
        for k, ent in enumerate(entities):        # ¿qué entidad solapa este token?
            if s < ent["end"] and e > ent["start"]:
                hit = k
                break
        if hit is None:
            labels.append(0)                      # O
            prev = None
        else:
            ent = entities[hit]
            # B- si es el primer token de esa entidad, I- si continúa la misma entidad que el token anterior
            labels.append(LABEL2ID[("I-" if prev == hit else "B-") + ent["label"]])
            prev = hit
    return labels


def encode_rows(rows, tok):
    # Prepara un conjunto de mensajes para el entrenador: ids, máscara y etiquetas por ventana.
    feats = {"input_ids": [], "attention_mask": [], "labels": []}
    for r in rows:
        ents = sorted(r["entities"], key=lambda e: e["start"])
        for ids, mask, offs in windows_for_text(tok, r["text"]):
            feats["input_ids"].append(ids)
            feats["attention_mask"].append(mask)
            feats["labels"].append(align_labels(offs, ents))
    return feats
""", tag="LIB")

md("""## 4. Inferencia y métricas (comunes a PyTorch y ONNX)

Para poder comparar todo con la misma vara, la evaluación es igual para el modelo en GPU (PyTorch) y para el
modelo exportado en CPU (ONNX): ambos son un "predictor" con un método `logits(ids, máscara)`. El proceso:
1. `predict_probs`: ventanas → modelo → probabilidades por token (softmax), promediando las ventanas solapadas.
2. `spans_from_probs` / `decode_spans`: en cada token se elige la clase más probable y los tokens seguidos del
   mismo tipo se unen en un tramo de caracteres (`{"start", "end", "label"}`), al que se recortan espacios y signos.
3. `evaluate`: compara los tramos predichos con los anotados y calcula todas las métricas.""")
code(r"""
class TorchPredictor:
    # Modelo de PyTorch en GPU (precisión bfloat16), para evaluar justo después de entrenar.
    def __init__(self, model_dir):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).eval().cuda()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def logits(self, ids, mask):
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            out = self.model(input_ids=self.torch.tensor(ids).cuda(), attention_mask=self.torch.tensor(mask).cuda())
        return out.logits.float().cpu().numpy()


class OnnxPredictor:
    # Modelo exportado a ONNX, ejecutado en CPU con ONNX Runtime (como en el navegador o en un portátil).
    def __init__(self, onnx_path, tok_dir, threads=4):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads         # hilos de CPU (la extensión del navegador usa 1)
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(tok_dir)
        cfg = json.load(open(Path(tok_dir) / "config.json"))
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def logits(self, ids, mask):
        feed = {"input_ids": np.array(ids, dtype=np.int64), "attention_mask": np.array(mask, dtype=np.int64)}
        return self.sess.run(["logits"], {k: v for k, v in feed.items() if k in self.inputs})[0]


TRIM_LEAD, TRIM_TAIL = "([{¿¡\"'«", ".,;:)]}\"'»"


def trim_span(text, s, e, label):
    # Quita espacios y signos de puntuación de los bordes de un tramo ("(Lucía)," -> "Lucía").
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    if label != "CON":  # una contraseña sí puede empezar o acabar con signos de puntuación
        while e > s and text[e - 1] in TRIM_TAIL and not (text[e - 1] == ")" and "(" in text[s:e]):
            e -= 1
        while s < e and text[s] in TRIM_LEAD and not (text[s] == "(" and ")" in text[s:e]):
            s += 1
    return s, e


def decode_spans(text, tokens, id2label):
    # tokens: [(inicio, fin, clase)] ordenados por posición. Une tokens consecutivos B-X, I-X, I-X… en un tramo.
    spans, cur = [], None
    for s, e, lid in tokens:
        lab = id2label[lid]
        if lab == "O":                            # fuera de cualquier dato: cierra el tramo en curso
            if cur:
                spans.append(cur)
                cur = None
            continue
        bio, typ = lab.split("-", 1)
        if cur and typ == cur[2] and bio == "I":  # continúa el tramo del mismo tipo
            cur = (cur[0], e, typ)
        else:                                     # empieza un tramo nuevo
            if cur:
                spans.append(cur)
            cur = (s, e, typ)
    if cur:
        spans.append(cur)
    out = []
    for s, e, typ in spans:
        s, e = trim_span(text, s, e, typ)
        if e > s:
            out.append({"start": s, "end": e, "label": typ})
    return out


def predict_probs(rows, predictor, batch=64):
    # Para cada mensaje devuelve (offsets [n,2], probabilidades [n,17]) por token. Las ventanas de todos los mensajes
    # se ordenan por longitud y se agrupan en lotes (menos relleno = más rápido); al final se reparten por mensaje,
    # promediando los tokens que aparecen en dos ventanas solapadas.
    tok = predictor.tok
    windows = []
    for i, r in enumerate(rows):
        for ids, mask, offs in windows_for_text(tok, r["text"]):
            windows.append((i, ids, mask, offs))
    windows.sort(key=lambda w: len(w[1]))
    acc = collections.defaultdict(dict)           # mensaje -> {offset del token: [suma de probabilidades, nº de ventanas]}
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for b in range(0, len(windows), batch):
        chunk = windows[b:b + batch]
        L = max(len(w[1]) for w in chunk)
        ids = [w[1] + [pad] * (L - len(w[1])) for w in chunk]          # relleno hasta la ventana más larga del lote
        mask = [w[2] + [0] * (L - len(w[2])) for w in chunk]
        lg = predictor.logits(ids, mask).astype(np.float32)
        p = np.exp(lg - lg.max(-1, keepdims=True))                     # softmax: logits -> probabilidades
        p /= p.sum(-1, keepdims=True)
        for k, (ri, _, _, offs) in enumerate(chunk):
            for t, o in enumerate(offs):
                if o is None or o[0] == o[1]:
                    continue
                a = acc[ri].setdefault(o, [0.0, 0])
                a[0] = a[0] + p[k, t]
                a[1] += 1
    out = []
    n_cls = len(predictor.id2label)
    for i in range(len(rows)):
        items = sorted(acc[i].items())
        offs = np.array([o for o, _ in items], dtype=np.int32).reshape(-1, 2)
        probs = np.array([v[0] / v[1] for _, v in items], dtype=np.float32).reshape(-1, n_cls)
        out.append((offs, probs))
    return out


def spans_from_probs(rows, probs, id2label):
    # Clase más probable por token (argmax) -> tramos de caracteres.
    preds = []
    for r, (offs, p) in zip(rows, probs):
        toks = [(int(s), int(e), int(k)) for (s, e), k in zip(offs, p.argmax(-1))] if len(offs) else []
        preds.append(decode_spans(r["text"], toks, id2label))
    return preds


def predict_rows(rows, predictor, batch=64):
    return spans_from_probs(rows, predict_probs(rows, predictor, batch), predictor.id2label)


def evaluate(rows, preds):
    # Compara predicciones y anotaciones de un conjunto de mensajes. Devuelve un diccionario de métricas.
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()   # por etiqueta
    ov_hit = collections.Counter(); ov_tot = collections.Counter()
    gold_chars = leaked = 0
    leak_by = collections.Counter(); chars_by = collections.Counter()
    neg_rows = neg_alarm = 0
    per_chars = per_as_pub = pub_chars = pub_masked = 0
    for r, P in zip(rows, preds):
        G = r["entities"]
        # 1) Coincidencia estricta: mismo inicio, fin y etiqueta.
        gset = {(e["start"], e["end"], e["label"]) for e in G}
        pset = {(e["start"], e["end"], e["label"]) for e in P}
        for x in pset & gset: tp[x[2]] += 1       # acierto
        for x in pset - gset: fp[x[2]] += 1       # predicho pero no anotado
        for x in gset - pset: fn[x[2]] += 1       # anotado pero no predicho
        # 2) Fuga: qué caracteres quedan ocultos (cualquier etiqueta salvo PUB, que nunca se oculta).
        covered = np.zeros(len(r["text"]) + 1, dtype=bool)
        as_pub = np.zeros(len(r["text"]) + 1, dtype=bool)
        for e in P:
            (as_pub if e["label"] in NON_PII else covered)[e["start"]:e["end"]] = True
        for e in G:
            ov_tot[e["label"]] += 1
            if any(p["label"] == e["label"] and p["start"] < e["end"] and p["end"] > e["start"] for p in P):
                ov_hit[e["label"]] += 1           # encontrado al menos en parte con la etiqueta correcta
            n = e["end"] - e["start"]
            if e["label"] in NON_PII:             # personaje público: ¿se ha ocultado de más?
                pub_chars += n; pub_masked += int(covered[e["start"]:e["end"]].sum())
                continue
            miss = int(n - covered[e["start"]:e["end"]].sum())         # caracteres de este dato que se escapan
            gold_chars += n; leaked += miss
            chars_by[e["label"]] += n; leak_by[e["label"]] += miss
            if e["label"] == "PER":               # persona confundida con personaje público (no se oculta)
                per_chars += n; per_as_pub += int((as_pub[e["start"]:e["end"]] & ~covered[e["start"]:e["end"]]).sum())
        # 3) Falsa alarma: mensaje sin datos personales en el que se oculta algo.
        if not any(e["label"] not in NON_PII for e in G):
            neg_rows += 1
            neg_alarm += any(p["label"] not in NON_PII for p in P)
    res = {}
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    res["strict_P"] = TP / max(TP + FP, 1)                         # precisión
    res["strict_R"] = TP / max(TP + FN, 1)                         # exhaustividad (recall)
    res["strict_F1"] = 2 * TP / max(2 * TP + FP + FN, 1)           # media armónica de las dos
    f1s = []
    for l in LABELS:
        d = 2 * tp[l] + fp[l] + fn[l]
        res[f"P_{l}"] = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else float("nan")
        res[f"R_{l}"] = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else float("nan")
        res[f"F1_{l}"] = 2 * tp[l] / d if d else float("nan")     # nan = la etiqueta no aparece en este conjunto
        if d and l not in NON_PII:
            f1s.append(res[f"F1_{l}"])
        if l not in NON_PII:
            res[f"leak_{l}"] = leak_by[l] / chars_by[l] if chars_by[l] else float("nan")
    res["macro_F1"] = float(np.mean(f1s)) if f1s else float("nan")   # media de las etiquetas de datos personales
    res["overlap_R"] = sum(ov_hit.values()) / max(sum(ov_tot.values()), 1)
    res["leak_rate"] = leaked / max(gold_chars, 1)
    res["false_alarm"] = neg_alarm / neg_rows if neg_rows else float("nan")
    res["per_as_pub"] = per_as_pub / per_chars if per_chars else float("nan")
    res["pub_masked"] = pub_masked / pub_chars if pub_chars else float("nan")
    res["n_rows"] = len(rows)
    return res


def rows_for(s):
    return {"val": val_rows, "test": test_rows, "test_long": long_rows, "test_v2": test2_rows, "test_v3": test3_rows}[s]


def save_probs(name, s, probs):
    # Guarda en Drive las probabilidades por token (en float16 para que ocupen la mitad). Con ellas se puede, en
    # local y sin GPU: cambiar el umbral de decisión, aplicar el post-proceso y hacer pruebas estadísticas.
    with gzip.open(f"{RUNS_DIR}/probs/{name}__{s}.pkl.gz", "wb") as f:
        pickle.dump([(o, p.astype(np.float16)) for o, p in probs], f)


def evaluate_all(predictor, name, extra=None, sets=EVAL_SETS, dump_probs=True):
    # Evalúa un modelo en varios conjuntos (y en sus cortes por idioma y por valores rotos) y añade cada resultado
    # como una línea en results.jsonl. `extra` = datos del experimento (modelo, learning rate, semilla, tamaño…).
    out = []
    for s in sets:
        rows = rows_for(s)
        if not rows:
            continue
        t0 = time.time()
        probs = predict_probs(rows, predictor)
        dt = time.time() - t0
        preds = spans_from_probs(rows, probs, predictor.id2label)
        if dump_probs and s in PROB_SETS:
            save_probs(name, s, probs)
        out.append({"run": name, "set": s, "secs": round(dt, 1), **(extra or {}), **evaluate(rows, preds)})
        if s in ("test", "test_v2", "test_v3"):   # cortes: test_v2[es], test_v2[en], test_v2[mixed], test_v2[broken]
            for lang in ("es", "en", "mixed"):
                idx = [i for i, r in enumerate(rows) if row_lang(r) == lang]
                if idx:
                    out.append({"run": name, "set": f"{s}[{lang}]", **(extra or {}),
                                **evaluate([rows[i] for i in idx], [preds[i] for i in idx])})
            idx = [i for i, r in enumerate(rows) if row_broken(r)]
            if idx:
                out.append({"run": name, "set": f"{s}[broken]", **(extra or {}),
                            **evaluate([rows[i] for i in idx], [preds[i] for i in idx])})
    with open(f"{RUNS_DIR}/results.jsonl", "a") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    return out


def load_results():
    p = f"{RUNS_DIR}/results.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def run_done(name):
    # ¿Este experimento ya terminó en una sesión anterior? (permite reanudar tras una desconexión)
    return any(r["run"] == name and r["set"] == "val" for r in load_results())


def show(records, cols=("run", "set", "strict_F1", "macro_F1", "overlap_R", "leak_rate", "false_alarm",
                        "F1_PER", "F1_LOC", "F1_ID", "F1_FIN", "F1_DEMO", "F1_CON", "F1_CONTACT", "F1_PUB")):
    # Tabla compacta para ver resultados en el cuaderno.
    import pandas as pd
    df = pd.DataFrame(records)
    cols = [c for c in cols if c in df.columns] + [c for c in ("size_mb", "ram_mb", "ms_1t_1000ch") if c in df.columns]
    return df[cols].round(4)
""", tag="LIB")

md("""## 5. Entrenamiento

`train_model` ajusta un candidato con el protocolo común usando el `Trainer` de Hugging Face:
- añade al modelo preentrenado una capa de clasificación por token (17 clases) que empieza aleatoria;
- en cada época evalúa en validación (*macro F1* por token) y al final conserva la mejor época;
- guarda en Drive el historial de pérdidas y métricas (para las **curvas de aprendizaje** de la memoria).

`train_and_eval` entrena y evalúa en todos los conjuntos, y se salta lo que ya está hecho.""")
code(r"""
import inspect
import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,
                          DataCollatorForTokenClassification, set_seed)
from datasets import Dataset
torch.backends.cuda.matmul.allow_tf32 = True      # multiplicaciones más rápidas en A100 sin pérdida apreciable


def compute_metrics(p):
    # Métrica rápida por token que usa el Trainer para elegir la mejor época (no es la métrica final del TFM).
    # Junta B-X e I-X en la misma clase X y calcula el F1 de cada tipo y la media.
    preds, labels = p.predictions.argmax(-1), p.label_ids
    m = labels != -100
    pt = np.where(preds > 0, (preds + 1) // 2, 0)[m]
    lt = np.where(labels > 0, (labels + 1) // 2, 0)[m]
    f1 = []
    for t in range(1, len(LABELS) + 1):
        tp = ((pt == t) & (lt == t)).sum(); fp = ((pt == t) & (lt != t)).sum(); fn = ((pt != t) & (lt == t)).sum()
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    pii_recall = ((pt > 0) & (lt > 0)).sum() / max((lt > 0).sum(), 1)
    return {"tok_macro_f1": float(np.mean(f1)), "tok_pii_recall": float(pii_recall)}


_TA_PARAMS = inspect.signature(TrainingArguments.__init__).parameters
# lotes de longitud parecida = menos relleno (transformers 5 renombró group_by_length)
GROUP_BY_LENGTH = ({"train_sampling_strategy": "group_by_length"} if "train_sampling_strategy" in _TA_PARAMS
                   else {"group_by_length": True})


def count_params(model):
    # Parámetros totales y cuántos son de la tabla de embeddings (el vocabulario): explica buena parte del tamaño.
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    return total, emb


def train_model(key, lr, seed, run_name, trainer_cls=None, ds_hook=None):
    # Devuelve (carpeta del modelo final, información del entrenamiento). Mismo protocolo para todos los candidatos.
    # Opciones de CANDIDATES[key] para modelos grandes (si faltan, el protocolo común de siempre):
    #   per_device  lote por GPU; el resto se acumula hasta BATCH (mismo lote efectivo, menos memoria)
    #   epochs      épocas (el LLM de 7 B usa menos, por coste)
    #   model_kwargs  argumentos de carga (p. ej. dtype bf16 para el LLM)
    #   prepare     función (modelo, tokenizador) -> modelo: añade LoRA, atención bidireccional…
    #   grad_ckpt   gradient checkpointing (recalcula activaciones para ahorrar memoria)
    # trainer_cls y ds_hook los usa la destilación: otro Trainer (pérdida con el profesor) y otro conjunto de
    # entrenamiento (con el índice de cada ventana para buscar sus probabilidades del profesor).
    cfg = CANDIDATES[key]
    set_seed(seed)                                 # misma semilla = mismo resultado (reproducibilidad)
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    t0 = time.time()
    ds_tr = Dataset.from_dict(ds_hook(train_rows, tok) if ds_hook else encode_rows(train_rows, tok))
    ds_va = Dataset.from_dict(encode_rows(val_rows, tok))
    print(f"[{run_name}] ventanas: train {len(ds_tr)}, val {len(ds_va)} (tokenizado en {time.time()-t0:.0f}s)")
    # modelo preentrenado + capa de clasificación nueva con nuestras 17 clases
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model"], num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID, **cfg.get("model_kwargs", {}))
    n_params, n_emb = count_params(model)
    vocab = int(model.get_input_embeddings().weight.shape[0])
    if "prepare" in cfg:
        model = cfg["prepare"](model, tok)
    out_dir = f"{WORK_DIR}/{run_name}"
    per_device = cfg.get("per_device", BATCH)
    epochs = cfg.get("epochs", EPOCHS)
    warmup = int(WARMUP * math.ceil(len(ds_tr) / BATCH) * epochs)   # 6 % de los pasos con learning rate creciente
    if cfg.get("ckpt_dir"):
        # Entrenamientos largos (el LLM): puntos de control cada `save_steps` pasos EN DRIVE, para continuar donde iba
        # si Colab corta el entorno. Sin evaluación por época (es una sola época: no hay época que elegir).
        ckpt_dir = f"{cfg['ckpt_dir']}/{run_name}"
        schedule = dict(output_dir=ckpt_dir, eval_strategy="no", save_strategy="steps",
                        save_steps=cfg.get("save_steps", 200), save_total_limit=1)
    else:
        ckpt_dir = None
        schedule = dict(output_dir=out_dir, eval_strategy="epoch", save_strategy="epoch",
                        load_best_model_at_end=True,     # evaluar cada época y quedarse con la mejor
                        metric_for_best_model="tok_macro_f1", greater_is_better=True, save_total_limit=1)
    args = TrainingArguments(
        per_device_train_batch_size=per_device, gradient_accumulation_steps=BATCH // per_device,
        per_device_eval_batch_size=cfg.get("eval_batch", 64),
        learning_rate=lr, num_train_epochs=epochs, weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup, lr_scheduler_type="linear", bf16=True,          # bf16: más rápido en A100
        gradient_checkpointing=cfg.get("grad_ckpt", False), remove_unused_columns=ds_hook is None,
        logging_steps=50, report_to="none", seed=seed, dataloader_num_workers=2, **schedule, **GROUP_BY_LENGTH,
        **({"data_seed": seed} if "data_seed" in _TA_PARAMS else {}))
    trainer = (trainer_cls or Trainer)(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                                       data_collator=DataCollatorForTokenClassification(tok), processing_class=tok,
                                       compute_metrics=compute_metrics)
    resume = None
    if ckpt_dir and os.path.isdir(ckpt_dir):
        found = sorted(Path(ckpt_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        resume = str(found[-1]) if found else None
        if resume:
            print(f"[{run_name}] se continúa desde {resume}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    minutes = (time.time() - t0) / 60
    print(f"[{run_name}] entrenado en {minutes:.1f} min")
    json.dump(trainer.state.log_history, open(f"{RUNS_DIR}/logs/{run_name}.json", "w"))   # curvas de aprendizaje
    if ckpt_dir:                                   # terminado: el punto de control de Drive ya no hace falta
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    best_epoch = None                              # qué época resultó la mejor en validación
    if trainer.state.best_model_checkpoint:
        step = int(trainer.state.best_model_checkpoint.rstrip("/").split("-")[-1])
        best_epoch = next((h["epoch"] for h in trainer.state.log_history if h.get("step") == step and "eval_loss" in h), None)
    final = f"{out_dir}/final"
    if hasattr(trainer.model, "merge_and_unload"):  # LoRA: se suman los adaptadores al modelo y se guarda completo
        trainer.model.merge_and_unload().save_pretrained(final)
    else:
        trainer.save_model(final)
    tok.save_pretrained(final)
    info = {"model": key, "lr": lr, "seed": seed, "train_min": round(minutes, 2), "best_epoch": best_epoch,
            "params_m": round(n_params / 1e6, 1), "emb_params_m": round(n_emb / 1e6, 1),
            "vocab": vocab, "train_windows": len(ds_tr)}
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()         # liberar memoria de la GPU antes del siguiente modelo
    return final, info


def train_and_eval(key, lr, seed):
    # Entrena un candidato con un learning rate y una semilla, y lo evalúa en todos los conjuntos.
    name = f"{key}-lr{lr:g}-s{seed}"               # p. ej. "mmbert-small-lr5e-05-s42"
    if run_done(name):
        print("ya hecho:", name)
        return name, None
    final, info = train_model(key, lr, seed, name)
    pred = TorchPredictor(final)
    evaluate_all(pred, name, {**info, "fmt": "torch-bf16"})
    del pred; gc.collect(); torch.cuda.empty_cache()
    return name, final
""", tag="TRAIN_LIB")

md("""### 5a. Búsqueda de learning rate (semilla 42) — se elige SOLO con validación

El *learning rate* (tamaño de cada paso de ajuste) es el hiperparámetro que más influye. Para no favorecer a ningún
modelo, a cada uno se le prueban 3 valores y se queda con el mejor según el *strict F1* en **validación**. Los tests
no intervienen en la elección (si lo hicieran, sus resultados serían optimistas).

El modelo con el mejor learning rate se guarda en el disco local (`BEST_DIR/<clave>`) para exportarlo en la sección 6.
Duración orientativa: 6–10 min por entrenamiento en A100; en L4, ~9 min DistilBERT y algo más los modelos grandes.""")
code(r"""
def val_score(name):
    r = [x for x in load_results() if x["run"] == name and x["set"] == "val"]
    return r[-1]["strict_F1"] if r else None


def best_lr_for(key):
    # El learning rate de la rejilla con mejor strict F1 en validación (entre los ya entrenados).
    scores = {lr: val_score(f"{key}-lr{lr:g}-s{SEEDS[0]}") for lr in CANDIDATES[key]["lr_grid"]}
    scores = {k: v for k, v in scores.items() if v is not None}
    return max(scores, key=scores.get) if scores else None


BEST_LR_FILE = f"{RUNS_DIR}/best_lr.json"
best_lr = json.load(open(BEST_LR_FILE)) if os.path.exists(BEST_LR_FILE) else {}
for key in RUN_CANDIDATES:
    grid = CANDIDATES[key]["lr_grid"] if DO_LR_SEARCH else [CANDIDATES[key]["lr_grid"][1]]
    for lr in grid:
        name, final = train_and_eval(key, lr, SEEDS[0])
        if final and best_lr_for(key) == lr:   # es el mejor hasta ahora: se guarda (o sustituye) en el disco local
            shutil.rmtree(f"{BEST_DIR}/{key}", ignore_errors=True)
            shutil.copytree(final, f"{BEST_DIR}/{key}")
            json.dump({"run": name, "lr": lr, "seed": SEEDS[0]}, open(f"{BEST_DIR}/{key}/origin.json", "w"))
        if final:
            shutil.rmtree(Path(final).parent, ignore_errors=True)   # borrar checkpoints locales (ocupan mucho)
    best_lr[key] = best_lr_for(key)
    json.dump(best_lr, open(BEST_LR_FILE, "w"), indent=1)
    print(key, "→ learning rate elegido:", best_lr[key])
display(show([r for r in load_results() if r["set"] == "val"]))
""")

md("""### 5b. Semillas adicionales con el learning rate elegido

Un mismo modelo entrenado con otra semilla (otro orden de los datos y otra inicialización de la capa nueva) da
resultados algo distintos. Entrenar con 3 semillas permite dar media ± desviación típica y saber si una diferencia
entre modelos es real o puro azar.""")
code(r"""
for key in RUN_CANDIDATES:
    for seed in SEEDS[1:]:
        name, final = train_and_eval(key, best_lr[key], seed)
        if final:
            shutil.rmtree(Path(final).parent, ignore_errors=True)
""")

md("### 5c. Resumen: media ± desviación típica sobre las 3 semillas (con el learning rate elegido)")
code(r"""
def seed_summary(results, sets=("test_v2", "test", "test_long", "test_v3"), metrics=("strict_F1", "macro_F1", "leak_rate", "false_alarm")):
    df = pd.DataFrame(results)
    df = df[df.fmt == "torch-bf16"]
    keep = []
    for key, lr in best_lr.items():               # solo las ejecuciones con el learning rate elegido
        keep += [f"{key}-lr{lr:g}-s{s}" for s in SEEDS]
    df = df[df.run.isin(keep) & df.set.isin(sets)]
    agg = df.groupby(["model", "set"])[list(metrics)].agg(["mean", "std", "count"])
    return agg.round(4)


summary = seed_summary(load_results())
summary.to_csv(f"{RUNS_DIR}/summary_seeds.csv")
display(summary)
""")

md("""## 6. Exportación a ONNX y eficiencia (FP32 / FP16 / INT8)

Para ejecutar el modelo en el navegador (o en cualquier CPU) se exporta a **ONNX**, un formato estándar que
entiende ONNX Runtime (también en WebAssembly). Se prueban cuatro precisiones numéricas de los pesos:
- **fp32**: la original (4 bytes por peso);
- **fp16**: media precisión (2 bytes): mitad de tamaño;
- **int8-matmul**: los pesos del codificador en enteros de 8 bits (por canal); embeddings y clasificador en fp32;
- **int8-all**: cuantización dinámica por defecto de ONNX Runtime (todo en 8 bits: el más pequeño).

Para cada variante se mide: tamaño del fichero; eficacia en test_v2, test_long y test_v3 (¿cuánto se pierde al
reducir precisión?); **latencia** en CPU (mediana de 15 mensajes, 1 y 4 hilos, ~200, ~1.000 y ~2.000 caracteres) y
**RAM** (memoria extra que usa el proceso al cargar y ejecutar el modelo, medida en un proceso aparte para que no
se mezcle con la memoria del cuaderno). Las cifras de Colab son orientativas; las definitivas se miden en el
portátil y en el navegador.""")
code(r"""
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_fp32(src_dir, out_dir):
    # Exporta el modelo de PyTorch a ONNX (fp32) y comprueba que da los mismos resultados que PyTorch.
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    tok = AutoTokenizer.from_pretrained(src_dir)
    try:
        model = AutoModelForTokenClassification.from_pretrained(src_dir, attn_implementation="eager")
    except (TypeError, ValueError):
        model = AutoModelForTokenClassification.from_pretrained(src_dir)
    model.eval().float().cpu()
    sample = tok(["Hola, soy Ana García y mi DNI es 12345678Z.", "Hi"], padding=True, return_tensors="pt")

    class Wrapper(torch.nn.Module):
        # El modelo devuelve un objeto; ONNX necesita un tensor: el envoltorio devuelve solo los logits.
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapped = Wrapper(model).eval()
    path = f"{out_dir}/model_fp32.onnx"
    # dynamic_axes: el tamaño del lote y la longitud del texto pueden variar en cada llamada
    kwargs = dict(input_names=["input_ids", "attention_mask"], output_names=["logits"], opset_version=17,
                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                "logits": {0: "batch", 1: "seq"}}, do_constant_folding=True)
    try:   # exportador clásico (TorchScript); si la versión de torch no lo permite, se usa el nuevo (dynamo)
        torch.onnx.export(wrapped, (sample["input_ids"], sample["attention_mask"]), path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapped, (sample["input_ids"], sample["attention_mask"]), path, **kwargs)
    except Exception as err:
        print("Falló el exportador TorchScript, se usa dynamo:", err)
        torch.onnx.export(wrapped, (sample["input_ids"], sample["attention_mask"]), path, dynamo=True, **kwargs)
    m = onnx.load(path, load_external_data=True)
    onnx.save(m, path, save_as_external_data=False)   # un solo fichero
    tok.save_pretrained(out_dir)
    model.config.save_pretrained(out_dir)
    # Comprobación: misma salida en ONNX y en PyTorch (diferencia máxima < 0,001)
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    o = sess.run(["logits"], {"input_ids": sample["input_ids"].numpy(), "attention_mask": sample["attention_mask"].numpy()})[0]
    with torch.no_grad():
        t = model(**{k: sample[k] for k in ("input_ids", "attention_mask")}).logits.numpy()
    real = sample["attention_mask"].numpy().astype(bool)
    diff = float(np.abs(o - t)[real].max())
    print(f"max |onnx - torch| en tokens reales = {diff:.2e}")
    assert diff < 1e-3, "la exportación ONNX no coincide con PyTorch"
    return path


def export_fp16(fp32_path, out_dir):
    # Convierte los pesos a fp16 con la herramienta de ONNX Runtime (entradas y salidas siguen en fp32/int64).
    from onnxruntime.transformers.onnx_model import OnnxModel
    m = OnnxModel(onnx.load(fp32_path))
    m.convert_float_to_float16(keep_io_types=True)
    path = f"{out_dir}/model_fp16.onnx"
    m.save_model_to_file(path)
    return path


def classifier_nodes(fp32_path):
    # Nombres de los nodos de la capa final de clasificación, para dejarla en fp32 al cuantizar (es pequeña y sensible).
    g = onnx.load(fp32_path).graph
    producers = {o: n for n in g.node for o in n.output}
    names, frontier = [], ["logits"]
    while frontier:  # recorre el grafo desde la salida hacia atrás hasta encontrar la multiplicación del clasificador
        n = producers.get(frontier.pop())
        if n is None:
            continue
        names.append(n.name)
        if n.op_type in ("MatMul", "Gemm"):
            break
        frontier += [i for i in n.input if i in producers]
    return names


def export_int8(fp32_path, out_dir, mode="matmul"):
    # Cuantización dinámica a int8: los pesos se guardan en 8 bits y las activaciones se cuantizan al vuelo.
    path = f"{out_dir}/model_int8_{mode}.onnx"
    if mode == "all":
        quantize_dynamic(fp32_path, path, weight_type=QuantType.QInt8)
    else:
        quantize_dynamic(fp32_path, path, weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul"], nodes_to_exclude=classifier_nodes(fp32_path))
    return path


LAT_CHARS = (200, 1000, 2000)                      # longitudes de mensaje para medir la latencia


def latency_texts(rows, n_chars, k=8, seed=0):
    # Mensajes de prueba de `n_chars` caracteres, pegando mensajes reales del test hasta llegar a esa longitud.
    rng = random.Random(seed)
    texts = []
    for _ in range(k):
        buf = ""
        while len(buf) < n_chars:
            buf += rng.choice(rows)["text"] + "\n"
        texts.append(buf[:n_chars])
    return texts


def latency_ms(onnx_path, tok_dir, threads, n_chars, rows, reps=15):
    # Mediana del tiempo de un mensaje completo (tokenizar + ventanas + modelo), de uno en uno, como en la extensión.
    pred = OnnxPredictor(onnx_path, tok_dir, threads=threads)
    texts = latency_texts(rows, n_chars)
    predict_probs([{"text": texts[0]}], pred, batch=1)  # calentamiento (la primera llamada siempre es más lenta)
    times = []
    for i in range(reps):
        t0 = time.perf_counter()
        predict_probs([{"text": texts[i % len(texts)]}], pred, batch=1)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# Pequeño programa que se ejecuta en un proceso aparte: carga el modelo, lo ejecuta 3 veces con un texto de
# 2.000 caracteres e informa de la memoria residente (antes de cargar, tras cargar y el pico, muestreado cada 2 ms).
# Corrección del 5-nov: antes se usaba ru_maxrss, pero en Linux el proceso hijo hereda el máximo del cuaderno
# (6,7 GB en Colab) y la medida salía 0. Ahora se usa la memoria residente actual (psutil), que no se hereda.
RAM_PROBE = r'''
import sys, json, threading, time
import numpy as np, psutil, onnxruntime as ort
from tokenizers import Tokenizer
path, tok_json, text = sys.argv[1], sys.argv[2], open(sys.argv[3]).read()
proc = psutil.Process()
enc = Tokenizer.from_file(tok_json).encode(text)
ids = np.array([enc.ids[:512]], dtype=np.int64)
feed_all = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
base = proc.memory_info().rss
peak = [base]
stop = False
def sample():                       # hilo que apunta el máximo de memoria mientras el modelo trabaja
    while not stop:
        peak[0] = max(peak[0], proc.memory_info().rss); time.sleep(0.002)
th = threading.Thread(target=sample, daemon=True); th.start()
so = ort.SessionOptions(); so.intra_op_num_threads = 1
sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
loaded = proc.memory_info().rss
names = {i.name for i in sess.get_inputs()}
for _ in range(3):
    sess.run(["logits"], {k: v for k, v in feed_all.items() if k in names})
stop = True; th.join()
MB = 1024 * 1024
print(json.dumps({"ram_base_mb": base / MB, "ram_loaded_mb": loaded / MB, "ram_peak_mb": max(peak[0], loaded) / MB}))
'''


def ram_mb(onnx_path, tok_dir, rows):
    probe = f"{WORK_DIR}/ram_probe.py"
    open(probe, "w").write(RAM_PROBE)
    txt = f"{WORK_DIR}/ram_text.txt"
    open(txt, "w").write(latency_texts(rows, 2000, k=1)[0])
    out = subprocess.run([sys.executable, "-I", probe, onnx_path, f"{tok_dir}/tokenizer.json", txt],
                         capture_output=True, text=True, check=True)
    r = json.loads(out.stdout.strip().splitlines()[-1])
    r = {k: round(v, 1) for k, v in r.items()}
    r["ram_mb"] = round(r["ram_peak_mb"] - r["ram_base_mb"], 1)   # lo que añade el modelo (cargado + inferencia)
    return r


def efficiency(onnx_path, tok_dir, rows):
    # Todas las medidas de eficiencia de una variante: tamaño, latencias (1 y 4 hilos, 3 longitudes) y RAM.
    rec = {"size_mb": round(os.path.getsize(onnx_path) / 1e6, 1)}
    for th in (1, 4):
        for n in LAT_CHARS:
            rec[f"ms_{th}t_{n}ch"] = round(latency_ms(onnx_path, tok_dir, th, n, rows), 1)
    rec.update(ram_mb(onnx_path, tok_dir, rows))
    return rec
""", tag="EXPORT_LIB")

md("""Para cada candidato: exporta su mejor modelo a las 4 precisiones, mide eficiencia y eficacia, y guarda en Drive
las variantes que usaría el navegador (fp16 e int8-all) de los tres modelos ligeros; de XLM-R (la referencia, que no
se plantea llevar al navegador) solo se guardan las métricas, para no llenar Drive.""")
code(r"""
ONNX_VARIANTS = ("fp32", "fp16", "int8-matmul", "int8-all")
ONNX_TO_DRIVE = ("model_fp16.onnx", "model_int8_all.onnx")   # + tokenizador y configuración
LIGHT = ("distilbert-mbert", "minilm-l12", "mmbert-small")


def best_model_dir(key):
    # Carpeta del mejor modelo de `key`: disco local; si no está (sesión nueva), la copia antigua en Drive;
    # y si tampoco, se vuelve a entrenar con el learning rate elegido y la semilla 42 (mismo modelo).
    for d in (f"{BEST_DIR}/{key}", f"{RUNS_DIR}/models/{key}-best"):
        if os.path.exists(f"{d}/origin.json"):
            return d
    lr = best_lr[key]
    name = f"{key}-lr{lr:g}-s{SEEDS[0]}"
    print(f"{key}: no hay modelo guardado, se vuelve a entrenar {name} (solo para exportarlo)")
    final, _ = train_model(key, lr, SEEDS[0], name + "-reexport")
    d = f"{BEST_DIR}/{key}"
    shutil.copytree(final, d, dirs_exist_ok=True)
    json.dump({"run": name, "lr": lr, "seed": SEEDS[0]}, open(f"{d}/origin.json", "w"))
    shutil.rmtree(Path(final).parent, ignore_errors=True)
    return d


for key in RUN_CANDIDATES:
    src = best_model_dir(key)
    origin = json.load(open(f"{src}/origin.json"))   # de qué entrenamiento viene (learning rate y semilla)
    exp = f"{WORK_DIR}/export/{key}"
    os.makedirs(exp, exist_ok=True)
    fp32 = export_fp32(src, exp)
    paths = {"fp32": fp32, "fp16": export_fp16(fp32, exp),
             "int8-matmul": export_int8(fp32, exp, "matmul"), "int8-all": export_int8(fp32, exp, "all")}
    for fmt in ONNX_VARIANTS:
        name = f"{origin['run']}:onnx-{fmt}"
        if any(r["run"] == name for r in load_results()):
            print("ya hecho:", name); continue
        eff = efficiency(paths[fmt], exp, test2_rows)
        pred = OnnxPredictor(paths[fmt], exp, threads=8)   # 8 hilos solo para evaluar más deprisa
        evaluate_all(pred, name, {"model": key, "lr": origin["lr"], "seed": origin["seed"], "fmt": f"onnx-{fmt}", **eff},
                     sets=("test_v2", "test_long", "test_v3"), dump_probs=(fmt != "fp32"))
        print(name, eff)
    if key in LIGHT:                                # a Drive solo lo que usaría el navegador
        dst = f"{RUNS_DIR}/onnx/{key}"
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(exp):
            if not f.endswith(".onnx") or f in ONNX_TO_DRIVE:
                shutil.copy(f"{exp}/{f}", dst)
    shutil.rmtree(exp, ignore_errors=True)
eff_cols = ["run", "set", "strict_F1", "leak_rate", "false_alarm", "size_mb", "ram_mb", "ms_1t_200ch", "ms_1t_1000ch", "ms_1t_2000ch", "ms_4t_1000ch"]
df = pd.DataFrame([r for r in load_results() if str(r.get("fmt", "")).startswith("onnx") and r["set"] == "test_v2"])
display(df[[c for c in eff_cols if c in df.columns]].round(4))
""")

md("""## 7. Copia final

Guarda el resumen por semillas y todos los resultados en CSV. Para el análisis local hay que descargar de
`MyDrive/tfm_pii/tfm/`: `results.jsonl`, `logs/`, `probs/`, `best_lr.json`, `dataset_stats.csv` y `onnx/`
(`models/distilbert-mbert-best` es de la primera ejecución y se puede borrar al terminar).""")
code(r"""
summary = seed_summary(load_results())
summary.to_csv(f"{RUNS_DIR}/summary_seeds.csv")
pd.DataFrame(load_results()).to_csv(f"{RUNS_DIR}/results_all.csv", index=False)
print("Todo en", RUNS_DIR)
""")


# ---------------------------------------------------------------------------
def build():
    """Escribe el .ipynb (formato JSON de Jupyter) con las celdas definidas arriba."""
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for c in CELLS:
        if c[0] == "markdown":
            nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": c[1].splitlines(keepends=True)})
        else:
            nb["cells"].append({"cell_type": "code", "metadata": {"tags": [c[2]] if c[2] else []},
                                "execution_count": None, "outputs": [], "source": c[1].splitlines(keepends=True)})
    out = HERE / "tfm_comparativa.ipynb"
    json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
    print("escrito", out, "-", len(nb["cells"]), "celdas")


def lib_source(tags=("LIB",)):
    """Código de las celdas con esas etiquetas, para ejecutarlo en las pruebas locales."""
    return "\n\n".join(c[1] for c in CELLS if c[0] == "code" and c[2] in tags)


if __name__ == "__main__":
    build()
