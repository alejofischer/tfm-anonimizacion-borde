"""Tablas de resultados para la memoria (capítulo 5), a partir de results/colab/results.jsonl, results/baselines.jsonl
y results/edge_mac.jsonl.

  python analysis/tables.py

Escribe en results/tables/ un CSV por tabla y un resumen en Markdown (tablas.md) para revisarlas de un vistazo.

Decisión documentada: las semillas 13 y 2026 de mmBERT-small de la primera ejecución se sustituyen por su repetición
(«-r2»): en la primera ejecución la evaluación tras recargar el modelo no reproducía la validación del propio
entrenamiento (ver registro_colab.md y el apartado de amenazas a la validez). Si la repetición no existe todavía, se
usan las originales y se avisa.
"""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "tables"
MODELS = ["distilbert-mbert", "minilm-l12", "mmbert-small", "xlmr-base"]
NAMES = {"distilbert-mbert": "DistilBERT multilingüe", "minilm-l12": "Multilingual MiniLM-L12",
         "mmbert-small": "mmBERT-small", "xlmr-base": "XLM-R base (referencia)",
         "baseline-rules": "Solo reglas", "baseline-presidio": "Presidio + spaCy"}
TEST_SETS = ["test_v2", "test", "test_long", "test_v3"]
LABELS = ["PER", "LOC", "ID", "FIN", "DEMO", "CON", "CONTACT"]


def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def seed_runs(results, best_lr):
    """Nombre de run de cada (modelo, semilla) con el learning rate elegido; mmBERT s13/s2026 → versión r2."""
    names = {r["run"] for r in results}
    out = {}
    for m in MODELS:
        lr = best_lr[m]
        for s in (42, 13, 2026):
            base = f"{m}-lr{lr:g}-s{s}"
            r2 = base + "-r2"
            out[(m, s)] = r2 if r2 in names else base
            if m == "mmbert-small" and s != 42 and r2 not in names:
                print(f"AVISO: falta {r2}; se usa la ejecución original")
    return out


def get(results, run, s):
    rs = [r for r in results if r["run"] == run and r["set"] == s]
    return rs[-1] if rs else None


def fmt(mean, std=None, pct=False, d=3):
    k = 100 if pct else 1
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "—"
    if pct:
        return f"{mean*k:.1f} ± {std*k:.1f}" if std is not None else f"{mean*k:.1f}"
    return f"{mean:.{d}f} ± {std:.{d}f}" if std is not None else f"{mean:.{d}f}"


def write(name, header, rows, md):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.csv", "w") as f:
        f.write(";".join(header) + "\n")
        for r in rows:
            f.write(";".join(str(x) for x in r) + "\n")
    md.append(f"\n### {name}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |\n")


def main():
    results = load(RES / "colab" / "results.jsonl")
    base = load(RES / "baselines.jsonl")
    edge = load(RES / "edge_mac.jsonl")
    best_lr = json.load(open(RES / "colab" / "best_lr.json"))
    runs = seed_runs(results, best_lr)
    md = ["# Tablas de resultados (generadas por analysis/tables.py)\n"]

    # T1. Eficacia principal: media ± desviación típica sobre 3 semillas, por conjunto de test.
    metrics = [("strict_F1", False), ("macro_F1", False), ("leak_rate", True), ("false_alarm", True)]
    for s in TEST_SETS:
        rows = []
        for m in MODELS:
            recs = [get(results, runs[(m, sd)], s) for sd in (42, 13, 2026)]
            recs = [r for r in recs if r]
            row = [NAMES[m], len(recs)]
            for k, pct in metrics:
                vals = [r[k] for r in recs if r.get(k) is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
                row.append(fmt(statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, pct) if vals else "—")
            rows.append(row)
        for b in ("baseline-rules", "baseline-presidio"):
            r = get(base, b, s)
            if r:
                rows.append([NAMES[b], "—"] + [fmt(r[k], None, pct) for k, pct in metrics])
        write(f"T1_eficacia_{s}", ["Modelo", "Semillas", "F1 estricto", "Macro F1", "Fuga (%)", "Falsas alarmas (%)"], rows, md)

    # T2. F1 estricto por etiqueta en test_v2 (media de 3 semillas).
    rows = []
    for m in MODELS:
        recs = [get(results, runs[(m, sd)], "test_v2") for sd in (42, 13, 2026)]
        row = [NAMES[m]]
        for l in LABELS:
            vals = [r[f"F1_{l}"] for r in recs if r and not math.isnan(r[f"F1_{l}"])]
            row.append(f"{statistics.mean(vals):.3f}" if vals else "—")
        rows.append(row)
    for b in ("baseline-rules", "baseline-presidio"):
        r = get(base, b, "test_v2")
        if r:
            rows.append([NAMES[b]] + [f"{r[f'F1_{l}']:.3f}" if not math.isnan(r[f"F1_{l}"]) else "—" for l in LABELS])
    write("T2_f1_por_etiqueta_test_v2", ["Modelo"] + LABELS, rows, md)

    # T3. Por idioma y valores «rotos» en test_v2 (F1 estricto y fuga, media de 3 semillas).
    slices = ["test_v2[es]", "test_v2[en]", "test_v2[mixed]", "test_v2[broken]"]
    rows = []
    for m in MODELS:
        row = [NAMES[m]]
        for s in slices:
            recs = [get(results, runs[(m, sd)], s) for sd in (42, 13, 2026)]
            f1 = [r["strict_F1"] for r in recs if r]
            lk = [r["leak_rate"] for r in recs if r]
            row.append(f"{statistics.mean(f1):.3f} / {100*statistics.mean(lk):.1f} %" if f1 else "—")
        rows.append(row)
    write("T3_idioma_y_rotos_test_v2", ["Modelo", "Español", "Inglés", "Mixto", "Valores rotos"], rows, md)

    # T4. Precisión numérica (ONNX, semilla 42): calidad en test_v2 frente a PyTorch.
    rows = []
    for m in MODELS:
        run42 = runs[(m, 42)]
        t = get(results, run42, "test_v2")
        rows.append([NAMES[m], "PyTorch (bf16, GPU)", "—", fmt(t["strict_F1"]), fmt(t["leak_rate"], None, True)])
        for v in ("fp32", "fp16", "int8-matmul", "int8-all"):
            r = get(results, f"{run42}:onnx-{v}", "test_v2")
            if r:
                rows.append([NAMES[m], f"ONNX {v}", r.get("size_mb", "—"), fmt(r["strict_F1"]), fmt(r["leak_rate"], None, True)])
    write("T4_onnx_calidad_test_v2", ["Modelo", "Formato", "Tamaño (MB)", "F1 estricto", "Fuga (%)"], rows, md)

    # T5. Eficiencia en el portátil (si ya se midió).
    if edge:
        rows = []
        for r in edge:
            rows.append([NAMES.get(r["model"], r["model"]), r["fmt"].replace("onnx-", ""), r["size_mb"],
                         r["ram_model_mb"], r["ram_total_mb"], r["ms_1t_200ch"], r["ms_1t_1000ch"], r["ms_1t_2000ch"],
                         r["ms_4t_1000ch"], r["test_v2_strict_F1"]])
        write("T5_eficiencia_portatil", ["Modelo", "Precisión", "Tamaño (MB)", "RAM modelo (MB)", "RAM pico (MB)",
                                         "ms 200 car. (1 hilo)", "ms 1000 car. (1 hilo)", "ms 2000 car. (1 hilo)",
                                         "ms 1000 car. (4 hilos)", "F1 test_v2"], rows, md)

    # T6. Coste de entrenamiento y tamaño de los modelos.
    rows = []
    for m in MODELS:
        recs = [get(results, runs[(m, sd)], "val") for sd in (42, 13, 2026)]
        r = recs[0]
        mins = [x["train_min"] for x in recs if x]
        rows.append([NAMES[m], r["params_m"], r["emb_params_m"], r["vocab"], best_lr[m],
                     f"{statistics.mean(mins):.1f}", "/".join(str(int(x["best_epoch"])) if x and x.get("best_epoch") else "?" for x in recs)])
    write("T6_modelos_y_coste", ["Modelo", "Parámetros (M)", "De ellos, embeddings (M)", "Vocabulario",
                                 "Learning rate elegido", "Minutos por entrenamiento (L4)", "Mejor época (s42/s13/s2026)"], rows, md)

    (OUT / "tablas.md").write_text("".join(md))
    print("escrito", OUT)


if __name__ == "__main__":
    main()
