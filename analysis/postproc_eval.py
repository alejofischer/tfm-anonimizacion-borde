"""Efecto del post-proceso sobre las probabilidades guardadas en Colab (sin volver a ejecutar ningún modelo).

  python analysis/postproc_eval.py <carpeta dataset v3> <pub_names.txt> [<carpeta del modelo de producción>]

Para cada entrenamiento (4 candidatos × 3 semillas, PyTorch) y cada conjunto de test compara tres decodificaciones:
  - argmax: la clase más probable por token (lo que mide el cuaderno; es la métrica «del modelo solo»);
  - balanced (umbral 0,90) y strict (umbral 0,98): un token se oculta si P(O) + P(PUB) < umbral, más ampliación
    a palabra completa, filtro de personajes públicos y reglas fijas (lib/postprocess.pipeline, el mismo código que
    usa el producto). Es la métrica «del sistema completo».
Si se indica la carpeta del modelo de producción (mmBERT-small v3, entrenado el 14-oct con otro
cuaderno), se evalúa también con exactamente las mismas métricas, como referencia externa.

Resultados en results/postproc.jsonl.
"""
import gzip
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import postprocess  # noqa: E402  (lib/postprocess.py)

RES = common.ROOT / "results"
OUT = RES / "postproc.jsonl"
MODES = {"argmax": None, "balanced": 0.90, "strict": 0.98}
SETS = ("test_v2", "test", "test_long", "test_v3")


def tfm_runs():
    best_lr = json.load(open(RES / "colab" / "best_lr.json"))
    names = {p.name.split("__")[0] for p in (RES / "colab" / "probs").glob("*.pkl.gz") if ":onnx" not in p.name}
    runs = []
    for m, lr in best_lr.items():
        for s in (42, 13, 2026):
            base = f"{m}-lr{lr:g}-s{s}"
            runs.append((m, s, base + "-r2" if base + "-r2" in names else base))
    return runs


def spans_for(text, offs, probs, mode, pub_names):
    tokens = [(int(a), int(b), p.astype("float32")) for (a, b), p in zip(offs, probs)]
    if mode == "argmax":
        return common.decode_spans(text, [(s, e, int(p.argmax())) for s, e, p in tokens], common.ID2LABEL)
    return postprocess.pipeline(text, tokens, common.ID2LABEL, threshold=MODES[mode], pub_names=pub_names)


def main():
    data_dir, pub_file = sys.argv[1], sys.argv[2]
    prod_dir = sys.argv[3] if len(sys.argv) > 3 else None
    pub_names = set(Path(pub_file).read_text(encoding="utf-8").split("\n"))
    tests = common.load_tests(data_dir)
    out = []
    for model, seed, run in tfm_runs():
        for s in SETS:
            probs = pickle.load(gzip.open(RES / "colab" / "probs" / f"{run}__{s}.pkl.gz"))
            rows = tests[s]
            for mode in MODES:
                preds = [spans_for(r["text"], o, p, mode, pub_names) for r, (o, p) in zip(rows, probs)]
                rec = {"run": run, "model": model, "seed": seed, "set": s, "mode": mode, **common.evaluate(rows, preds)}
                out.append(rec)
            print(run, s, {m: round(x["strict_F1"], 3) for m, x in zip(MODES, out[-3:])},
                  {m: round(x["leak_rate"], 3) for m, x in zip(MODES, out[-3:])}, flush=True)
    if prod_dir:   # modelo de producción (referencia)
        sys.path.insert(0, prod_dir)
        import predict  # noqa: E402
        det = predict.PiiDetector()
        for s in SETS:
            rows = tests[s]
            toks = [det.token_probs(r["text"]) for r in rows]
            for mode in MODES:
                if mode == "argmax":
                    preds = [common.decode_spans(r["text"], [(a, b, int(p.argmax())) for a, b, p in t], det.id2label)
                             for r, t in zip(rows, toks)]
                else:
                    preds = [postprocess.pipeline(r["text"], t, det.id2label, threshold=MODES[mode], pub_names=pub_names)
                             for r, t in zip(rows, toks)]
                out.append({"run": "production-v3", "model": "production-v3", "seed": None, "set": s, "mode": mode,
                            **common.evaluate(rows, preds)})
            print("producción", s, {m: round(x["strict_F1"], 3) for m, x in zip(MODES, out[-3:])},
                  {m: round(x["leak_rate"], 3) for m, x in zip(MODES, out[-3:])}, flush=True)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
