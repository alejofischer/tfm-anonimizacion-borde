"""Eficiencia en el borde: exporta a ONNX el mejor modelo de cada candidato y mide tamaño, latencia y RAM en ESTE
ordenador (el portátil del autor), con el mismo código de exportación y de latencia que el cuaderno de Colab.

  python analysis/edge_bench.py <carpeta con los modelos *-best> <carpeta del dataset v3> [<carpeta de trabajo>]

Por qué aquí y no en Colab: (1) el objetivo es el borde (un portátil corriente, sin GPU), y la CPU de Colab es un
servidor; (2) en Colab la RAM salió mal medida: el proceso de medida, lanzado desde el cuaderno, heredaba el máximo de
memoria del propio cuaderno (6,7 GB), así que el modelo nunca lo superaba y la diferencia daba 0.

RAM: se mide en un proceso nuevo y vacío (sin torch), con psutil: memoria residente antes de cargar, después de cargar
y el pico durante 3 inferencias de un texto de 2.000 caracteres (muestreo cada 2 ms en un hilo aparte).
Además se comprueba que cada variante ONNX da el mismo strict F1 en test_v2 que en Colab (exportación correcta).

Resultados en results/edge_mac.jsonl (una línea por modelo y precisión).
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

NS = common.NS
exec(common._B.lib_source(("EXPORT_LIB",)), NS)          # export_fp32, export_fp16, export_int8, latency_ms…
OUT = common.ROOT / "results" / "edge_mac.jsonl"
VARIANTS = ("fp32", "fp16", "int8-matmul", "int8-all", "int8-matmul-rr", "int8-all-rr")


def export_int8_rr(fp32_path, out_dir, mode):
    # Igual que export_int8 del cuaderno pero con reduce_range=True (pesos en 7 bits). Motivo: en CPU x86 sin
    # instrucciones VNNI (como el i7-8559U de este portátil), ONNX Runtime puede desbordar al multiplicar en 8 bits y
    # el modelo cuantizado «normal» da basura (F1 = 0 con mmBERT). Con VNNI (servidores de Colab) no pasa.
    from onnxruntime.quantization import quantize_dynamic, QuantType
    path = f"{out_dir}/model_int8_{mode}_rr.onnx"
    if mode == "all":
        quantize_dynamic(fp32_path, path, weight_type=QuantType.QInt8, reduce_range=True)
    else:
        quantize_dynamic(fp32_path, path, weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
                         op_types_to_quantize=["MatMul"], nodes_to_exclude=NS["classifier_nodes"](fp32_path))
    return path

RAM_PROBE = r'''
import sys, json, threading, time
import numpy as np, psutil, onnxruntime as ort
from tokenizers import Tokenizer
path, tok_json, text = sys.argv[1], sys.argv[2], open(sys.argv[3]).read()
proc = psutil.Process()
tok = Tokenizer.from_file(tok_json)
enc = tok.encode(text)
ids = np.array([enc.ids[:512]], dtype=np.int64)
feed_all = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
base = proc.memory_info().rss
peak = [base]
stop = False
def sample():
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


def ram(onnx_path, tok_dir, rows, work):
    probe, txt = f"{work}/ram_probe.py", f"{work}/ram_text.txt"
    open(probe, "w").write(RAM_PROBE)
    open(txt, "w").write(NS["latency_texts"](rows, 2000, k=1)[0])
    out = subprocess.run([sys.executable, "-I", probe, onnx_path, f"{tok_dir}/tokenizer.json", txt],
                         capture_output=True, text=True, check=True)
    r = {k: round(v, 1) for k, v in json.loads(out.stdout.strip().splitlines()[-1]).items()}
    r["ram_model_mb"] = round(r["ram_loaded_mb"] - r["ram_base_mb"], 1)    # lo que ocupa el modelo cargado
    r["ram_total_mb"] = round(r["ram_peak_mb"] - r["ram_base_mb"], 1)      # modelo + inferencia (pico)
    return r


def done():
    if not OUT.exists():
        return set()
    return {json.loads(l)["run"] for l in open(OUT)}


def main():
    models_dir, data_dir = sys.argv[1], sys.argv[2]
    work = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("TMPDIR", "/tmp") + "/edge"
    test2 = common.load_jsonl(f"{data_dir}/test_v2.jsonl")
    NS["test2_rows"] = test2
    machine = {"machine": platform.machine(), "cpu": subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor()}
    finished = done()
    for d in sorted(Path(models_dir).glob("*-best")):
        key = d.name[:-len("-best")]
        origin = json.load(open(d / "origin.json"))
        # Los modelos los guardó transformers 5 (Colab); el transformers 4.x de este portátil no conoce su clase de
        # tokenizador genérica. Es el mismo tokenizer.json: basta con nombrar la clase equivalente de la 4.x.
        # Lo mismo con la configuración de ModernBERT: la 5.x guarda el «theta» de RoPE en rope_parameters, y la 4.x,
        # si no encuentra global/local_rope_theta, usa 10.000 en las capas locales (mmBERT usa 160.000 en todas).
        cf = d / "config.json"
        mc = json.load(open(cf))
        rp = mc.get("rope_parameters")
        if mc.get("model_type") == "modernbert" and isinstance(rp, dict) and "local_rope_theta" not in mc:
            mc["global_rope_theta"] = rp["full_attention"]["rope_theta"]
            mc["local_rope_theta"] = rp["sliding_attention"]["rope_theta"]
            json.dump(mc, open(cf, "w"), indent=1)
        tc = d / "tokenizer_config.json"
        cfg = json.load(open(tc))
        if cfg.get("tokenizer_class") == "TokenizersBackend":
            cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            if isinstance(cfg.get("extra_special_tokens"), list):   # la 4.x lo espera como diccionario
                cfg.pop("extra_special_tokens")
            json.dump(cfg, open(tc, "w"), indent=1)
        exp = f"{work}/{key}"
        os.makedirs(exp, exist_ok=True)
        paths = {}
        for fmt in VARIANTS:
            name = f"{origin['run']}:onnx-{fmt}"
            if name in finished:
                print("ya hecho:", name)
                continue
            if not paths:      # exportar solo si hace falta medir algo de este modelo
                fp32 = f"{exp}/model_fp32.onnx"
                if not os.path.exists(fp32):
                    NS["export_fp32"](str(d), exp)
                paths = {"fp32": fp32,
                         "fp16": f"{exp}/model_fp16.onnx" if os.path.exists(f"{exp}/model_fp16.onnx") else NS["export_fp16"](fp32, exp),
                         "int8-matmul": f"{exp}/model_int8_matmul.onnx" if os.path.exists(f"{exp}/model_int8_matmul.onnx") else NS["export_int8"](fp32, exp, "matmul"),
                         "int8-all": f"{exp}/model_int8_all.onnx" if os.path.exists(f"{exp}/model_int8_all.onnx") else NS["export_int8"](fp32, exp, "all")}
                for mode in ("matmul", "all"):
                    q = f"{exp}/model_int8_{mode}_rr.onnx"
                    paths[f"int8-{mode}-rr"] = q if os.path.exists(q) else export_int8_rr(fp32, exp, mode)
            p = paths[fmt]
            rec = {"run": name, "model": key, "fmt": f"onnx-{fmt}", **machine,
                   "size_mb": round(os.path.getsize(p) / 1e6, 1)}
            for th in (1, 4):
                for n in NS["LAT_CHARS"]:
                    rec[f"ms_{th}t_{n}ch"] = round(NS["latency_ms"](p, exp, th, n, test2), 1)
            rec.update(ram(p, exp, test2, work))
            pred = NS["OnnxPredictor"](p, exp, threads=4)
            rec["test_v2_strict_F1"] = round(common.evaluate(test2, NS["predict_rows"](test2, pred))["strict_F1"], 4)
            print(json.dumps(rec))
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
