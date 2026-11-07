# 6-nov, 15:00: Qwen terminó (parada temprana en la época 3; se usa la época 2) y guardó el modelo en el disco local,
# pero la evaluación posterior falló. Aquí se evalúa ese modelo guardado SIN volver a entrenar: se sustituye
# train_model por una versión que solo devuelve la carpeta del modelo y sus datos.
_name = f"{TEACHER_B}-lr{CANDIDATES[TEACHER_B]['lr_grid'][0]:g}-s{SEEDS[0]}"
_final = f"{WORK_DIR}/{_name}/final"
if not teacher_ready(_name) and os.path.exists(f"{_final}/model.safetensors"):
    _curve = json.load(open(f"{RUNS_DIR}/logs/{_name}-epochs.json"))
    print("curva en validación por época:", _curve)
    _train_model = train_model
    def train_model(key, lr, seed, run_name, **kw):
        # parámetros del modelo base sin LoRA (7.111.111.202 - 40.431.121 entrenables) y su tabla de embeddings
        return _final, {"model": key, "lr": lr, "seed": seed, "train_min": None, "best_epoch": max(_curve, key=lambda x: x[1])[0],
                        "params_m": 7070.7, "emb_params_m": 545.0, "vocab": 152064, "train_windows": 75103}
    try:
        TEACHER_B_RUN = train_teacher(TEACHER_B, CANDIDATES[TEACHER_B]["lr_grid"][0], SEEDS[0], predictor_cls=LLMPredictor)
    finally:
        train_model = _train_model
else:
    TEACHER_B_RUN = train_teacher(TEACHER_B, CANDIDATES[TEACHER_B]["lr_grid"][0], SEEDS[0], predictor_cls=LLMPredictor)
