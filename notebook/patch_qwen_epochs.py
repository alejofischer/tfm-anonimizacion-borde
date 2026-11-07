# 6.3 bis (5-nov, noche) — Qwen2.5-7B con evaluación por época y parada temprana.
#
# Antes: 2 épocas fijas, sin evaluar entre épocas (se usaba el modelo del final).
# Ahora: mismo máximo de épocas que los demás modelos (EPOCHS = 4), se evalúa en validación al final de cada época y
# se para cuando una época no mejora a la mejor anterior (parada temprana, paciencia 1; Prechelt, 1998). Se usa la
# MEJOR época, igual que con los modelos pequeños (métrica tok_macro_f1 en validación). La curva por época queda en
# logs/<run>.json y en ckpt/<run>/epochs.json: sirve para mostrar en la memoria cuántas épocas necesita un modelo
# grande preentrenado (Devlin et al., 2019: 2-4 épocas; Li et al., 2023 y Dukić y Šnajder, 2024: pocas épocas con LoRA).
#
# Cómo se guarda (todo en Drive, para continuar si Colab corta el entorno):
#   - punto de control completo cada 200 pasos (como antes): para retomar a mitad de una época;
#   - los adaptadores LoRA + capa de clasificación de CADA época (epoch-N.pt, ~160 MB): para recuperar la mejor.
# Nota: el learning rate baja linealmente hasta 0 a lo largo de las 4 épocas; si se para antes, la mejor época se
# entrenó con un learning rate algo mayor que en un plan de menos épocas. Es lo normal en la parada temprana.
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict

# min_delta (decidido el 6-nov ANTES de ver la época 2): una época que mejora menos de 0,002 de tok_macro_f1 en
# validación no compensa otra época de 7 B (~2,7 h de L4) y se para (umbral de mejora mínima; Prechelt, 1998).
CANDIDATES[TEACHER_B].update(epochs=EPOCHS, patience=1, min_delta=0.002)


def _epoch_curve(state):
    # [(época, tok_macro_f1 en validación), ...] en el orden en que se evaluaron
    return [(round(h["epoch"]), h["eval_tok_macro_f1"]) for h in state.log_history if "eval_tok_macro_f1" in h]


def _stop_reason(curve, patience, min_delta):
    # ¿Hay que parar tras la última época evaluada? Devuelve el motivo o None.
    if not curve:
        return None
    ep = curve[-1][0]
    best_ep, best_f1 = max(curve, key=lambda x: x[1])
    if ep - best_ep >= patience:
        return f"la época {ep} no mejora a la {best_ep}"
    if len(curve) > 1:
        gain = curve[-1][1] - max(f for _, f in curve[:-1])
        if gain < min_delta:
            return f"la época {ep} mejora solo {gain:+.4f} (< {min_delta})"
    return None


class EpochKeeper(TrainerCallback):
    # Tras evaluar al final de cada época: guarda los pesos entrenables de esa época y decide si parar.
    def __init__(self, folder, patience, min_delta=0.0):
        self.folder, self.patience, self.min_delta, self.trainer = folder, patience, min_delta, None

    def on_train_begin(self, args, state, control, **kwargs):
        # al continuar desde un punto de control posterior a una evaluación que ya pedía parar: se para enseguida
        why = _stop_reason(_epoch_curve(state), self.patience, self.min_delta)
        if why:
            print(f"parada temprana (ya decidida antes del corte): {why}")
            control.should_training_stop = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        ep = round(state.epoch)
        os.makedirs(self.folder, exist_ok=True)
        torch.save(get_peft_model_state_dict(self.trainer.model), f"{self.folder}/epoch-{ep}.pt")
        curve = _epoch_curve(state)
        json.dump(curve, open(f"{self.folder}/epochs.json", "w"))
        best_ep, best_f1 = max(curve, key=lambda x: x[1])
        print(f"época {ep}: tok_macro_f1 val {metrics.get('eval_tok_macro_f1'):.4f} (mejor: época {best_ep}, {best_f1:.4f})")
        why = _stop_reason(curve, self.patience, self.min_delta)
        if why:
            print(f"parada temprana: {why}")
            control.should_training_stop = True


def train_model(key, lr, seed, run_name, trainer_cls=None, ds_hook=None):
    # Igual que la versión de la sección 4, salvo los entrenamientos largos (ckpt_dir): evaluación por época,
    # parada temprana y uso de la mejor época.
    cfg = CANDIDATES[key]
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    t0 = time.time()
    ds_tr = Dataset.from_dict(ds_hook(train_rows, tok) if ds_hook else encode_rows(train_rows, tok))
    ds_va = Dataset.from_dict(encode_rows(val_rows, tok))
    print(f"[{run_name}] ventanas: train {len(ds_tr)}, val {len(ds_va)} (tokenizado en {time.time()-t0:.0f}s)")
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model"], num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID, **cfg.get("model_kwargs", {}))
    n_params, n_emb = count_params(model)
    vocab = int(model.get_input_embeddings().weight.shape[0])
    if "prepare" in cfg:
        model = cfg["prepare"](model, tok)
    out_dir = f"{WORK_DIR}/{run_name}"
    per_device = cfg.get("per_device", BATCH)
    epochs = cfg.get("epochs", EPOCHS)
    warmup = int(WARMUP * math.ceil(len(ds_tr) / BATCH) * epochs)
    keeper = None
    if cfg.get("ckpt_dir"):
        ckpt_dir = f"{cfg['ckpt_dir']}/{run_name}"
        # evaluar al final de cada época; guardar el estado completo cada 200 pasos (para continuar si se corta)
        schedule = dict(output_dir=ckpt_dir, eval_strategy="epoch", save_strategy="steps",
                        save_steps=cfg.get("save_steps", 200), save_total_limit=1)
        keeper = EpochKeeper(f"{ckpt_dir}/epochs", cfg.get("patience", 1), cfg.get("min_delta", 0.0))
    else:
        ckpt_dir = None
        schedule = dict(output_dir=out_dir, eval_strategy="epoch", save_strategy="epoch",
                        load_best_model_at_end=True, metric_for_best_model="tok_macro_f1", greater_is_better=True,
                        save_total_limit=1)
    args = TrainingArguments(
        per_device_train_batch_size=per_device, gradient_accumulation_steps=BATCH // per_device,
        per_device_eval_batch_size=cfg.get("eval_batch", 64),
        learning_rate=lr, num_train_epochs=epochs, weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup, lr_scheduler_type="linear", bf16=True,
        gradient_checkpointing=cfg.get("grad_ckpt", False), remove_unused_columns=ds_hook is None,
        logging_steps=50, report_to="none", seed=seed, dataloader_num_workers=2, **schedule, **GROUP_BY_LENGTH,
        **({"data_seed": seed} if "data_seed" in _TA_PARAMS else {}))
    trainer = (trainer_cls or Trainer)(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                                       data_collator=DataCollatorForTokenClassification(tok), processing_class=tok,
                                       compute_metrics=compute_metrics, callbacks=[keeper] if keeper else None)
    if keeper:
        keeper.trainer = trainer
    resume = None
    if ckpt_dir and os.path.isdir(ckpt_dir):
        found = sorted(Path(ckpt_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        # un punto de control a medias (p. ej. Drive lleno o el entorno cortado mientras se escribía) no sirve:
        # se usa el último completo (adaptadores, optimizador y estado del Trainer)
        need = ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json")
        complete = [p for p in found if all((p / f).is_file() and (p / f).stat().st_size > 0 for f in need)]
        for p in found:
            if p not in complete:
                print(f"[{run_name}] punto de control incompleto, se ignora: {p.name}")
        resume = str(complete[-1]) if complete else None
        if resume:
            print(f"[{run_name}] se continúa desde {resume}")
        elif found:
            print(f"[{run_name}] ningún punto de control completo: se empieza desde el principio")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    minutes = (time.time() - t0) / 60
    print(f"[{run_name}] entrenado en {minutes:.1f} min (en esta sesión)")
    json.dump(trainer.state.log_history, open(f"{RUNS_DIR}/logs/{run_name}.json", "w"))
    best_epoch = None
    if keeper:
        curve = _epoch_curve(trainer.state)
        best_epoch, best_f1 = max(curve, key=lambda x: x[1])
        print(f"[{run_name}] curva en validación por época: {curve} -> se usa la época {best_epoch}")
        set_peft_model_state_dict(trainer.model, torch.load(f"{keeper.folder}/epoch-{best_epoch}.pt"))
        json.dump(curve, open(f"{RUNS_DIR}/logs/{run_name}-epochs.json", "w"))
    elif trainer.state.best_model_checkpoint:
        step = int(trainer.state.best_model_checkpoint.rstrip("/").split("-")[-1])
        best_epoch = next((h["epoch"] for h in trainer.state.log_history if h.get("step") == step and "eval_loss" in h), None)
    final = f"{out_dir}/final"
    if hasattr(trainer.model, "merge_and_unload"):
        trainer.model.merge_and_unload().save_pretrained(final)
    else:
        trainer.save_model(final)
    tok.save_pretrained(final)
    if ckpt_dir:                                   # terminado y guardado: los puntos de control de Drive sobran
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    info = {"model": key, "lr": lr, "seed": seed, "train_min": round(minutes, 2), "best_epoch": best_epoch,
            "params_m": round(n_params / 1e6, 1), "emb_params_m": round(n_emb / 1e6, 1),
            "vocab": vocab, "train_windows": len(ds_tr)}
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
    return final, info


print("Qwen: hasta", CANDIDATES[TEACHER_B]["epochs"], "épocas, evaluación por época y parada temprana "
      f"(paciencia 1, mejora mínima {CANDIDATES[TEACHER_B]['min_delta']})")
