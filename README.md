# Comparativa de arquitecturas ligeras para la anonimización contextual de texto bilingüe en el borde

Código, datos y resultados de la parte experimental del Trabajo Fin de Máster *Comparativa de arquitecturas ligeras de
aprendizaje profundo para la anonimización contextual de texto bilingüe (español-inglés) en el borde*
(Máster Universitario en Inteligencia Artificial, UNIR, 2026). Autor: Alejandro Fischer González.

## Contenido
- `data/corpus_pii_es_en_v3.zip` — corpus bilingüe (train, val y cuatro conjuntos de prueba), guía de anotación y
  licencias (ver su `README.md`).
- `notebook/build_tfm_notebook.py` → `notebook/tfm_comparativa.ipynb` (Google Colab): entrenamiento de los cuatro
  codificadores con el mismo protocolo (búsqueda de learning rate, 3 semillas), evaluación, exportación a ONNX
  (FP32 / FP16 / INT8) y medidas de eficiencia.
- `notebook/build_distill_notebook.py` → `notebook/tfm_destilacion.ipynb`: profesores (XLM-R large y Qwen2.5-7B con
  LoRA y atención bidireccional), selección de profesores en validación y alumnos destilados (mmBERT-small).
  `patch_qwen_epochs.py` y `recover_qwen.py`: evaluación por época, parada temprana y continuación tras cortes.
- `notebook/build_rerun_mmbert.py` → repetición de dos semillas de mmBERT-small con comprobación de la evaluación.
- `notebook/smoke*.py` — pruebas locales en CPU de todo el código de los cuadernos.
- `analysis/` — líneas base (reglas y Presidio), eficiencia en un portátil (`edge_bench.py`), post-proceso
  (`postproc_eval.py`), F1 por token y por carácter (`extra_metrics.py`, `extra_teachers.py`), métricas por tipo de
  entidad (`per_label.py`), análisis exploratorio (`eda.py`), prueba bootstrap (`bootstrap.py`), intervalos por tipo (`per_label_ci.py`) y tablas (`tables.py`).
- `lib/postprocess.py` — reglas deterministas y post-proceso.
- `results/` — resultados de todas las ejecuciones (JSONL/CSV) y probabilidades por token de los conjuntos de prueba,
  a partir de los cuales se generan las tablas y figuras de la memoria.

## Reproducir

### Entorno
- **Entrenamiento:** Google Colab con GPU NVIDIA L4 (24 GB) o A100, entre octubre y noviembre de 2026. Los cuadernos
  instalan `transformers>=4.48`, `onnx`, `onnxruntime` y `peft` sobre el entorno de Colab de esas fechas; las versiones
  exactas del entorno de Colab no se registraron.
- **Modelos de partida (Hugging Face):** `distilbert/distilbert-base-multilingual-cased`,
  `microsoft/Multilingual-MiniLM-L12-H384`, `jhu-clsp/mmBERT-small`, `FacebookAI/xlm-roberta-base`,
  `FacebookAI/xlm-roberta-large` y `Qwen/Qwen2.5-7B`.
- **Semillas:** 42 (búsqueda de la tasa de aprendizaje), 13 y 2026. Bootstrap: semilla 2026, 10.000 remuestreos.
- **Análisis y eficiencia:** Python 3.11 en un portátil con Intel Core i7-8559U (sin GPU); versiones en
  `requirements.txt`.

### Orden de ejecución
1. Subir `data/corpus_pii_es_en_v3.zip` a Google Drive, en `MyDrive/tfm_pii/`.
2. `notebook/tfm_comparativa.ipynb` (generado por `build_tfm_notebook.py`): los cuatro codificadores. Si se corta la
   sesión, se vuelve a ejecutar y lo terminado se salta.
3. `notebook/tfm_rerun_mmbert.ipynb` (`build_rerun_mmbert.py`): repite las semillas 13 y 2026 de mmBERT-small, cuya
   evaluación falló en la primera ejecución (ver la cabecera del script).
4. `notebook/tfm_destilacion.ipynb` (`build_distill_notebook.py`): profesores y alumnos. Durante la ejecución real se
   aplicaron, dentro del mismo entorno y en este orden, `fix_kd.py` (corrige las etiquetas blandas de los tokens de
   espacio; los alumnos válidos son `kdA2` y `kdAB2`), `patch_qwen_epochs.py` (Qwen con evaluación por época y parada
   temprana) y `recover_qwen.py` (evalúa el Qwen guardado sin reentrenar). Cada script explica en su cabecera qué
   corrige; se conservan para que el proceso sea trazable.
5. Análisis local con el corpus descomprimido (los resultados ya están en `results/`).

### Qué genera cada tabla de la memoria
| Tablas de la memoria | Script | Resultado |
|---|---|---|
| Corpus y análisis exploratorio (7, 8, fig. 2) | `analysis/eda.py` | `results/eda.json`, `results/dataset_stats.csv` |
| Entrenamiento y eficacia por semilla (9–11, fig. 3) | cuadernos de Colab + `analysis/tables.py` | `results/colab/` |
| F1 estricto, por token y por carácter | `analysis/extra_metrics.py`, `analysis/extra_teachers.py` | `results/extra_metrics*.jsonl` |
| Por tipo de entidad (12–14, fig. 4) | `analysis/per_label.py` | `results/per_label_test_v2.json` |
| Intervalos por tipo de entidad (15) | `analysis/per_label_ci.py` | `results/per_label_ci_test_v2.json` |
| Líneas base | `analysis/baselines.py` | `results/baselines.jsonl` |
| Eficiencia en el portátil (17, 18) | `analysis/edge_bench.py` | `results/edge_mac.jsonl` |
| Post-proceso (fig. 5) | `analysis/postproc_eval.py` | `results/postproc.jsonl` |
| Profesores y alumnos (19–21) | cuaderno de destilación + `analysis/extra_teachers.py` | `results/colab/`, `results/extra_metrics_teachers.jsonl` |
| Prueba bootstrap (23) | `analysis/bootstrap.py` | `results/bootstrap_test_v2.json` |

La versión evaluada en la memoria es la etiqueta `v1.0-entrega2`.

## Licencias
- Código: MIT (ver `LICENSE`).
- Corpus: CC BY 4.0, con las atribuciones de ai4privacy OpenPII-1.5M (CC BY 4.0) y OpenAssistant oasst2 (Apache-2.0)
  indicadas en su `README.md`.
