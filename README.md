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
  entidad (`per_label.py`), análisis exploratorio (`eda.py`), prueba bootstrap (`bootstrap.py`) y tablas (`tables.py`).
- `lib/postprocess.py` — reglas deterministas y post-proceso.
- `results/` — resultados de todas las ejecuciones (JSONL/CSV) y probabilidades por token de los conjuntos de prueba,
  a partir de los cuales se generan las tablas y figuras de la memoria.

## Reproducir
1. Subir `data/corpus_pii_es_en_v3.zip` a Google Drive, en `MyDrive/tfm_pii/`.
2. Abrir `notebook/tfm_comparativa.ipynb` en Colab (GPU A100 o L4) y ejecutar todo; después,
   `notebook/tfm_destilacion.ipynb`. Si se corta la sesión, se vuelve a ejecutar y lo terminado se salta.
3. Análisis local (sin GPU), con el corpus descomprimido:
   ```
   python analysis/extra_metrics.py <carpeta del corpus> <ejecuciones…>
   python analysis/per_label.py <carpeta del corpus>
   python analysis/bootstrap.py <carpeta del corpus>
   python analysis/eda.py <carpeta del corpus>
   ```

## Licencias
- Código: MIT (ver `LICENSE`).
- Corpus: CC BY 4.0, con las atribuciones de ai4privacy OpenPII-1.5M (CC BY 4.0) y OpenAssistant oasst2 (Apache-2.0)
  indicadas en su `README.md`.
