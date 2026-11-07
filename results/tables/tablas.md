# Tablas de resultados (generadas por analysis/tables.py)

### T1_eficacia_test_v2

| Modelo | Semillas | F1 estricto | Macro F1 | Fuga (%) | Falsas alarmas (%) |
|---|---|---|---|---|---|
| DistilBERT multilingüe | 3 | 0.866 ± 0.016 | 0.835 ± 0.016 | 8.1 ± 1.0 | 8.1 ± 2.3 |
| Multilingual MiniLM-L12 | 3 | 0.848 ± 0.021 | 0.813 ± 0.026 | 7.5 ± 1.7 | 9.0 ± 1.8 |
| mmBERT-small | 3 | 0.890 ± 0.011 | 0.865 ± 0.015 | 6.3 ± 0.5 | 6.0 ± 1.6 |
| XLM-R base (referencia) | 3 | 0.885 ± 0.009 | 0.856 ± 0.010 | 5.4 ± 0.2 | 9.5 ± 2.5 |
| Solo reglas | — | 0.240 | 0.321 | 78.6 | 0.0 |
| Presidio + spaCy | — | 0.514 | 0.383 | 45.4 | 20.0 |

### T1_eficacia_test

| Modelo | Semillas | F1 estricto | Macro F1 | Fuga (%) | Falsas alarmas (%) |
|---|---|---|---|---|---|
| DistilBERT multilingüe | 3 | 0.846 ± 0.003 | 0.811 ± 0.005 | 7.8 ± 0.8 | 2.1 ± 0.6 |
| Multilingual MiniLM-L12 | 3 | 0.853 ± 0.022 | 0.824 ± 0.028 | 6.5 ± 1.2 | 2.5 ± 1.0 |
| mmBERT-small | 3 | 0.877 ± 0.007 | 0.852 ± 0.009 | 4.2 ± 0.2 | 1.2 ± 0.2 |
| XLM-R base (referencia) | 3 | 0.883 ± 0.005 | 0.861 ± 0.005 | 4.4 ± 0.5 | 1.8 ± 0.4 |
| Solo reglas | — | 0.219 | 0.219 | 81.6 | 0.0 |
| Presidio + spaCy | — | 0.433 | 0.283 | 51.3 | 16.5 |

### T1_eficacia_test_long

| Modelo | Semillas | F1 estricto | Macro F1 | Fuga (%) | Falsas alarmas (%) |
|---|---|---|---|---|---|
| DistilBERT multilingüe | 3 | 0.769 ± 0.027 | 0.713 ± 0.035 | 12.6 ± 1.9 | — |
| Multilingual MiniLM-L12 | 3 | 0.844 ± 0.011 | 0.809 ± 0.017 | 8.0 ± 1.8 | — |
| mmBERT-small | 3 | 0.865 ± 0.006 | 0.835 ± 0.008 | 5.4 ± 0.4 | — |
| XLM-R base (referencia) | 3 | 0.879 ± 0.006 | 0.850 ± 0.010 | 5.4 ± 0.8 | — |
| Solo reglas | — | 0.222 | 0.224 | 81.7 | — |
| Presidio + spaCy | — | 0.365 | 0.268 | 45.5 | — |

### T1_eficacia_test_v3

| Modelo | Semillas | F1 estricto | Macro F1 | Fuga (%) | Falsas alarmas (%) |
|---|---|---|---|---|---|
| DistilBERT multilingüe | 3 | 0.808 ± 0.040 | 0.782 ± 0.009 | 9.9 ± 0.3 | 17.7 ± 9.6 |
| Multilingual MiniLM-L12 | 3 | 0.792 ± 0.013 | 0.780 ± 0.029 | 11.7 ± 2.1 | 13.6 ± 5.9 |
| mmBERT-small | 3 | 0.842 ± 0.001 | 0.845 ± 0.007 | 10.0 ± 1.4 | 2.7 ± 3.1 |
| XLM-R base (referencia) | 3 | 0.831 ± 0.015 | 0.722 ± 0.089 | 11.1 ± 1.7 | 8.2 ± 4.1 |
| Solo reglas | — | 0.098 | 0.241 | 88.3 | 0.0 |
| Presidio + spaCy | — | 0.347 | 0.376 | 50.0 | 71.4 |

### T2_f1_por_etiqueta_test_v2

| Modelo | PER | LOC | ID | FIN | DEMO | CON | CONTACT |
|---|---|---|---|---|---|---|---|
| DistilBERT multilingüe | 0.980 | 0.680 | 0.934 | 0.828 | 0.729 | 0.711 | 0.985 |
| Multilingual MiniLM-L12 | 0.968 | 0.732 | 0.875 | 0.760 | 0.718 | 0.699 | 0.936 |
| mmBERT-small | 0.985 | 0.721 | 0.923 | 0.795 | 0.783 | 0.860 | 0.991 |
| XLM-R base (referencia) | 0.985 | 0.725 | 0.903 | 0.828 | 0.789 | 0.821 | 0.941 |
| Solo reglas | 0.000 | 0.000 | 0.436 | 0.478 | 0.000 | 0.761 | 0.571 |
| Presidio + spaCy | 0.767 | 0.141 | 0.438 | 0.464 | 0.047 | 0.000 | 0.825 |

### T3_idioma_y_rotos_test_v2

| Modelo | Español | Inglés | Mixto | Valores rotos |
|---|---|---|---|---|
| DistilBERT multilingüe | 0.871 / 10.2 % | 0.872 / 6.1 % | 0.841 / 5.6 % | 0.942 / 2.4 % |
| Multilingual MiniLM-L12 | 0.858 / 9.1 % | 0.841 / 7.0 % | 0.836 / 3.9 % | 0.958 / 2.2 % |
| mmBERT-small | 0.885 / 8.0 % | 0.903 / 5.1 % | 0.881 / 3.7 % | 0.951 / 1.7 % |
| XLM-R base (referencia) | 0.889 / 6.9 % | 0.882 / 4.4 % | 0.879 / 2.6 % | 0.975 / 1.9 % |

### T4_onnx_calidad_test_v2

| Modelo | Formato | Tamaño (MB) | F1 estricto | Fuga (%) |
|---|---|---|---|---|
| DistilBERT multilingüe | PyTorch (bf16, GPU) | — | 0.884 | 7.5 |
| DistilBERT multilingüe | ONNX fp32 | 539.1 | 0.884 | 7.5 |
| DistilBERT multilingüe | ONNX fp16 | 269.6 | 0.884 | 7.5 |
| DistilBERT multilingüe | ONNX int8-matmul | 412.0 | 0.874 | 8.3 |
| DistilBERT multilingüe | ONNX int8-all | 135.1 | 0.862 | 8.6 |
| Multilingual MiniLM-L12 | PyTorch (bf16, GPU) | — | 0.872 | 6.2 |
| Multilingual MiniLM-L12 | ONNX fp32 | 470.3 | 0.872 | 6.1 |
| Multilingual MiniLM-L12 | ONNX fp16 | 235.3 | 0.848 | 7.3 |
| Multilingual MiniLM-L12 | ONNX int8-matmul | 406.9 | 0.847 | 7.2 |
| Multilingual MiniLM-L12 | ONNX int8-all | 118.1 | 0.839 | 7.4 |
| mmBERT-small | PyTorch (bf16, GPU) | — | 0.879 | 6.9 |
| mmBERT-small | ONNX fp32 | 563.1 | 0.884 | 5.5 |
| mmBERT-small | ONNX fp16 | 281.9 | 0.881 | 6.9 |
| mmBERT-small | ONNX int8-matmul | 437.0 | 0.840 | 10.7 |
| mmBERT-small | ONNX int8-all | 141.6 | 0.819 | 10.6 |
| XLM-R base (referencia) | PyTorch (bf16, GPU) | — | 0.889 | 5.4 |
| XLM-R base (referencia) | ONNX fp32 | 1110.1 | 0.867 | 6.4 |
| XLM-R base (referencia) | ONNX fp16 | 555.2 | 0.896 | 5.2 |
| XLM-R base (referencia) | ONNX int8-matmul | 855.9 | 0.874 | 7.3 |
| XLM-R base (referencia) | ONNX int8-all | 278.2 | 0.856 | 8.7 |

### T5_eficiencia_portatil

| Modelo | Precisión | Tamaño (MB) | RAM modelo (MB) | RAM pico (MB) | ms 200 car. (1 hilo) | ms 1000 car. (1 hilo) | ms 2000 car. (1 hilo) | ms 1000 car. (4 hilos) | F1 test_v2 |
|---|---|---|---|---|---|---|---|---|---|
| DistilBERT multilingüe | fp32 | 539.1 | 661.4 | 675.1 | 96.8 | 609.6 | 1436.8 | 143.6 | 0.8835 |

### T6_modelos_y_coste

| Modelo | Parámetros (M) | De ellos, embeddings (M) | Vocabulario | Learning rate elegido | Minutos por entrenamiento (L4) | Mejor época (s42/s13/s2026) |
|---|---|---|---|---|---|---|
| DistilBERT multilingüe | 134.7 | 91.8 | 119547 | 8e-05 | 8.6 | 4/4/3 |
| Multilingual MiniLM-L12 | 117.5 | 96.0 | 250037 | 0.0001 | 10.8 | 4/3/2 |
| mmBERT-small | 140.6 | 98.3 | 256000 | 3e-05 | 16.0 | 4/3/3 |
| XLM-R base (referencia) | 277.5 | 192.0 | 250002 | 2e-05 | 18.0 | 4/4/4 |
