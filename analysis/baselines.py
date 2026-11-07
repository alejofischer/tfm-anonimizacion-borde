"""Líneas base sin aprendizaje profundo, evaluadas con las mismas métricas que los modelos.

  python analysis/baselines.py <carpeta dataset v3> rules      # solo reglas (regex + dígitos de control)
  python analysis/baselines.py <carpeta dataset v3> presidio   # Microsoft Presidio + spaCy *_lg (ES/EN)

Resultados en results/baselines.jsonl (se añaden). Latencia: mediana por mensaje en test_v2 (CPU, 1 proceso).

Presidio: idioma del analizador según la etiqueta de idioma del mensaje (es / en; mixto u otro → ambos y unión).
Es una ventaja para Presidio (en producción habría que detectar el idioma). Correspondencia de entidades con la
taxonomía del trabajo en PRESIDIO_MAP; las que no tienen equivalente (URL, IP, fechas genéricas…) se descartan,
igual que en las guías de anotación.
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import postprocess  # noqa: E402  (lib/postprocess.py)

OUT = common.ROOT / "results" / "baselines.jsonl"
PRESIDIO_MAP = {
    "PERSON": "PER", "LOCATION": "LOC", "NRP": "DEMO",
    "EMAIL_ADDRESS": "CONTACT", "PHONE_NUMBER": "CONTACT",
    "ES_NIF": "ID", "ES_NIE": "ID", "US_SSN": "ID", "US_PASSPORT": "ID", "US_DRIVER_LICENSE": "ID",
    "UK_NHS": "ID", "UK_NINO": "ID", "MEDICAL_LICENSE": "ID", "US_ITIN": "ID",
    "IBAN_CODE": "FIN", "CREDIT_CARD": "FIN", "US_BANK_NUMBER": "FIN", "CRYPTO": "FIN",
}


def merge_overlaps(spans):
    spans = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))
    out = []
    for s in spans:
        if out and s["start"] < out[-1]["end"]:
            if s["end"] - s["start"] > out[-1]["end"] - out[-1]["start"]:
                out[-1] = s
            continue
        out.append(s)
    return out


def rules_predict(text):
    return [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in merge_overlaps(postprocess.rule_spans(text))]


def make_presidio():
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "es", "model_name": "es_core_news_lg"}, {"lang_code": "en", "model_name": "en_core_web_lg"}]})
    engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["es", "en"])

    def predict(text, lang):
        langs = {"es": ["es"], "en": ["en"]}.get(lang, ["es", "en"])
        spans = []
        for lg in langs:
            for r in engine.analyze(text=text, language=lg):
                lab = PRESIDIO_MAP.get(r.entity_type)
                if lab:
                    spans.append({"start": r.start, "end": r.end, "label": lab})
        return merge_overlaps(spans)
    return predict


def main():
    data_dir, which = sys.argv[1], sys.argv[2]
    tests = common.load_tests(data_dir)
    if which == "rules":
        predict = lambda text, lang: rules_predict(text)  # noqa: E731
        name = "baseline-rules"
    else:
        predict = make_presidio()
        name = "baseline-presidio"
    records = []
    for set_name, rows in tests.items():
        times, preds = [], []
        for r in rows:
            t0 = time.perf_counter()
            preds.append(predict(r["text"], common.row_lang(r)))
            times.append((time.perf_counter() - t0) * 1000)
        extra = {"model": name, "fmt": "cpu", "ms_median": round(statistics.median(times), 2)}
        recs = common.evaluate_with_slices(name, set_name, rows, preds, extra)
        records += recs
        m = recs[0]
        print(f"{name:18s} {set_name:9s} F1={m['strict_F1']:.3f} macroF1={m['macro_F1']:.3f} "
              f"fuga={m['leak_rate']:.3f} FA={m['false_alarm']:.3f} ms={extra['ms_median']}")
    OUT.parent.mkdir(exist_ok=True)
    common.append_results(OUT, records)


if __name__ == "__main__":
    main()
