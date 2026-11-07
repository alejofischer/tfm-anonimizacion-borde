"""Post-processing applied AFTER the model. Order matters and is fixed:

  1. decode   — token probabilities -> spans (optionally masking more readily: P(O) + P(PUB) < threshold)
  2. expand   — grow each model span to whole words, merge fragments ("Fis" "0" "5" -> "Fisio-Sur-2025")
  2b. pub     — (v3 models) a PUB span (public figure as a topic) stays unmasked only if its name is in the
                public-figures dictionary; otherwise, or when the customer masks public figures, it becomes PER
  3. rules    — deterministic patterns, applied ONLY to text no model span covers
  (4. names   — dictionary, same principle; added later)

PUB spans are returned with label "PUB" for information; callers must never mask them (see predict.mask).
Models without PUB (v2) behave exactly as before.

The model always wins: later steps never relabel or shrink an earlier span, they only fill gaps.
Spans are dicts {"start", "end", "label"} over character offsets, like the dataset.
Written as plain functions on strings so it can be ported to the extension's JavaScript as-is.
"""
import re
import unicodedata

NON_PII = {"PUB"}   # labels that are returned but never masked

# ---------------------------------------------------------------- 1. decode


def decode(text, tokens, id2label, threshold=None):
    """tokens: [(start, end, probs)]. threshold=None -> plain argmax (the notebook's behaviour).
    threshold=t -> a token is PII when P(O) + P(PUB) < t, labelled with its best personal-data class;
    otherwise it is PUB if PUB beats O, else O. Without PUB labels: P(O) < t, as in v2."""
    o_id = next(i for i, l in id2label.items() if l == "O")
    pub_ids = {i for i, l in id2label.items() if l.split("-", 1)[-1] in NON_PII and l != "O"}
    spans, cur = [], None
    for s, e, p in tokens:
        if threshold is None:
            lid = int(p.argmax())
        else:
            keep = p[o_id] + sum(p[i] for i in pub_ids)   # probability that the token needs no masking
            if keep < threshold:
                lid = max((v, i) for i, v in enumerate(p) if i != o_id and i not in pub_ids)[1]
            elif pub_ids and max(p[i] for i in pub_ids) > p[o_id]:
                lid = max((p[i], i) for i in pub_ids)[1]
            else:
                lid = o_id
        lab = id2label[lid]
        if lab == "O":
            if cur:
                spans.append(cur)
            cur = None
            continue
        bio, typ = lab.split("-", 1)
        # continue the current span for I- of the same type, or for any same-type token that touches it
        if cur and typ == cur["label"] and (bio == "I" or s <= cur["end"]):
            cur["end"] = e
        else:
            if cur:
                spans.append(cur)
            cur = {"start": s, "end": e, "label": typ}
    if cur:
        spans.append(cur)
    return [t for t in (_trim(text, sp) for sp in spans) if t]


_LEAD, _TAIL = "([{¿¡\"'«", ".,;:)]}\"'»"


def _trim(text, sp):
    s, e = sp["start"], sp["end"]
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    if sp["label"] != "CON":
        while e > s and text[e - 1] in _TAIL and not (text[e - 1] == ")" and "(" in text[s:e]):
            e -= 1
        while s < e and text[s] in _LEAD and not (text[s] == "(" and ")" in text[s:e]):
            s += 1
    return {**sp, "start": s, "end": e} if e > s else None


# ---------------------------------------------------------------- 2. expand

# characters that end a word when growing a span; CON values may contain most punctuation,
# so for CON only whitespace, quotes and key/value separators stop the growth
_WORD_STOP = set(" \t\n\r,;:()[]{}\"'«»¿¡?!/")
_CON_STOP = set(" \t\n\r\"'«»`=,;()[]{}")
_SENT_END = ".,;:!?)]}\"'»"


def expand(text, spans):
    out = []
    for sp in spans:
        stop = _CON_STOP if sp["label"] == "CON" else _WORD_STOP
        s, e = sp["start"], sp["end"]
        while s > 0 and text[s - 1] not in stop:
            s -= 1
        while e < len(text) and text[e] not in stop:
            e += 1
        # don't swallow sentence punctuation at the very end ("...es Lunes.2025.Madrid. Gracias")
        while e > s + 1 and text[e - 1] in _SENT_END and (e == len(text) or text[e].isspace()):
            e -= 1
        out.append({**sp, "start": s, "end": e})
    return merge(out)


def merge(spans):
    """Merge overlapping or touching spans; the merged label is the one covering most characters
    (a PUB part never wins over a personal-data part: the merged span is masked)."""
    spans = sorted(spans, key=lambda x: (x["start"], -x["end"]))
    out = []
    for sp in spans:
        if out and sp["start"] <= out[-1]["end"]:
            last = out[-1]
            cover = last.setdefault("_cover", {last["label"]: last["end"] - last["start"]})
            cover[sp["label"]] = cover.get(sp["label"], 0) + (sp["end"] - sp["start"])
            last["end"] = max(last["end"], sp["end"])
            pii = {k: v for k, v in cover.items() if k not in NON_PII}
            last["label"] = max(pii or cover, key=(pii or cover).get)
        else:
            out.append(dict(sp))
    for sp in out:
        sp.pop("_cover", None)
    return out


# ---------------------------------------------------------------- 3. rules


def _dni_ok(num, letter):
    return "TRWAGMYFPDXBNJZSQVHLCKE"[int(num) % 23] == letter.upper()


def _iban_ok(raw):
    s = re.sub(r"[\s-]", "", raw).upper()
    if not (15 <= len(s) <= 34):
        return False
    s = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in s)) % 97 == 1


def _luhn_ok(raw):
    d = [int(c) for c in raw if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    tot = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x = x * 2 - 9 if x > 4 else x * 2
        tot += x
    return tot % 10 == 0


_PLACEHOLDER = re.compile(
    r"^(?:[•*x·.#_-]{3,}|\$\{?\w+\}?|<[^>]*>|\{\{?[^}]*\}\}?|%\w+%|\[[^\]]*\]|(?:process\.env|os\.environ|env)\b.*"
    r"|.*(?:\.\.\.|…)|.*\b(?:your|tu|su|my|mi)_?\w*|.*(?:changeme|change_me|placeholder|example|ejemplo|redacted|xxxx).*)$", re.I)


def _password_like(v):
    # avoid masking ordinary words after "la clave es ...": require a digit, a symbol, or mixed case;
    # never mask placeholders (••••, ${DB_PASSWORD}, <YOUR_API_KEY>, process.env.X, changeme...)
    if len(v) < 4 or len(v) > 128 or v.endswith("=") or _PLACEHOLDER.match(v):
        return False
    return bool(re.search(r"\d", v) or re.search(r"[^\w\s]", v) or (re.search(r"[a-z]", v) and re.search(r"[A-Z].*[A-Z]|[a-z][A-Z]", v)))


_CUE = r"(?:contraseña|password|passwd|clave|pass|pwd|pin|código|codigo|code|wifi|wi-fi)"
_CUE_STRONG = r"(?:contraseña|password|passwd|clave|pwd)"   # can be followed by the value directly, with no verb
RULES = [
    # --- CON: well-known secret formats
    ("CON", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), 0),
    ("CON", re.compile(r"\b(?:sk|pk|rk)[-_](?:live|test)[-_][A-Za-z0-9]{12,}"), 0),
    ("CON", re.compile(r"\b[a-z]{2,10}_live_[A-Za-z0-9]{12,}"), 0),
    ("CON", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), 0),
    ("CON", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), 0),
    ("CON", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), 0),
    ("CON", re.compile(r"\beyJ[\w-]{8,}\.eyJ[\w-]{8,}\.[\w-]{8,}"), 0),
    ("CON", re.compile(r"\bAccountKey=([A-Za-z0-9+/=]{20,})"), 1),
    # --- CON: key=value in config / code, and credentials inside URLs
    ("CON", re.compile(r"(?i)\b\w*(?:password|passwd|pwd|secret|token|api_?key|access_?key|private_?key|auth_?key)\w*['\"]?[ \t]*[=:,][ \t]*['\"]?([^\s'\";,)]{4,})"), 1),
    ("CON", re.compile(r"\b[a-z][a-z0-9+.-]*://[^:/\s@]+:([^@\s/]+)@"), 1),
    # --- CON: "la clave es X" / "the password is X" / "pass: X" / "user / X"
    ("CON", re.compile(r"(?i)\b" + _CUE + r"\b(?:\s*[:=]|[^\n.:=]{0,40}?\b(?:es|is|son|are|era|was)\b:?)\s*[\"'«]?([^\s\"'»,;]+)"), 1),
    ("CON", re.compile(r"(?i)\b(?:admin|root|user(?:name)?|usuario|login|[\w.+-]+@[\w-]+\.[\w.]+)\s+/\s+([^\s,;]+)"), 1),
    # --- CON: "y la clave Reunion#2025" (cue word directly followed by a password-like value)
    ("CON", re.compile(r"(?i)\b" + _CUE_STRONG + r"\s+[\"'«]?([^\s\"'»,;]+)"), 1),
    # --- ID / FIN with checksums (only exact, valid values; the model handles broken ones)
    ("ID", re.compile(r"\b(\d{2}\.?\d{3}\.?\d{3})[\s-]?([A-Za-z])\b"), "dni"),
    ("ID", re.compile(r"\b([XYZxyz])[\s-]?(\d{7})[\s-]?([A-Za-z])\b"), "nie"),
    ("FIN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]{4}){3,7}(?:[ -]?[A-Z0-9]{1,4})?\b"), "iban"),
    ("FIN", re.compile(r"\b[2-6]\d{3}(?:[ -]?\d{4}){2}[ -]?\d{3,4}(?:[ -]?\d{3})?\b"), "card"),
    # --- CONTACT
    ("CONTACT", re.compile(r"(?<![:/\w.+-])[\w.+-]+@(?![\w.-]*\.(?:local|internal|lan|corp|localdomain)\b)[A-Za-z][\w-]*(?:\.[\w-]+)*\.[A-Za-z]{2,}\b(?!\.\w)"), 0),
]


def rule_spans(text):
    found = []
    for label, rx, grp in RULES:
        for m in rx.finditer(text):
            if grp == "dni":
                if not _dni_ok(m.group(1).replace(".", ""), m.group(2)):
                    continue
                s, e = m.span()
            elif grp == "nie":
                if not _dni_ok("XYZ".index(m.group(1).upper()).__str__() + m.group(2), m.group(3)):
                    continue
                s, e = m.span()
            elif grp == "iban":
                if not _iban_ok(m.group(0)):
                    continue
                s, e = m.span()
            elif grp == "card":
                if not _luhn_ok(m.group(0)):
                    continue
                s, e = m.span()
            else:
                s, e = m.span(grp)
                if label == "CONTACT" and re.search(r"\b(?:ssh|scp|sftp|git)\s+\S*$", text[max(0, s - 30):s]):
                    continue  # user@host in a shell command is a login, not an email
                if label == "CON" and grp == 1 and not _password_like(m.group(1).rstrip("?.")):  # "contraseña débil?"
                    continue
            # strip closing quotes/punctuation a value regex may have picked up
            while e > s and text[e - 1] in ".,;:)'\"»":
                e -= 1
            if e > s:
                found.append({"start": s, "end": e, "label": label, "src": "rule"})
    return found


def fill_gaps(spans, extra):
    """Add `extra` spans only where no existing span is: existing spans are never changed."""
    out = [dict(s) for s in spans]
    for x in sorted(extra, key=lambda x: (x["start"], -x["end"])):
        segs = [(x["start"], x["end"])]
        for sp in out:
            nxt = []
            for a, b in segs:
                if sp["end"] <= a or sp["start"] >= b:
                    nxt.append((a, b))
                    continue
                if a < sp["start"]:
                    nxt.append((a, sp["start"]))
                if sp["end"] < b:
                    nxt.append((sp["end"], b))
            segs = nxt
        for a, b in segs:
            if b - a >= 2:
                out.append({**x, "start": a, "end": b})
    return sorted(out, key=lambda x: x["start"])


# ---------------------------------------------------------------- 2b. public figures (v3)


def pub_norm(s):
    """Same normalisation as gen/build_pub_dict.py: lower case, no accents, dots/commas/hyphens -> space."""
    s = "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[.,'’\"()\-‐]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def pub_guard(text, spans, pub_names, mask_public=False):
    """A PUB span stays unmasked only if the model said PUB (context) AND the name is a known public figure.
    Otherwise — or when the customer masks public figures — it becomes PER. Both failures are safe."""
    out = []
    for sp in spans:
        if sp["label"] == "PUB":
            name = re.sub(r"(?:'s|’s)$", "", text[sp["start"]:sp["end"]])
            if mask_public or not pub_names or pub_norm(name) not in pub_names:
                sp = {**sp, "label": "PER", "src": "pub_guard"}
        out.append(sp)
    return out


def pipeline(text, tokens, id2label, threshold=None, do_expand=True, do_rules=True, pub_names=None, mask_public=False):
    spans = decode(text, tokens, id2label, threshold)
    if do_expand:
        spans = expand(text, spans)
    if any(sp["label"] == "PUB" for sp in spans):
        spans = pub_guard(text, spans, pub_names, mask_public)
    if do_rules:
        spans = fill_gaps(spans, rule_spans(text))
    return spans
