"""Format-aware prompt construction for the ORena SAVE FOCUS challenge.

The FOCUS Evaluator parses model output with *strict* per-format readers
(see ``focus.data.formats``):

* ``binary``      -> must be exactly "yes" / "no"
* ``number``      -> must be a bare non-negative integer ("4")
* ``percentage``  -> "96.67" or "96.67%"  (exact value compare, abs_tol=1e-9!)
* ``fo_class``    -> one or more registered FO names, comma-separated, or "none"
* ``time``        -> "hh:mm:ss"
* ``open_ended``  -> free text <=300 chars      (LLM judge)
* ``matching``    -> regex-validated text        (LLM judge)
* ``multiple_choice`` -> free text               (LLM judge)

For the exact-match formats (binary/number/percentage/fo_class/time) the model
MUST emit only the bare value or it scores 0. We therefore (a) inject a strong
format instruction into the prompt and (b) post-process the raw output to
recover the bare value when the model wraps it in prose.

This module is model-agnostic: it returns a system prompt, the user question
(augmented with a format hint), and a ``parse(raw) -> str`` cleaner keyed on the
answer format. The cleaner is best-effort — if it cannot confidently extract a
valid value it returns the stripped raw text so the Evaluator can decide.
"""

from __future__ import annotations

import re
from pathlib import Path

# FO class names accepted by focus.data.formats.FOClass. The static tuple is a
# FALLBACK; at import we parse the class headers out of the runtime-provided
# /input/FO_definitions.json when present (else the bundled txt), because the
# organizers ADD classes between phases (v0.3.4 added "Absorbable Hemostatic
# Agent" with the note that the final metadata uses it). Parsing the metadata
# keeps us correct for any future class without a code change.
FO_NAMES = (
    "Sponge", "Clip", "Specimen Bag", "Silicone Loop", "External Drain",
    "Needle", "Gallstone", "Specimen", "Mesh", "Absorbable Hemostatic Agent",
)


_FO_DEFINITIONS_PATH = Path(__file__).parent / "FO_definitions.txt"


def _parse_fo_names(defs_text: str) -> tuple[str, ...]:
    """Extract class names = underlined headers after 'Foreign Object Classes'."""
    tail = defs_text.split("Foreign Object Classes", 1)[-1]
    names = re.findall(r"^([A-Z][A-Za-z ]+?)\n-{3,}\s*$", tail, flags=re.M)
    return tuple(n.strip() for n in names)


def _load_fo_names() -> tuple[str, ...]:
    import json as _json
    for src in (Path("/input/FO_definitions.json"), _FO_DEFINITIONS_PATH):
        try:
            text = src.read_text()
            if src.suffix == ".json":
                text = _json.loads(text)  # the json holds one plain-text string
            names = _parse_fo_names(text)
            if len(names) >= 9:  # sanity: never accept a short/garbled parse
                return names
        except Exception:
            continue
    return FO_NAMES


FO_NAMES = _load_fo_names()
_FO_LOWER = {n.lower(): n for n in FO_NAMES}

# Verified surgical-safety knowledge (RSI retention, electrosurgery, drains, etc.)
# for the knowledge-bound capability buckets 4b/5a/5b. Facts web-verified vs
# AORN/ASCRS sources. Injected only when include_knowledge=True (A/B gated).
_KNOWLEDGE_PATH = Path(__file__).parent / "surgical_knowledge.txt"


def fo_definitions() -> str:
    try:
        return _FO_DEFINITIONS_PATH.read_text()
    except OSError:
        return ""


def surgical_knowledge() -> str:
    try:
        return _KNOWLEDGE_PATH.read_text()
    except OSError:
        return ""


BASE_SYSTEM = (
    "You are a meticulous surgical AI assistant. You are shown endoscopic "
    "imagery from a minimally invasive procedure and must answer a question "
    "about foreign objects (objects fully introduced into the body that must "
    "be accounted for). Reason carefully from the visual evidence, then give "
    "ONLY the final answer in the exact requested format with no explanation, "
    "no units, and no extra words."
)


# ── per-format instructions appended to the question ──────────────────────

_FORMAT_HINT = {
    "binary": "Answer with exactly one word: yes or no.",
    "number": "Answer with a single non-negative integer (digits only), e.g. 3.",
    "percentage": (
        "Answer with a percentage number only, e.g. 42 or 42.5 (no % sign needed)."
    ),
    "fo_class": (
        "Answer with the foreign-object class name(s) only, comma-separated if "
        "more than one, chosen from this exact list: "
        + ", ".join(FO_NAMES)
        + ". If none are present, answer exactly: none."
    ),
    "time": "Answer with a timestamp in hh:mm:ss format only, e.g. 00:01:23.",
    "open_ended": "Answer concisely in a few words (max 300 characters).",
    "matching": "Answer concisely with only the requested value.",
    "multiple_choice": (
        "Answer with exactly one of the provided options, copied verbatim, and "
        "nothing else."
    ),
}


def build_question(question: str, answer_format: str) -> str:
    """Augment the raw question with a strict format instruction."""
    hint = _FORMAT_HINT.get(answer_format, "")
    # Multi-event time questions ("At which time points ... chronologically")
    # legitimately have 1..N gold timestamps; the evaluator (>=0.3.3) compares
    # comma-lists count-matched. The default single-timestamp hint caps these
    # at one answer, so swap in a list-aware hint for that template only.
    if answer_format == "time" and "time points" in question.lower():
        hint = ("List every distinct time point at which this happens, in "
                "chronological order, comma-separated, each in hh:mm:ss format, "
                "e.g. 00:01:23, 00:04:56. If it happens only once, give just "
                "that one timestamp.")
    if hint:
        return f"{question.strip()}\n\n{hint}"
    return question.strip()


# Self-gating guard for the knowledge card (Option A). Lets the model decide when
# the card is relevant, instead of us routing on answer_format. Goal: keep the card's
# gains on purpose/complication/retention questions while avoiding the temporal-ordering
# distraction it caused when always-on.
_KNOWLEDGE_GUARD = (
    "The following SURGICAL-SAFETY KNOWLEDGE is relevant ONLY when the question asks "
    "about the PURPOSE or function of an object, WHY it is used, a COMPLICATION or "
    "consequence, or WHETHER an object normally remains in the body. If the question "
    "instead asks about TIMING, DURATION, COUNTING, ORDER of events, or LOCATION, "
    "IGNORE this section entirely and answer only from the video evidence.\n\n"
)


def system_prompt(include_fo_defs: bool = True, include_knowledge: bool = False,
                  knowledge_guard: bool = False) -> str:
    parts = [BASE_SYSTEM]
    if include_fo_defs:
        defs = fo_definitions()
        if defs:
            parts.append(defs)
    if include_knowledge:
        kb = surgical_knowledge()
        if kb:
            parts.append((_KNOWLEDGE_GUARD + kb) if knowledge_guard else kb)
    return "\n\n".join(parts)


# ── output cleaning / extraction ──────────────────────────────────────────

_INT_RE = re.compile(r"-?\d+")
_FLOAT_RE = re.compile(r"\d+(?:\.\d+)?")
_TIME_RE = re.compile(r"\d{1,2}:\d{2}:\d{2}")


def _clean_binary(raw: str) -> str:
    t = raw.strip().lower()
    # direct
    if t in ("yes", "no"):
        return t
    # leading token
    m = re.match(r"\W*(yes|no)\b", t)
    if m:
        return m.group(1)
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"
    return raw.strip()


def _clean_number(raw: str) -> str:
    t = raw.strip()
    if t.isdigit():
        return t
    m = _INT_RE.search(t.replace(",", ""))
    if m:
        return str(abs(int(m.group())))
    # spelled-out small numbers
    words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "none": "0", "no": "0",
    }
    for w, d in words.items():
        if re.search(rf"\b{w}\b", t.lower()):
            return d
    return raw.strip()


def _clean_percentage(raw: str) -> str:
    t = raw.strip().rstrip("%").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return t
    m = _FLOAT_RE.search(raw)
    if m:
        return m.group()
    return raw.strip()


def _clean_time(raw: str) -> str:
    # Keep EVERY timestamp the model emits (evaluator >=0.3.3 accepts comma-
    # separated lists and compares count-matched, chronologically aligned,
    # each within tolerance). Truncating to the first match guarantees a
    # count-mismatch fail on multi-event questions. Single-time answers are
    # unaffected (one match -> one timestamp).
    matches = _TIME_RE.findall(raw)
    if matches:
        out = []
        for ts in matches:
            h, mn, s = ts.split(":")
            out.append(f"{int(h):02d}:{int(mn):02d}:{int(s):02d}")
        # de-dup preserving order (models sometimes repeat the same time)
        seen: set[str] = set()
        out = [t for t in out if not (t in seen or seen.add(t))]
        return ", ".join(out)
    return raw.strip()


_FO_NEGATION_RE = re.compile(
    r"\b(none|no|not any|n[o']t)\b.*\b(foreign\s+object|object|fo)s?\b|"
    r"\bnothing\b|\bnone\b",
    re.IGNORECASE,
)


def _clean_fo_class(raw: str) -> str:
    t = raw.strip()
    low = t.lower()
    if low in ("none", "no foreign object", "no foreign objects", "nothing"):
        return "none"
    # find all FO names mentioned (longest first so "Specimen Bag" beats "Specimen")
    found: list[str] = []
    scratch = low
    for name in sorted(FO_NAMES, key=len, reverse=True):
        if name.lower() in scratch:
            found.append(name)
            scratch = scratch.replace(name.lower(), " ")
    if found:
        # de-dup preserving order
        seen = set()
        uniq = [n for n in found if not (n in seen or seen.add(n))]
        return ", ".join(uniq)
    # no class name present — treat negation / "none" phrasing as the NONE answer
    if _FO_NEGATION_RE.search(low):
        return "none"
    return t


_CLEANERS = {
    "binary": _clean_binary,
    "number": _clean_number,
    "percentage": _clean_percentage,
    "time": _clean_time,
    "fo_class": _clean_fo_class,
}


def clean_answer(raw: str, answer_format: str) -> str:
    """Best-effort extraction of a format-valid bare answer from raw model text.

    For judge formats (open_ended/matching/multiple_choice) the raw text is
    returned stripped, since semantic judging is more forgiving.
    """
    if raw is None:
        return ""
    # Thinking models (GLM-4.1V-Thinking, etc.): the final answer follows the
    # last </think>. If present, keep only the post-think text. Also handle the
    # GLM <answer>...</answer> wrapper. If thinking is unclosed (truncated), the
    # answer never came — leave raw so it scores as a (correct) miss.
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    m = re.search(r"<answer>(.*?)</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        raw = m.group(1)
    # strip any remaining paired think tags / stray tags
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"</?(think|answer)>", "", raw, flags=re.IGNORECASE)
    raw = raw.strip()
    # strip a leading "Answer:" style prefix
    raw = re.sub(r"^\s*(final\s+)?answer\s*[:\-]\s*", "", raw, flags=re.IGNORECASE)
    fn = _CLEANERS.get(answer_format)
    if fn:
        return fn(raw)
    # judge formats: collapse whitespace, cap length
    cleaned = " ".join(raw.split())
    return cleaned[:300]
