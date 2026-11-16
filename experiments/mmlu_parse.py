"""Strict MMLU answer parser for the collection scripts.

Port of rules R1 - R3c in analysis/lib_parse_mmlu.R (the analysis re-parses every
response with the R version, which also has R4, option-text matching). Keep the two
in sync. Never guesses from prose: the old collectors took the first a-d letter
anywhere ("Answer: C" -> A).
"""
import re

_CLEAN = re.compile(r"\*\*|__|`|#+ ")
_R1 = re.compile(r"^[(\[]?([A-Da-d])[)\]]?[.:,]?$")
_R2 = re.compile(r"^[(\[]?(?:([A-D])(?:[)\].:,]\s|[)\]]?\s*\n)|([a-d])[)\].]\s)")
_R3 = re.compile(r"(?i:\banswer|\bcorrect (?:option|choice|letter))\s*(?i:is|would be|should be)?"
                 r"\s*:?\s*[(\[]?(?:([A-D])(?![A-Za-z])|([a-d])[).])")
_R3B = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-Da-d])\)?\s*\}")
_R3C = re.compile(r"\bis:?\s*\(?([A-D])\)?\.?\s*$")


def parse_answer(resp):
    """Return 'A'-'D' if the response commits to one letter, else None."""
    if not isinstance(resp, str) or not resp.strip():
        return None
    t = _CLEAN.sub("", resp.replace("\u0120", " ").replace("\u010a", "\n")).strip()
    m = _R1.match(t) or _R2.match(t) or _R3.search(t) or _R3B.search(t) or _R3C.search(t)
    if m is None:
        return None
    letter = next(g for g in m.groups() if g)
    return letter.upper()
