"""Deterministic scoring functions for answer- and agent-level evaluation.

Every function here is pure, offline and provider-agnostic. Two check types
are strictly separated:

* **deterministic** — exact set/string operations whose result is always
  correct for what they measure (term coverage, citation presence,
  fabricated-citation detection, tool-selection scoring);
* **heuristic** — the numeric-claim support check. It extracts digit-form
  quantities from the answer and verifies each one (number + magnitude,
  units intentionally ignored) appears in the evidence. It can produce both
  false positives (same number, different meaning; spelled-out numbers are
  not extracted at all) and false negatives (rounding, re-scaled units such
  as raw "391035000000" vs "391 billion"). It is a signal, not a proof of
  faithfulness — an optional LLM-as-judge can be layered on top, but is
  never required for the offline suite.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

_WS_RE = re.compile(r"\s+")

# A quantity: digits (optionally comma-grouped / decimal) plus an optional
# magnitude word. Unit suffixes are consumed but NOT compared (see module
# docstring): "%", "percent", "dollars", "USD", "per share", ... all map to
# no unit. The magnitude word (million/billion/trillion) IS compared.
_NUMBER_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*"
    r"(million|billion|trillion)?\s*"
    r"(?:%|percent|per\s+share|euros?|dollars?|usd|eur|tonnes?|years?|days?)?\b",
    re.IGNORECASE,
)


def normalize_text(text: Any) -> str:
    """Whitespace-collapsed lowercase used for all containment checks."""
    return _WS_RE.sub(" ", str(text or "").lower()).strip()


def term_coverage(answer: str, terms: Sequence[str]) -> Dict[str, Any]:
    """Deterministic: which expected facts/terms appear in the answer."""
    haystack = normalize_text(answer)
    covered = [t for t in terms if normalize_text(t) in haystack]
    missing = [t for t in terms if normalize_text(t) not in haystack]
    total = len(terms)
    return {
        "total": total,
        "covered": covered,
        "missing": missing,
        "coverage": (len(covered) / total) if total else 1.0,
    }


def extract_number_tokens(text: str) -> List[str]:
    """Canonical ``number+magnitude`` tokens for every quantity in ``text``.

    "412 million dollars" -> "412million"; "$1.85" -> "1.85"; "0.87%" -> "0.87".
    """
    tokens = []
    for match in _NUMBER_RE.finditer(normalize_text(text)):
        number = match.group(1).replace(",", "")
        if "." in number:
            # "238.10" and its JSON float repr "238.1" must compare equal.
            number = number.rstrip("0").rstrip(".")
        magnitude = (match.group(2) or "").lower()
        tokens.append(f"{number}{magnitude}")
    return tokens


def numeric_claim_support(answer: str, evidence_text: str) -> Dict[str, Any]:
    """Heuristic: answer quantities that do not appear in the evidence."""
    claims = extract_number_tokens(answer)
    evidence = set(extract_number_tokens(evidence_text))
    unsupported = sorted({c for c in claims if c not in evidence})
    unique_claims = set(claims)
    return {
        "claims": len(claims),
        "unsupported": unsupported,
        "unsupported_count": len(unsupported),
        # Rate is per unique unsupported claim type over unique claims made.
        "unsupported_rate": (len(unsupported) / len(unique_claims)) if unique_claims else 0.0,
    }


def citation_checks(
    answer_citations: Sequence[str],
    *,
    expected_sources: Sequence[str],
    evidence_sources: Iterable[str],
    min_citations: int,
) -> Dict[str, Any]:
    """Citation presence and correctness.

    ``fabricated`` is deterministic: a cited source that was never in the
    retrieved/tool evidence could not have grounded the answer.
    """
    cited = list(answer_citations)
    available = set(evidence_sources)
    fabricated = sorted({c for c in cited if c not in available})
    expected_missing = [s for s in expected_sources if s not in cited]
    if cited:
        correctness = (len(cited) - len(fabricated)) / len(cited)
    else:
        correctness = 1.0 if not expected_sources else 0.0
    return {
        "cited": cited,
        "cited_count": len(cited),
        "min_required": min_citations,
        "presence": len(cited) >= min_citations,
        "fabricated": fabricated,
        "expected_missing": expected_missing,
        "citation_correctness": correctness,
    }


def _call_attr(call: Any, key: str) -> Any:
    if isinstance(call, dict):
        return call.get(key)
    return getattr(call, key, None)


def score_tools(
    expected_tools: Sequence[str],
    tools_called: Sequence[Any],
) -> Dict[str, Any]:
    """Deterministic tool-selection scoring (multiset comparison).

    * ``missing`` — expected tools never called;
    * ``unexpected`` — calls of tools that were never expected;
    * ``unnecessary`` — extra invocations of expected tools beyond the
      expected multiplicity (e.g. the same lookup twice).
    """
    expected = list(expected_tools)
    called_names = [_call_attr(c, "name") for c in tools_called]
    ok_flags = [bool(_call_attr(c, "ok")) for c in tools_called]

    missing: List[str] = []
    for name in set(expected):
        shortfall = expected.count(name) - called_names.count(name)
        missing.extend([name] * shortfall)

    unexpected_calls = [n for n in called_names if n not in expected]
    unnecessary_calls: List[str] = []
    for name in set(called_names) & set(expected):
        extras = called_names.count(name) - expected.count(name)
        unnecessary_calls.extend([name] * max(0, extras))

    executed = len(ok_flags)
    succeeded = sum(ok_flags)
    return {
        "expected": expected,
        "called": called_names,
        "total_calls": executed,
        "missing": sorted(missing),
        "unexpected": unexpected_calls,
        "unnecessary": sorted(unnecessary_calls),
        "unexpected_count": len(unexpected_calls),
        "unnecessary_count": len(unnecessary_calls),
        "selection_correct": not missing and not unexpected_calls and not unnecessary_calls,
        "execution_success_rate": (succeeded / executed) if executed else 1.0,
    }


def detector_flags(
    terms: Dict[str, Any],
    citations: Dict[str, Any],
    numeric: Dict[str, Any],
    tools: Dict[str, Any],
) -> List[str]:
    """Machine-readable flags raised by the deterministic/heuristic checks."""
    flags = []
    if terms["missing"]:
        flags.append("missing_facts")
    if not citations["presence"]:
        flags.append("missing_citation")
    if citations["fabricated"]:
        flags.append("fabricated_citation")
    if citations["expected_missing"]:
        flags.append("missing_expected_source")
    if numeric["unsupported_count"]:
        flags.append("unsupported_numbers")
    if tools["missing"]:
        flags.append("missing_expected_tool")
    if tools["unexpected_count"]:
        flags.append("unexpected_tool")
    if tools["unnecessary_count"]:
        flags.append("unnecessary_tool")
    if tools["execution_success_rate"] < 1.0:
        flags.append("tool_execution_failure")
    return flags


def is_grounded(
    terms: Dict[str, Any],
    citations: Dict[str, Any],
    numeric: Dict[str, Any],
) -> bool:
    """Grounded = full fact coverage + required, unfabricated citations + no
    unsupported numeric claims. Tool behaviour is scored separately."""
    return (
        terms["coverage"] == 1.0
        and citations["presence"]
        and citations["citation_correctness"] == 1.0
        and not citations["expected_missing"]
        and numeric["unsupported_count"] == 0
    )
