"""Deterministic content guards around the LLM.

`injection_flags` is OBSERVABILITY ONLY: it records suspicious instructions for logs and response
metadata. Security does not depend on it, because regexes are trivially bypassed. The actual
protection is that other customers' data never reaches the prompt (permissions.py).

`foreign_identifiers` is a last-line output check: an answer naming identifiers the caller was
never shown is withheld, whatever the model was talked into.
"""

import re
from collections.abc import Iterable

from app.config import Principal

_INJECTION_PATTERNS = {
    "ignore_instructions": r"\b(ignore|disregard|forget)\b.{0,40}\b(instructions|rules|prompt)\b",
    "role_override": r"(^|\n)\s*(system|admin|administrator|developer|assistant)\b[^\n:]{0,20}:",
    "persona_switch": r"\byou are now\b|\b(audit|developer|god) mode\b",
    "data_exfiltration": (
        r"\b(reveal|dump|export|include|list)\b.{0,60}\b(every|all|other)\s+customers?\b"
    ),
    "hidden_markup": r"<!--",
    "assistant_addressed": r"\bnote to the (assistant|ai|model)\b",
}
_COMPILED = {name: re.compile(p, re.I) for name, p in _INJECTION_PATTERNS.items()}
_CUSTOMER_ID_RE = re.compile(r"\bC-\d{4}\b")
_SOURCE_ID_RE = re.compile(r"\b(?:T|KB|LOG)-[0-9A-F]{4,8}\b")


def injection_flags(label: str, text: str, principal: Principal) -> list[str]:
    """Return labels like 'T-0009:role_override' for suspicious content in `text`."""
    flags = [f"{label}:{name}" for name, rx in _COMPILED.items() if rx.search(text)]
    if principal.role == "customer" and any(
        cid != principal.customer_id for cid in _CUSTOMER_ID_RE.findall(text)
    ):
        flags.append(f"{label}:foreign_customer_reference")
    return flags


def foreign_identifiers(
    answer_text: str, principal: Principal, shown_ids: Iterable[str]
) -> list[str]:
    """Identifiers in the answer that the caller does not own and was not shown."""
    shown = set(shown_ids)
    foreign = [
        cid
        for cid in _CUSTOMER_ID_RE.findall(answer_text)
        if principal.role == "customer" and cid != principal.customer_id
    ]
    foreign += [sid for sid in _SOURCE_ID_RE.findall(answer_text) if sid not in shown]
    return sorted(set(foreign))
