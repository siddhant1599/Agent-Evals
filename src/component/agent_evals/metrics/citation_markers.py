"""Removal of ALCE-style citation markers such as ``[1][2]`` from a generated answer (D-126).

On a dataset whose answers cite their passages (ASQA, D-118), every metric except the citation
metrics scores the answer with its markers removed, as ALCE's evaluation does before exact match,
ROUGE and claim checks (``eval.py``, ``main``). The rule is ALCE's ``remove_citations`` unchanged
(``utils.py``, ALCE repository commit ``246c476``, MIT licence), so scores match ALCE's own
normalization. Like the original, it also deletes every ``]`` and every `` |``.
"""

from __future__ import annotations

import re

__all__ = ["CITATION", "remove_citations"]

#: ALCE's citation pattern (``eval.py``): ``[`` and the passage number, counted from 1. A sentence
#: with no match cites nothing.
CITATION = re.compile(r"\[(\d+)")

_SPACED_MARKER = re.compile(r" \[\d+")
_MARKER = re.compile(r"\[\d+")


def remove_citations(text: str) -> str:
    """Remove citation markers the way ALCE's ``remove_citations`` does.

    Args:
        text: A generated answer.

    Returns:
        The answer with every ``[<digits>`` (and the space before it) removed, then every `` |``
        and every ``]``. Whitespace elsewhere is left as it was.
    """
    return _MARKER.sub("", _SPACED_MARKER.sub("", text)).replace(" |", "").replace("]", "")
