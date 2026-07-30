"""
Entity extraction from free text.

Shared by every agent that ends up holding a string — OCR, chat screenshots,
documents, audio transcripts. One implementation so a phone number found in a
chat and the same number found in a document produce the identical entity id and
correlate for free.

Two things shape the design:

**OCR output is dirty.** RapidOCR routinely drops spaces
("inthatwhitecarKL07AB1234") and confuses O/0 and I/1 in registration plates.
The patterns are written to survive that, and plates are normalised before an
entity id is derived so "KL 07 AB 1234", "KL-07-AB-1234" and "KL07AB1234" all
converge on one vehicle.

**A false entity is worse than a missed one.** Every extracted entity becomes a
node the planner may reason over and potentially escalate. The patterns are
therefore strict — anchored, length-checked, and validated where the format
allows it — and it is better to miss an unusual phone number than to invent one
from a date.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Entity, EntityType

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

#: Indian registration plates: two letters (state), 1-2 digits (district),
#: 1-3 letters (series), four digits. Separators optional because OCR eats them.
#:
#: There is deliberately NO word boundary at the start. RapidOCR routinely runs
#: words together — a real observed output was "inthatwhitecarKL07AB1234" — and a
#: leading \b makes the plate unfindable in exactly the case that matters. The
#: trailing \b plus the state-code check below carry the false-positive load.
PLATE_RE = re.compile(
    r"([A-Z]{2})[\s\-.]?(\d{1,2})[\s\-.]?([A-Z]{1,3})[\s\-.]?(\d{4})\b"
)

#: Indian mobile numbers: optional +91/0 prefix, then 10 digits starting 6-9.
PHONE_RE = re.compile(r"(?:(?:\+|00)91[\s\-]?|\b0)?([6-9]\d{9})\b")

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

URL_RE = re.compile(r"\bhttps?://[^\s<>\"']+|\bwww\.[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

#: Social handles. Requires a letter somewhere so "@1234" is not a handle.
HANDLE_RE = re.compile(r"(?<![A-Za-z0-9._])@([A-Za-z][A-Za-z0-9._]{2,29})\b")

#: Valid Indian state/UT codes. Checking against this is what stops random
#: uppercase pairs in OCR noise becoming vehicles.
_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP",
    "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL",
    "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}


@dataclass
class Extraction:
    """One entity found in text, with the snippet it came from."""

    entity: Entity
    matched_text: str
    context: str

    def describe(self) -> str:
        return f"{self.entity.type.value}: {self.entity.value}"


def _context(text: str, start: int, end: int, window: int = 40) -> str:
    """Surrounding text, so a claim can show where a value came from."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].replace("\n", " ").strip()
    return f"…{snippet}…" if left > 0 or right < len(text) else snippet


def extract_plates(text: str) -> list[Extraction]:
    """Vehicle registration numbers, normalised to a canonical form."""
    found: list[Extraction] = []
    seen: set[str] = set()

    for match in PLATE_RE.finditer(text.upper()):
        state, district, series, number = match.groups()
        if state not in _STATE_CODES:
            continue

        canonical = f"{state}{int(district):02d}{series}{number}"
        if canonical in seen:
            continue
        seen.add(canonical)

        found.append(
            Extraction(
                entity=Entity(
                    type=EntityType.PLATE,
                    value=canonical,
                    attributes={
                        "state_code": state,
                        "district": district,
                        "series": series,
                        "number": number,
                        "as_written": match.group(0),
                    },
                ),
                matched_text=match.group(0),
                context=_context(text, match.start(), match.end()),
            )
        )
    return found


def extract_phones(text: str) -> list[Extraction]:
    found: list[Extraction] = []
    seen: set[str] = set()

    # Collapse spacing inside digit runs first — OCR splits numbers arbitrarily.
    compact = re.sub(r"(?<=\d)[\s\-](?=\d)", "", text)

    for match in PHONE_RE.finditer(compact):
        number = match.group(1)
        if number in seen:
            continue
        # A ten-digit run inside a longer digit string is usually an id, not a
        # phone. The check is against the start of the WHOLE match, not group 1 —
        # otherwise a recognised "+91" prefix looks like adjacent digits and the
        # number it introduces gets thrown away.
        before = compact[max(0, match.start(0) - 1) : match.start(0)]
        after = compact[match.end(1) : match.end(1) + 1]
        if before.isdigit() or after.isdigit():
            continue
        seen.add(number)

        found.append(
            Extraction(
                entity=Entity(
                    type=EntityType.PHONE,
                    value=f"+91{number}",
                    attributes={"national": number, "country": "IN"},
                ),
                matched_text=match.group(0),
                context=_context(compact, match.start(), match.end()),
            )
        )
    return found


def extract_emails(text: str) -> list[Extraction]:
    found, seen = [], set()
    for match in EMAIL_RE.finditer(text):
        value = match.group(0).lower()
        if value in seen:
            continue
        seen.add(value)
        found.append(
            Extraction(
                entity=Entity(type=EntityType.EMAIL, value=value),
                matched_text=match.group(0),
                context=_context(text, match.start(), match.end()),
            )
        )
    return found


def extract_handles(text: str) -> list[Extraction]:
    found, seen = [], set()
    for match in HANDLE_RE.finditer(text):
        value = "@" + match.group(1)
        if value.lower() in seen:
            continue
        seen.add(value.lower())
        found.append(
            Extraction(
                entity=Entity(type=EntityType.HANDLE, value=value),
                matched_text=match.group(0),
                context=_context(text, match.start(), match.end()),
            )
        )
    return found


def extract_urls(text: str) -> list[Extraction]:
    found, seen = [], set()
    for match in URL_RE.finditer(text):
        value = match.group(0).rstrip(".,;)")
        if value.lower() in seen:
            continue
        seen.add(value.lower())
        found.append(
            Extraction(
                entity=Entity(type=EntityType.HANDLE, value=value, attributes={"kind": "url"}),
                matched_text=match.group(0),
                context=_context(text, match.start(), match.end()),
            )
        )
    return found


def extract_all(text: str) -> list[Extraction]:
    """Every entity type, in one pass over the text."""
    if not text or not text.strip():
        return []
    return [
        *extract_plates(text),
        *extract_phones(text),
        *extract_emails(text),
        *extract_handles(text),
        *extract_urls(text),
    ]
