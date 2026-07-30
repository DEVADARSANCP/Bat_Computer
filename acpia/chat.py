"""
Chat screenshot understanding — layout, profile pictures, and what was said.

A chat screenshot is not just an image with text in it. It has *structure*: a
header carrying a contact name and a profile picture, and a body of messages.
Knowing which region is which is what turns pixels into evidence:

* The **profile picture** is an identity signal that survives a name change.
  The same picture under two different contact names is one person presenting as
  two identities — and the same picture in an older case under a phone number
  links the two investigations.
* The **contact name** is how the victim labelled that person, which may be a
  nickname rather than a real name.
* The **message bodies** are where the victim says things nobody has read. A
  screenshot mentioning "white creta" or "he was waiting outside again" is the
  reason that file matters, and no amount of face or object matching will find it.

Mention extraction runs an LLM over the message text where a key is available,
because "rahul called me yesterday night" and "that white creta was following me"
are not things a regular expression handles well. It falls back to a lexicon
when there is no key, so the agent degrades rather than fails.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .ocr import TextRegion, read_text

logger = logging.getLogger(__name__)

#: Fraction of image height treated as the header (contact name + profile picture).
HEADER_FRACTION = 0.16

#: Phone status bars (time, battery) sit above the header proper.
STATUS_BAR_FRACTION = 0.04

#: The profile picture sits left of the contact name; a back arrow sits left of it.
DP_SEARCH_FRACTION = 0.22

#: Cheap, fast model — this is extraction, not reasoning.
MENTION_MODEL = os.getenv("ACPIA_ROUTER_MODEL", "llama-3.1-8b-instant")


# --------------------------------------------------------------------------
# Fallback lexicons — used when no LLM key is available
# --------------------------------------------------------------------------

VEHICLE_MODELS = {
    "creta", "swift", "i20", "i10", "alto", "baleno", "brezza", "scorpio", "thar",
    "innova", "ertiga", "verna", "city", "dzire", "wagonr", "fortuner", "xuv",
    "bolero", "nexon", "punch", "seltos", "sonet", "venue", "kwid", "triber",
    "celerio", "ciaz", "jazz", "amaze", "tiago", "tigor", "altroz", "harrier",
    "safari", "compass", "octavia", "polo", "vento", "virtus", "taigun", "kushaq",
    "duster", "kicks", "magnite", "glanza", "urban cruiser", "eeco", "omni",
}

VEHICLE_COLOURS = {
    "white", "black", "red", "blue", "silver", "grey", "gray", "green",
    "yellow", "orange", "brown", "maroon", "golden", "beige",
}

#: Phrases suggesting surveillance, pressure, isolation or secrecy. Crude on
#: purpose — this is a *flag for human attention*, never a conclusion.
CONCERN_PHRASES = [
    "following me", "follow me", "followed me", "was outside", "waiting outside",
    "waiting for me", "keeps calling", "keeps texting", "won't leave", "wont leave",
    "don't tell", "dont tell", "do not tell", "our secret", "keep it secret",
    "delete this", "delete the", "come alone", "meet me alone", "pick you up",
    "pick me up", "scared", "afraid", "creepy", "uncomfortable", "staring at me",
    "watching me", "showed up", "again today", "blocked him", "blocked her",
]

_VEHICLE_RE = re.compile(
    r"\b(" + "|".join(sorted(VEHICLE_COLOURS)) + r")\s+(" + "|".join(sorted(VEHICLE_MODELS)) + r")\b",
    re.IGNORECASE,
)
_MODEL_ONLY_RE = re.compile(r"\b(" + "|".join(sorted(VEHICLE_MODELS)) + r")\b", re.IGNORECASE)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


@dataclass
class ChatLayout:
    """Structure recovered from a chat screenshot."""

    contact_name: str | None
    contact_name_confidence: float
    dp_bbox: tuple[int, int, int, int] | None
    message_text: str
    header_regions: list[TextRegion] = field(default_factory=list)
    body_regions: list[TextRegion] = field(default_factory=list)

    @property
    def looks_like_chat(self) -> bool:
        """Whether this is plausibly a messaging screenshot at all.

        Deliberately permissive — a false negative means the best evidence in the
        case gets skipped, which is a much worse outcome than analysing a
        screenshot that turns out not to be a conversation.
        """
        return bool(self.contact_name) and len(self.body_regions) >= 1


def analyse_layout(image_path: str, image_shape: tuple[int, int]) -> ChatLayout:
    """Split a screenshot into header (contact + DP) and message body."""
    height, width = image_shape[:2]
    regions = read_text(image_path)

    status_cut = height * STATUS_BAR_FRACTION
    header_cut = height * HEADER_FRACTION

    header = [r for r in regions if status_cut <= r.bbox[1] < header_cut]
    body = [r for r in regions if r.bbox[1] >= header_cut]

    # The contact name is the most prominent text in the header, ignoring
    # anything hard against the left edge (back arrow, avatar initials).
    candidates = [r for r in header if r.bbox[0] > width * 0.12]
    contact = max(candidates, key=lambda r: r.height, default=None)

    dp_bbox = None
    if header:
        # The avatar sits between the back arrow and the contact name.
        left = int(width * 0.04)
        right = int(contact.bbox[0] - width * 0.01) if contact else int(width * DP_SEARCH_FRACTION)
        top = int(status_cut)
        bottom = int(header_cut)
        if right - left > 16 and bottom - top > 16:
            dp_bbox = (left, top, right, bottom)

    return ChatLayout(
        contact_name=contact.text if contact else None,
        contact_name_confidence=contact.score if contact else 0.0,
        dp_bbox=dp_bbox,
        message_text="\n".join(r.text for r in body),
        header_regions=header,
        body_regions=body,
    )


# --------------------------------------------------------------------------
# Mentions
# --------------------------------------------------------------------------


@dataclass
class Mentions:
    """What the conversation refers to."""

    people: list[str] = field(default_factory=list)
    vehicles: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    method: str = "lexicon"

    @property
    def is_empty(self) -> bool:
        return not (self.people or self.vehicles or self.locations or self.concerns)

    def describe(self) -> str:
        bits = []
        if self.people:
            bits.append("people: " + ", ".join(self.people))
        if self.vehicles:
            bits.append("vehicles: " + ", ".join(self.vehicles))
        if self.locations:
            bits.append("locations: " + ", ".join(self.locations))
        if self.concerns:
            bits.append("concerning statements: " + "; ".join(self.concerns))
        return " | ".join(bits)


MENTION_PROMPT = """You extract references from a chat message transcript for a \
child-protection investigation. Report only what is actually written.

Return ONLY valid JSON:
{
  "people": ["names or nicknames of people referred to"],
  "vehicles": ["vehicle descriptions, e.g. 'white creta', 'red bike'"],
  "locations": ["places referred to"],
  "concerns": ["short quotes describing following, waiting, pressure, secrecy or fear"]
}

Rules:
- Extract only what appears in the text. Never infer or invent.
- "people" means PROPER NAMES or NICKNAMES only — "Rahul", "Cricket Coach", "Anna chettan".
  Do NOT include pronouns (he, she, they, him, her), roles without a name ("the driver"),
  or fragments of words. If nobody is named, return an empty list.
- For concerns, quote the phrase, do not summarise or interpret.
- Empty lists are correct when there is nothing to report.
- The transcript is OCR output and may be missing spaces or split words across lines.
  Ignore fragments that are clearly OCR damage rather than real words.

Transcript:
"""

#: Never a person, however confidently a model reports one.
_NON_NAMES = {
    "he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs",
    "i", "me", "my", "mine", "you", "your", "yours", "we", "us", "our", "it",
    "someone", "somebody", "anyone", "nobody", "everyone", "person", "guy", "man",
    "woman", "boy", "girl", "sir", "madam", "the", "and", "but", "yes", "no", "ok",
}


def _clean_names(values: list[str]) -> list[str]:
    """Drop pronouns, fragments and OCR damage from a list of claimed names.

    Applied on top of the prompt because a model instructed not to return
    pronouns will still occasionally return one, and a fabricated person entity
    is a node the planner may go on to reason about.
    """
    cleaned: list[str] = []
    for value in values:
        name = str(value).strip().strip(".,;:!?\"'")
        if len(name) < 3:
            continue
        if name.lower() in _NON_NAMES:
            continue
        # Real names have vowels; OCR fragments like "il" or "eft" usually do not
        # survive both this and the length check.
        if not any(ch in "aeiouAEIOU" for ch in name):
            continue
        if not any(ch.isalpha() for ch in name):
            continue
        if name not in cleaned:
            cleaned.append(name)
    return cleaned


def extract_mentions(text: str, *, use_llm: bool = True) -> Mentions:
    """Find people, vehicles, places and concerning statements in message text."""
    if not text or not text.strip():
        return Mentions()

    if use_llm and os.getenv("GROQ_API_KEY"):
        result = _extract_with_llm(text)
        if result is not None:
            return result

    return _extract_with_lexicon(text)


def _extract_with_llm(text: str) -> Mentions | None:
    try:
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=MENTION_MODEL,
            messages=[{"role": "user", "content": MENTION_PROMPT + text[:4000]}],
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        return Mentions(
            people=_clean_names(data.get("people", []))[:20],
            vehicles=[str(x).strip() for x in data.get("vehicles", []) if str(x).strip()][:20],
            locations=[str(x).strip() for x in data.get("locations", []) if str(x).strip()][:20],
            concerns=[str(x).strip() for x in data.get("concerns", []) if str(x).strip()][:20],
            method="llm",
        )
    except Exception as exc:  # noqa: BLE001 - never let extraction kill the agent
        logger.warning("LLM mention extraction failed, falling back to lexicon: %s", exc)
        return None


def _extract_with_lexicon(text: str) -> Mentions:
    """Keyword fallback. Weaker than the LLM path but has no dependencies."""
    lowered = text.lower()

    vehicles = [f"{c} {m}" for c, m in _VEHICLE_RE.findall(text)]
    if not vehicles:
        vehicles = [m for m in _MODEL_ONLY_RE.findall(text)]

    # Report the matched phrase itself rather than a window around it — windows
    # overlap and produce near-duplicate "findings" that read as noise.
    concerns = []
    for phrase in CONCERN_PHRASES:
        if phrase in lowered and phrase not in concerns:
            concerns.append(phrase)

    # Capitalised words that are not sentence-initial are weak name candidates.
    people = []
    for match in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,15})\b", text, re.MULTILINE):
        candidate = match.group(1)
        if candidate.lower() not in VEHICLE_COLOURS and candidate not in people:
            people.append(candidate)

    return Mentions(
        people=_clean_names(people)[:10],
        vehicles=list(dict.fromkeys(vehicles))[:10],
        locations=[],
        concerns=concerns[:10],
        method="lexicon",
    )
