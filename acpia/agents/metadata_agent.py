"""
Metadata agent — EXIF, GPS, device and capture-time extraction from images.

Ported from the standalone `task1_metadata/metadata.py` tool. The logic that
mattered is unchanged (DMS→decimal conversion, reverse geocoding to a real
place name); what changed is that it now returns structured Claims instead of
printing to a terminal, so the orchestrator can reason over the output.

This is the cheapest agent in the system and it runs first on every image. That
ordering is deliberate: EXIF gives us a device, a place and a time in
milliseconds, which is often enough for the planner to decide what expensive
agent to run next.

Two things here are investigative signals rather than plain extraction:

* **Absent EXIF** is reported, not skipped. Metadata stripping is a behaviour,
  and a folder where every image has been scrubbed says something.
* **A `Software` tag** is surfaced as a manipulation hint — an image that has
  been through an editor deserves the synthetic-detection agent's attention.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ..schema import (
    Claim,
    ClaimKind,
    Entity,
    EntityType,
    Evidence,
    EvidenceKind,
    EvidenceRef,
    RelationType,
    Relationship,
    RiskLevel,
)
from .base import Agent, AgentContext, registry

#: EXIF datetimes are "YYYY:MM:DD HH:MM:SS" — not ISO, so parsed explicitly.
EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"

#: Editing software whose presence suggests the image is not straight off a camera.
EDITOR_HINTS = ("photoshop", "gimp", "lightroom", "snapseed", "picsart", "canva", "ai")


def dms_to_decimal(dms: tuple[float, float, float], ref: str) -> float:
    """Convert EXIF degrees/minutes/seconds to signed decimal degrees.

    Reverse geocoders want decimal degrees; EXIF stores DMS plus a N/S/E/W
    reference character that carries the sign.
    """
    degrees, minutes, seconds = dms
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    return -decimal if ref in ("S", "W") else decimal


class MetadataAgent(Agent):
    name = "metadata_agent"
    description = (
        "Extract embedded metadata from an image: capture device (make/model), original "
        "capture timestamp, GPS coordinates resolved to a real city/state/country, and "
        "camera settings. Also reports when metadata is ABSENT, which may indicate "
        "deliberate stripping, and flags images that carry editing-software tags. "
        "This is the fastest agent available and should normally be run first on any "
        "image, because the device, place and time it returns often determine which "
        "expensive agent is worth running next. "
        "Keywords: exif, metadata, gps, location, geotag, device, camera, timestamp, when, where."
    )
    handles = {EvidenceKind.IMAGE}
    cost = 1

    def run(self, evidence: Evidence, ctx: AgentContext) -> Iterator[Claim]:
        from exif import Image as ExifImage  # imported lazily to keep startup fast

        path = Path(evidence.path)
        ref = EvidenceRef(evidence_id=evidence.evidence_id, excerpt=evidence.filename)

        with path.open("rb") as handle:
            image = ExifImage(handle)

        if not image.has_exif:
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.OBSERVATION,
                confidence=1.0,
                risk=RiskLevel.LOW,
                reason=(
                    f"{evidence.filename} carries no EXIF metadata at all. This may be "
                    f"innocuous (many platforms strip it on upload) or may indicate "
                    f"deliberate removal."
                ),
                source=ref,
                payload={"has_exif": False},
            )
            return

        captured_at = self._capture_time(image)
        device = self._device_entity(image)
        location, geo = self._location(image)

        # --- device + place + time, as one linked assertion ------------------
        if device or location:
            entities = [e for e in (device, location) if e]
            relationships = []
            if device and location:
                relationships.append(
                    Relationship(
                        subject_id=device.id,
                        predicate=RelationType.SEEN_AT,
                        object_id=location.id,
                        attributes={
                            "when": captured_at.isoformat() if captured_at else None,
                            "source": "exif_gps",
                        },
                    )
                )

            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.RELATIONSHIP if relationships else ClaimKind.ENTITY,
                confidence=0.95,
                reason=self._describe(evidence.filename, device, location, captured_at),
                source=ref,
                observed_at=captured_at,
                entities=entities,
                relationships=relationships,
                payload={"gps": geo, "has_exif": True},
            )

        # --- camera settings, kept as context rather than a headline ---------
        settings = self._camera_settings(image)
        if settings:
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.OBSERVATION,
                confidence=0.9,
                reason=(
                    f"Camera settings recovered from {evidence.filename}. Useful for "
                    f"establishing whether images claiming different origins were in fact "
                    f"taken on the same equipment."
                ),
                source=ref,
                observed_at=captured_at,
                payload=settings,
            )

        # --- manipulation hint ------------------------------------------------
        software = self._software(image)
        if software:
            edited = any(hint in software.casefold() for hint in EDITOR_HINTS)
            yield Claim(
                case_id=ctx.case_id,
                agent=self.name,
                kind=ClaimKind.OBSERVATION,
                confidence=0.8 if edited else 0.5,
                risk=RiskLevel.MEDIUM if edited else RiskLevel.LOW,
                reason=(
                    f"{evidence.filename} records processing software '{software}'. "
                    + (
                        "This is a known image editor, so the file is not a straight "
                        "camera original and warrants synthetic/manipulation checking."
                        if edited
                        else "Origin software noted for provenance."
                    )
                ),
                source=ref,
                observed_at=captured_at,
                payload={"software": software, "is_known_editor": edited},
            )

    # ------------------------------------------------------------- extractors

    @staticmethod
    def _capture_time(image: Any) -> datetime | None:
        raw = image.get("datetime_original") or image.get("datetime")
        if not raw:
            return None
        try:
            return datetime.strptime(raw, EXIF_DATETIME_FORMAT)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _device_entity(image: Any) -> Entity | None:
        make = (image.get("make") or "").strip()
        model = (image.get("model") or "").strip()
        if not (make or model):
            return None

        label = f"{make} {model}".strip()
        return Entity(
            type=EntityType.DEVICE,
            value=label,
            attributes={
                "make": make or None,
                "model": model or None,
                "body_serial": image.get("body_serial_number"),
                "lens_model": image.get("lens_model"),
            },
        )

    @staticmethod
    def _location(image: Any) -> tuple[Entity | None, dict[str, Any] | None]:
        """Resolve EXIF GPS to a named place. Returns (entity, raw geo dict)."""
        lat_dms = image.get("gps_latitude")
        lon_dms = image.get("gps_longitude")
        if not lat_dms or not lon_dms:
            return None, None

        try:
            latitude = dms_to_decimal(lat_dms, image.get("gps_latitude_ref", "N"))
            longitude = dms_to_decimal(lon_dms, image.get("gps_longitude_ref", "E"))
        except (TypeError, ValueError):
            return None, None

        import pycountry
        import reverse_geocoder

        hits = reverse_geocoder.search((latitude, longitude), verbose=False)
        if not hits:
            return None, {"latitude": latitude, "longitude": longitude}

        hit = hits[0]
        country_obj = pycountry.countries.get(alpha_2=hit.get("cc", ""))
        country = country_obj.name if country_obj else hit.get("cc", "Unknown")
        city = hit.get("name", "Unknown")
        state = hit.get("admin1", "Unknown")

        geo = {
            "latitude": latitude,
            "longitude": longitude,
            "city": city,
            "state": state,
            "country": country,
            "maps_url": f"https://www.google.com/maps?q={latitude},{longitude}",
            "altitude": image.get("gps_altitude"),
        }
        entity = Entity(
            type=EntityType.LOCATION,
            value=f"{city}, {state}, {country}",
            attributes=geo,
        )
        return entity, geo

    @staticmethod
    def _camera_settings(image: Any) -> dict[str, Any]:
        settings = {
            "aperture": image.get("f_number"),
            "focal_length_mm": image.get("focal_length"),
            "exposure_time_s": image.get("exposure_time"),
            "iso": image.get("photographic_sensitivity") or image.get("iso_speed_ratings"),
            "orientation": image.get("orientation"),
        }
        return {k: v for k, v in settings.items() if v is not None}

    @staticmethod
    def _software(image: Any) -> str | None:
        value = image.get("software")
        return str(value).strip() if value else None

    @staticmethod
    def _describe(
        filename: str, device: Entity | None, location: Entity | None, when: datetime | None
    ) -> str:
        parts = [f"{filename}"]
        if device:
            parts.append(f"was captured on a {device.value}")
        if when:
            parts.append(f"at {when:%Y-%m-%d %H:%M:%S}")
        if location:
            geo = location.attributes
            parts.append(
                f"at {location.value} ({geo['latitude']:.5f}, {geo['longitude']:.5f})"
            )
        return " ".join(parts) + "."


registry.register(MetadataAgent())
