"""
Video processing — frame sampling, tracking, and event synthesis.

Ported in spirit from the earlier SEIVR++ pipeline: detect, track, and turn
positions into semantic events. What changed is that events now become Claims
with provenance rather than strings in a list, and identity comes from face
embeddings and plates rather than tracker ids alone.

**Frame sampling is not an optimisation, it is a requirement.** A 20-second clip
at 30 fps is 600 frames; running the full stack on every one would take minutes
per file on a 6 GB laptop and add nothing — a person walking does not change
meaningfully between consecutive frames. Sampling at a few frames per second
captures every event that matters at a fraction of the cost.

**Tracker ids are not identities.** ByteTrack assigns an id that survives within
one clip and breaks on occlusion or re-entry, and means nothing across clips.
So a track here is only a *grouping of detections*; who it actually is comes from
running face recognition on the track's best frame. That distinction is what
lets the same person be recognised in two different videos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)

#: Frames analysed per second of video. Two is enough to catch someone walking
#: into shot, approaching a vehicle and leaving.
SAMPLE_FPS = 2.0

#: Hard ceiling on sampled frames, so a long clip cannot stall an investigation.
MAX_FRAMES = 240

#: A track seen fewer times than this is usually a detector flicker.
MIN_TRACK_HITS = 2

#: Fraction of a person's box that must fall inside a vehicle's box to count as
#: "at the vehicle".
VEHICLE_PROXIMITY_IOU = 0.12

#: A vehicle filling most of the frame carries no positional information — every
#: person in shot is trivially "inside" its box, so proximity becomes meaningless
#: and "walked to the car" fires the moment anyone appears. Vehicles above this
#: share of the frame are ignored for proximity events (they are still detected,
#: identified and reported; they just cannot anchor a location event).
MAX_VEHICLE_FRAME_SHARE = 0.55

#: How close counts as "at the vehicle", as a multiple of the person's own
#: height. Half a body height is roughly a stride — close enough that the next
#: unsampled frame would have them touching it.
PROXIMITY_GAP_RATIO = 0.6


@dataclass
class TrackDetection:
    """One detection of one tracked object in one sampled frame."""

    frame_index: int
    timestamp: float
    bbox: tuple[int, int, int, int]
    confidence: float

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(x2 - x1, 0) * max(y2 - y1, 0)

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class Track:
    """One object followed across sampled frames."""

    track_id: int
    label: str
    detections: list[TrackDetection] = field(default_factory=list)

    @property
    def first(self) -> TrackDetection:
        return self.detections[0]

    @property
    def last(self) -> TrackDetection:
        return self.detections[-1]

    @property
    def duration(self) -> float:
        return self.last.timestamp - self.first.timestamp

    @property
    def hits(self) -> int:
        return len(self.detections)

    def best_detection(self) -> TrackDetection:
        """The frame most likely to yield a usable face or plate.

        Largest area weighted by detector confidence — a big, confident crop is
        the one worth spending face recognition or OCR on.
        """
        return max(self.detections, key=lambda d: d.area * (0.5 + 0.5 * d.confidence))

    def timespan(self) -> str:
        return f"{_clock(self.first.timestamp)}–{_clock(self.last.timestamp)}"


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def containment(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    """Fraction of `inner` that falls inside `outer`.

    Better than IoU for "is this person at this car" — a person is much smaller
    than a car, so IoU stays low even when they are standing right at the door.
    """
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(inner[2] - inner[0], 0) * max(inner[3] - inner[1], 0)
    return inter / area if area else 0.0


def box_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Shortest distance between two boxes. Zero when they overlap."""
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return (dx * dx + dy * dy) ** 0.5


def near_vehicle(person_box: tuple[int, int, int, int], vehicle_box: tuple[int, int, int, int]) -> bool:
    """Whether a person is close enough to a vehicle to be 'at' it.

    Overlap alone is too strict. Frames are sampled a few times a second, so the
    moment someone actually reaches a door is usually between samples — the last
    frame we have shows them a stride away. Accepting a gap proportional to the
    person's own height keeps that case, which is precisely the case that matters,
    while still rejecting someone on the far side of the street.
    """
    if containment(person_box, vehicle_box) >= VEHICLE_PROXIMITY_IOU:
        return True
    person_height = max(person_box[3] - person_box[1], 1)
    return box_gap(person_box, vehicle_box) <= person_height * PROXIMITY_GAP_RATIO


@dataclass
class VideoInfo:
    frame_count: int
    fps: float
    duration: float
    width: int
    height: int
    sampled: int


def collect_tracks(
    video_path: str,
    *,
    sample_fps: float = SAMPLE_FPS,
    max_frames: int = MAX_FRAMES,
) -> tuple[dict[int, Track], VideoInfo, dict[int, Any]]:
    """Detect and track objects through a video.

    Returns (tracks, info, best_frames) where best_frames holds the decoded frame
    for each track's best detection — kept so face recognition and plate reading
    can run without decoding the video a second time.
    """
    import cv2

    from .detection import WANTED_CLASSES, get_detector

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    step = max(1, int(round(fps / max(sample_fps, 0.1))))
    model = get_detector()

    tracks: dict[int, Track] = {}
    frames_for_track: dict[int, Any] = {}
    best_area: dict[int, int] = {}

    index = 0
    sampled = 0

    while sampled < max_frames:
        ok, frame = capture.read()
        if not ok:
            break

        if index % step == 0:
            sampled += 1
            timestamp = index / fps

            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.35
            )

            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None or boxes.id is None:
                    continue
                for box, track_id, class_id, conf in zip(
                    boxes.xyxy, boxes.id, boxes.cls, boxes.conf
                ):
                    cid = int(class_id)
                    if cid not in WANTED_CLASSES:
                        continue
                    tid = int(track_id)
                    x1, y1, x2, y2 = (int(v) for v in box.tolist())

                    detection = TrackDetection(
                        frame_index=index,
                        timestamp=timestamp,
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf),
                    )
                    track = tracks.setdefault(tid, Track(tid, WANTED_CLASSES[cid]))
                    track.detections.append(detection)

                    # Keep the frame only while it is the best one for this track.
                    if detection.area > best_area.get(tid, 0):
                        best_area[tid] = detection.area
                        frames_for_track[tid] = frame.copy()

        index += 1

    capture.release()

    info = VideoInfo(
        frame_count=frame_count,
        fps=fps,
        duration=frame_count / fps if fps else 0.0,
        width=width,
        height=height,
        sampled=sampled,
    )

    solid = {tid: t for tid, t in tracks.items() if t.hits >= MIN_TRACK_HITS}
    return solid, info, frames_for_track


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


@dataclass
class VideoEvent:
    """Something that happened, expressed in time."""

    kind: str
    description: str
    timestamp: float
    person_track: int | None = None
    vehicle_track: int | None = None
    confidence: float = 0.7

    @property
    def clock(self) -> str:
        return _clock(self.timestamp)


def detect_events(
    tracks: dict[int, Track], frame_size: tuple[int, int] | None = None
) -> list[VideoEvent]:
    """Turn trajectories into events worth reporting.

    The one that matters is a person reaching a vehicle and not being seen
    again — the shape of someone getting in. It is stated as *"consistent with
    entering"* rather than as fact, because a person walking behind a van
    produces exactly the same signature.

    `frame_size` is (width, height). Supplying it filters out vehicles that fill
    the frame, which would otherwise make every person trivially "at the car".
    """
    from .detection import VEHICLE_CLASSES

    vehicle_labels = set(VEHICLE_CLASSES.values())
    people = [t for t in tracks.values() if t.label == "person"]
    vehicles = [t for t in tracks.values() if t.label in vehicle_labels]

    if frame_size:
        frame_area = max(frame_size[0] * frame_size[1], 1)
        vehicles = [
            v
            for v in vehicles
            if v.best_detection().area / frame_area <= MAX_VEHICLE_FRAME_SHARE
        ]

    events: list[VideoEvent] = []
    last_timestamp = max(
        (t.last.timestamp for t in tracks.values()), default=0.0
    )

    for person in people:
        events.append(
            VideoEvent(
                kind="person_present",
                description=(
                    f"A person is visible from {_clock(person.first.timestamp)} to "
                    f"{_clock(person.last.timestamp)} ({person.duration:.1f}s)"
                ),
                timestamp=person.first.timestamp,
                person_track=person.track_id,
                confidence=0.85,
            )
        )

        for vehicle in vehicles:
            # Did they overlap in time at all?
            overlap = [
                (pd, vd)
                for pd in person.detections
                for vd in vehicle.detections
                if pd.frame_index == vd.frame_index
            ]
            if not overlap:
                continue

            close = [(pd, vd) for pd, vd in overlap if near_vehicle(pd.bbox, vd.bbox)]
            if not close:
                continue

            first_close = min(close, key=lambda pair: pair[0].timestamp)
            events.append(
                VideoEvent(
                    kind="person_at_vehicle",
                    description=(
                        f"A person is at a {vehicle.label} at "
                        f"{_clock(first_close[0].timestamp)}"
                    ),
                    timestamp=first_close[0].timestamp,
                    person_track=person.track_id,
                    vehicle_track=vehicle.track_id,
                    confidence=0.75,
                )
            )

            # The track ends at the vehicle and the video continues without them.
            person_ended_early = person.last.timestamp < last_timestamp - 0.5
            ended_at_vehicle = near_vehicle(person.last.bbox, vehicle.last.bbox)
            if person_ended_early and ended_at_vehicle:
                events.append(
                    VideoEvent(
                        kind="person_entered_vehicle",
                        description=(
                            f"A person reaches a {vehicle.label} at "
                            f"{_clock(person.last.timestamp)} and is not seen again — "
                            f"consistent with entering the vehicle"
                        ),
                        timestamp=person.last.timestamp,
                        person_track=person.track_id,
                        vehicle_track=vehicle.track_id,
                        confidence=0.65,
                    )
                )

    return sorted(events, key=lambda e: e.timestamp)
