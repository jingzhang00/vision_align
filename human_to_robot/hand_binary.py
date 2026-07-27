"""Stateless per-frame MANO geometry to an exact binary grasp action.

There is deliberately no interpolation, smoothing, hysteresis, event
detection, or state latch in this module.  Frame ``t`` depends only on MANO
landmarks and visibility at frame ``t``.

The geometry follows the action-alignment paper:

``virtual_finger = 0.7 * index_tip + 0.3 * middle_tip``

``ratio = ||thumb_tip - virtual_finger|| /
          ||virtual_finger - hand||``

The denominator makes the decision invariant to subject hand size and scene
scale.  A frame is closed when ``ratio <= 3/8``.  The threshold is fixed and
explicit; it is slightly wider than one third so a wide-object pinch is not
split by a small aperture change.  It is never calibrated from the episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import h5py
import numpy as np


PAPER_HAND_LANDMARKS = (
    "Hand",
    "ThumbTip",
    "IndexFingerTip",
    "MiddleFingerTip",
)
REQUIRED_LANDMARKS = PAPER_HAND_LANDMARKS
CLOSE_RATIO_THRESHOLD = 3.0 / 8.0
VIRTUAL_INDEX_WEIGHT = 0.7
VIRTUAL_MIDDLE_WEIGHT = 0.3


@dataclass(frozen=True)
class GraspRun:
    """Half-open interval containing consecutive per-frame closed labels."""

    start: int
    end: int
    confidence: float


@dataclass(frozen=True)
class HandBinaryDiagnostics:
    """Transparent inputs and decisions of the stateless classifier."""

    side: str
    method: str
    operation_enabled: bool
    operation_hands: tuple[str, ...]
    operation_hand_source: str
    close_ratio_threshold: float
    normalized_jaw_ratio: np.ndarray
    jaw_width_m: np.ndarray
    palm_length_m: np.ndarray
    frame_confidence: np.ndarray
    geometry_valid: np.ndarray
    visible: np.ndarray
    visibility_source: str
    closed_frame_count: int
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-serialisable diagnostics."""

        finite_ratio = self.normalized_jaw_ratio[
            np.isfinite(self.normalized_jaw_ratio)
        ]
        ratio_range = (
            [float(np.min(finite_ratio)), float(np.max(finite_ratio))]
            if finite_ratio.size
            else None
        )
        return {
            "side": self.side,
            "method": self.method,
            "operation_enabled": self.operation_enabled,
            "operation_hands": list(self.operation_hands),
            "operation_hand_source": self.operation_hand_source,
            "formula": (
                "||thumb-(0.7*index+0.3*middle)|| / "
                "||(0.7*index+0.3*middle)-Hand||"
            ),
            "decision": (
                "closed iff operation hand, visible, valid, and "
                f"normalized_jaw_ratio <= {self.close_ratio_threshold:g}"
            ),
            "close_ratio_threshold": self.close_ratio_threshold,
            "threshold_rationale": (
                "fixed per-frame threshold; no episode statistics, temporal "
                "smoothing, hysteresis, or state repair"
            ),
            "normalized_jaw_ratio_range": ratio_range,
            "valid_frame_fraction": float(np.mean(self.geometry_valid)),
            "visible_frame_fraction": float(np.mean(self.visible)),
            "closed_frame_count": self.closed_frame_count,
            "visibility_source": self.visibility_source,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class HandBinaryResult:
    """Exact per-frame binary labels and their contiguous diagnostic runs."""

    binary: np.ndarray
    runs: tuple[GraspRun, ...]
    diagnostics: HandBinaryDiagnostics


def _validate_tracks(
    positions: Mapping[str, np.ndarray],
    confidences: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    missing_positions = [
        name for name in PAPER_HAND_LANDMARKS if name not in positions
    ]
    missing_confidences = [
        name for name in PAPER_HAND_LANDMARKS if name not in confidences
    ]
    if missing_positions or missing_confidences:
        missing = sorted(set(missing_positions + missing_confidences))
        raise KeyError(f"Missing MANO hand landmarks: {', '.join(missing)}")

    hand = np.asarray(positions["Hand"])
    if hand.ndim != 2 or hand.shape[1] != 3:
        raise ValueError(f"Hand positions must have shape [N, 3], got {hand.shape}")
    frame_count = len(hand)
    if frame_count < 1:
        raise ValueError("At least one frame is required")

    clean_positions: dict[str, np.ndarray] = {}
    clean_confidences: dict[str, np.ndarray] = {}
    for name in PAPER_HAND_LANDMARKS:
        point = np.asarray(positions[name], dtype=np.float64)
        confidence = np.asarray(confidences[name], dtype=np.float64)
        if point.shape != (frame_count, 3):
            raise ValueError(
                f"{name} positions must have shape {(frame_count, 3)}, "
                f"got {point.shape}"
            )
        if confidence.shape != (frame_count,):
            raise ValueError(
                f"{name} confidence must have shape {(frame_count,)}, "
                f"got {confidence.shape}"
            )
        clean_positions[name] = point
        clean_confidences[name] = confidence
    return clean_positions, clean_confidences, frame_count


def _closed_runs(
    binary: np.ndarray,
    frame_confidence: np.ndarray,
) -> tuple[GraspRun, ...]:
    """Group labels for reporting only; this never changes a frame label."""

    padded = np.pad(np.asarray(binary, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return tuple(
        GraspRun(
            start=int(start),
            end=int(end),
            confidence=float(np.median(frame_confidence[start:end])),
        )
        for start, end in zip(starts, ends, strict=True)
    )


def infer_hand_binary(
    positions: Mapping[str, np.ndarray],
    confidences: Mapping[str, np.ndarray],
    *,
    side: str,
    operation_enabled: bool = True,
    operation_hands: tuple[str, ...] = ("left", "right"),
    operation_hand_source: str = "caller",
    visibility: np.ndarray | None = None,
    visibility_source: str = "caller",
    close_ratio_threshold: float = CLOSE_RATIO_THRESHOLD,
) -> HandBinaryResult:
    """Classify every frame independently from paper-aligned MANO geometry."""

    if side not in {"left", "right"}:
        raise ValueError(f"Unknown hand side: {side}")
    if (
        not np.isfinite(close_ratio_threshold)
        or close_ratio_threshold <= 0.0
    ):
        raise ValueError(
            "close_ratio_threshold must be a finite positive number"
        )
    clean_positions, clean_confidences, frame_count = _validate_tracks(
        positions,
        confidences,
    )
    if visibility is None:
        visible = np.ones(frame_count, dtype=bool)
        visibility_source = "not_provided_all_frames_visible"
    else:
        visible = np.asarray(visibility, dtype=bool)
        if visible.shape != (frame_count,):
            raise ValueError(
                f"visibility must have shape {(frame_count,)}, "
                f"got {visible.shape}"
            )

    hand = clean_positions["Hand"]
    thumb = clean_positions["ThumbTip"]
    index = clean_positions["IndexFingerTip"]
    middle = clean_positions["MiddleFingerTip"]
    virtual_finger = (
        VIRTUAL_INDEX_WEIGHT * index
        + VIRTUAL_MIDDLE_WEIGHT * middle
    )
    jaw_width = np.linalg.norm(thumb - virtual_finger, axis=1)
    palm_length = np.linalg.norm(virtual_finger - hand, axis=1)
    frame_confidence = np.min(
        np.stack(
            [
                clean_confidences[name]
                for name in PAPER_HAND_LANDMARKS
            ],
            axis=1,
        ),
        axis=1,
    )
    geometry_valid = (
        np.isfinite(
            np.stack(
                [hand, thumb, index, middle],
                axis=1,
            )
        ).all(axis=(1, 2))
        & np.isfinite(frame_confidence)
        & (frame_confidence > 0.0)
        & np.isfinite(jaw_width)
        & np.isfinite(palm_length)
        & (palm_length > 1e-6)
    )
    normalized_ratio = np.full(frame_count, np.nan, dtype=np.float64)
    normalized_ratio[geometry_valid] = (
        jaw_width[geometry_valid] / palm_length[geometry_valid]
    )
    binary = (
        operation_enabled
        & visible
        & geometry_valid
        & (normalized_ratio <= close_ratio_threshold + 1e-12)
    ).astype(np.uint8)

    warnings: list[str] = []
    if not operation_enabled:
        warnings.append("masked_by_hdf5_operation_hand_attribute")
    if not np.all(geometry_valid):
        warnings.append("invalid_geometry_frames_forced_open")
    if not np.all(visible):
        warnings.append("out_of_frame_hand_frames_forced_open")
    diagnostics = HandBinaryDiagnostics(
        side=side,
        method="stateless_paper_ratio_v1",
        operation_enabled=bool(operation_enabled),
        operation_hands=tuple(operation_hands),
        operation_hand_source=operation_hand_source,
        close_ratio_threshold=float(close_ratio_threshold),
        normalized_jaw_ratio=normalized_ratio,
        jaw_width_m=jaw_width,
        palm_length_m=palm_length,
        frame_confidence=frame_confidence,
        geometry_valid=geometry_valid,
        visible=visible,
        visibility_source=visibility_source,
        closed_frame_count=int(np.count_nonzero(binary)),
        warnings=tuple(warnings),
    )
    return HandBinaryResult(
        binary=binary,
        runs=_closed_runs(binary, frame_confidence),
        diagnostics=diagnostics,
    )


def _operation_hands_from_attrs(
    handle: h5py.File,
) -> tuple[tuple[str, ...], str]:
    """Read ``hand:left/right/both`` from scalar root task attributes."""

    matches: set[str] = set()
    matched_attrs: list[str] = []
    pattern = re.compile(r"\bhand\s*:\s*(left|right|both)\b", re.IGNORECASE)
    for name, value in handle.attrs.items():
        if isinstance(value, np.ndarray):
            values = value.reshape(-1).tolist()
        else:
            values = [value]
        for item in values:
            text = (
                item.decode("utf-8", errors="replace")
                if isinstance(item, bytes)
                else str(item)
            )
            found = pattern.findall(text)
            if found:
                matched_attrs.append(str(name))
            for label in found:
                normalized = label.lower()
                if normalized == "both":
                    matches.update(("left", "right"))
                else:
                    matches.add(normalized)
    if not matches:
        return ("left", "right"), "missing_default_both"
    return tuple(side for side in ("left", "right") if side in matches), (
        "hdf5_root_attrs:" + ",".join(sorted(set(matched_attrs)))
    )


def _camera_visibility(
    handle: h5py.File,
    *,
    side: str,
    hand_positions_world: np.ndarray,
    image_size: tuple[int, int] | None,
) -> tuple[np.ndarray, str]:
    """Project the current-frame wrist into the raw camera image."""

    frame_count = len(hand_positions_world)
    if "camera/intrinsic" not in handle or "transforms/camera" not in handle:
        return (
            np.ones(frame_count, dtype=bool),
            "camera_metadata_missing_all_frames_visible",
        )
    intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
    camera_to_world = np.asarray(
        handle["transforms/camera"],
        dtype=np.float64,
    )
    if intrinsic.shape != (3, 3):
        raise ValueError(
            f"camera/intrinsic must have shape (3, 3), got {intrinsic.shape}"
        )
    if camera_to_world.shape != (frame_count, 4, 4):
        raise ValueError(
            "transforms/camera must have shape "
            f"{(frame_count, 4, 4)}, got {camera_to_world.shape}"
        )
    if image_size is None:
        width = int(round(2.0 * intrinsic[0, 2]))
        height = int(round(2.0 * intrinsic[1, 2]))
        source = "camera_projection_K_principal_point_fallback"
    else:
        width, height = (int(image_size[0]), int(image_size[1]))
        source = "camera_projection_explicit_image_size"
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")

    homogeneous = np.concatenate(
        [
            np.asarray(hand_positions_world, dtype=np.float64),
            np.ones((frame_count, 1), dtype=np.float64),
        ],
        axis=1,
    )
    world_to_camera = np.linalg.inv(camera_to_world)
    points_camera = np.einsum(
        "nij,nj->ni",
        world_to_camera,
        homogeneous,
    )[:, :3]
    depth = points_camera[:, 2]
    pixel_x = np.full(frame_count, np.nan, dtype=np.float64)
    pixel_y = np.full(frame_count, np.nan, dtype=np.float64)
    projectable = np.isfinite(points_camera).all(axis=1) & (depth > 1e-6)
    pixel_x[projectable] = (
        intrinsic[0, 0]
        * points_camera[projectable, 0]
        / depth[projectable]
        + intrinsic[0, 2]
    )
    pixel_y[projectable] = (
        intrinsic[1, 1]
        * points_camera[projectable, 1]
        / depth[projectable]
        + intrinsic[1, 2]
    )
    visible = (
        projectable
        & (pixel_x >= 0.0)
        & (pixel_x < width)
        & (pixel_y >= 0.0)
        & (pixel_y < height)
    )
    return visible, f"{source}:{side}Hand"


def infer_binary_grasps_from_hdf5(
    hdf5_path: str | Path,
    *,
    image_size: tuple[int, int] | None = None,
    close_ratio_threshold: float = CLOSE_RATIO_THRESHOLD,
) -> dict[str, HandBinaryResult]:
    """Load both hands and classify each HDF5 frame independently."""
    path = Path(hdf5_path)
    tracks: dict[
        str,
        tuple[
            dict[str, np.ndarray],
            dict[str, np.ndarray],
            np.ndarray,
            str,
        ],
    ] = {}
    with h5py.File(path, "r") as handle:
        operation_hands, operation_source = _operation_hands_from_attrs(handle)
        for side in ("left", "right"):
            positions: dict[str, np.ndarray] = {}
            confidences: dict[str, np.ndarray] = {}
            for name in PAPER_HAND_LANDMARKS:
                transform_key = f"transforms/{side}{name}"
                confidence_key = f"confidences/{side}{name}"
                missing = [
                    key
                    for key in (transform_key, confidence_key)
                    if key not in handle
                ]
                if missing:
                    raise KeyError(f"{path} is missing: {', '.join(missing)}")
                transform = np.asarray(handle[transform_key], dtype=np.float64)
                if transform.ndim != 3 or transform.shape[1:] != (4, 4):
                    raise ValueError(
                        f"{transform_key} must have shape [N, 4, 4], "
                        f"got {transform.shape}"
                    )
                positions[name] = transform[:, :3, 3]
                confidences[name] = np.asarray(
                    handle[confidence_key],
                    dtype=np.float64,
                )
            visible, visibility_source = _camera_visibility(
                handle,
                side=side,
                hand_positions_world=positions["Hand"],
                image_size=image_size,
            )
            tracks[side] = (
                positions,
                confidences,
                visible,
                visibility_source,
            )
    return {
        side: infer_hand_binary(
            positions,
            confidences,
            side=side,
            operation_enabled=side in operation_hands,
            operation_hands=operation_hands,
            operation_hand_source=operation_source,
            visibility=visible,
            visibility_source=visibility_source,
            close_ratio_threshold=close_ratio_threshold,
        )
        for side, (
            positions,
            confidences,
            visible,
            visibility_source,
        ) in tracks.items()
    }
