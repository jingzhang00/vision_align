from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


HAND_KEYPOINT_NAMES = (
    "Hand",
    "ThumbTip",
    "IndexFingerTip",
    "MiddleFingerTip",
)
HAND_LANDMARK_NAMES = (
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerMetacarpal",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerMetacarpal",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerMetacarpal",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerMetacarpal",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
)
HAND_TARGET_MODE_ATTRIBUTE = "human_to_robot_hand_target_mode"
FINGERTIP_GRASP_TARGET = "fingertip_grasp"
RIGID_PALM_TARGET = "rigid_palm"
HAND_TARGET_MODES = frozenset(
    {
        FINGERTIP_GRASP_TARGET,
        RIGID_PALM_TARGET,
    }
)
@dataclass(frozen=True)
class ArticulatedHandTrajectory:
    """Complete articulated hand track in two useful coordinate systems.

    Both transform arrays have shape ``[N, 25, 4, 4]`` and use
    :data:`HAND_LANDMARK_NAMES` order. ``transforms_reference_camera`` is
    expressed in the first video camera frame. ``transforms_palm`` is
    expressed in the corresponding frame's tracked ``Hand`` transform, so it
    contains articulation without global hand or camera motion. Confidence has
    shape ``[N, 25]`` and is preserved without smoothing or thresholding;
    source files without confidence receive ones.
    """

    names: tuple[str, ...]
    transforms_reference_camera: np.ndarray
    transforms_palm: np.ndarray
    confidence: np.ndarray

    @property
    def points_reference_camera(self) -> np.ndarray:
        return self.transforms_reference_camera[..., :3, 3]

    @property
    def rotations_reference_camera(self) -> np.ndarray:
        return self.transforms_reference_camera[..., :3, :3]

    @property
    def points_palm(self) -> np.ndarray:
        return self.transforms_palm[..., :3, 3]

    @property
    def rotations_palm(self) -> np.ndarray:
        return self.transforms_palm[..., :3, :3]

    @property
    def frame_count(self) -> int:
        return int(self.transforms_reference_camera.shape[0])


@dataclass(frozen=True)
class HandTrajectory:
    """Hand tracks and declared action targets in the first camera frame.

    Action targets always use the tracked ``*Hand`` pose. ``fingertip_grasp``
    inputs additionally provide articulated fingertip tracks for the fingers;
    explicit ``rigid_palm`` inputs leave finger width unavailable.
    """

    left_action_reference_camera: np.ndarray
    right_action_reference_camera: np.ndarray
    left_action_reference_rotations: np.ndarray
    right_action_reference_rotations: np.ndarray
    left_widths: np.ndarray
    right_widths: np.ndarray
    target_mode: str
    left_articulated_hand: ArticulatedHandTrajectory | None = None
    right_articulated_hand: ArticulatedHandTrajectory | None = None

    @property
    def frame_count(self) -> int:
        return int(self.left_action_reference_camera.shape[0])


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim == 2:
        homogeneous = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=points.dtype)],
            axis=1,
        )
        return np.einsum("nij,nj->ni", transform, homogeneous)[:, :3]
    if points.ndim == 3:
        homogeneous = np.concatenate(
            [
                points,
                np.ones((*points.shape[:2], 1), dtype=points.dtype),
            ],
            axis=2,
        )
        return np.einsum("nij,nkj->nki", transform, homogeneous)[..., :3]
    raise ValueError(f"Expected points with rank 2 or 3, got {points.shape}")


def _actual_savgol_window(
    length: int,
    requested: int,
    polynomial_order: int,
) -> int | None:
    if length < 5:
        return None
    window = min(requested, length if length % 2 else length - 1)
    minimum = polynomial_order + 2
    if minimum % 2 == 0:
        minimum += 1
    window = max(window, minimum)
    if window > length:
        window = length if length % 2 else length - 1
    return window if window > polynomial_order else None


def _interpolate_valid(
    values: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.shape[0] != valid.shape[0]:
        raise ValueError("Values and validity mask have different lengths")
    if valid.sum() < 2:
        raise ValueError("The hand trajectory contains fewer than two valid frames")
    result = values.copy()
    frame_ids = np.arange(len(result), dtype=np.float64)
    flattened = result.reshape(len(result), -1)
    for column in range(flattened.shape[1]):
        finite = valid & np.isfinite(flattened[:, column])
        if finite.sum() < 2:
            raise ValueError(
                "The hand trajectory contains fewer than two finite samples"
            )
        flattened[:, column] = np.interp(
            frame_ids,
            frame_ids[finite],
            flattened[finite, column],
        )
    return flattened.reshape(result.shape)


def _smooth_values(
    values: np.ndarray,
    valid: np.ndarray,
    window: int,
    polynomial_order: int,
) -> np.ndarray:
    result = _interpolate_valid(values, valid)
    actual_window = _actual_savgol_window(
        len(result), window, polynomial_order
    )
    if actual_window is None:
        return result
    return savgol_filter(
        result,
        window_length=actual_window,
        polyorder=min(polynomial_order, actual_window - 1),
        axis=0,
        mode="interp",
    )


def _smooth_rigid_body_rotations(
    matrices: np.ndarray,
    valid: np.ndarray,
    window: int,
    polynomial_order: int,
) -> np.ndarray:
    """Smooth tracked rigid-body rotations without changing their frame."""

    matrices = np.asarray(matrices, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(matrices).all(
        axis=(1, 2)
    )
    if valid.sum() < 2:
        raise ValueError(
            "The hand orientation contains fewer than two valid frames"
        )
    anchor_id = int(np.flatnonzero(valid)[0])
    anchor = Rotation.from_matrix(matrices[anchor_id])
    relative_vectors = np.full((len(matrices), 3), np.nan, dtype=np.float64)
    relative_vectors[valid] = (
        anchor.inv() * Rotation.from_matrix(matrices[valid])
    ).as_rotvec()
    smoothed_vectors = _smooth_values(
        relative_vectors,
        valid,
        window,
        polynomial_order,
    )
    return (anchor * Rotation.from_rotvec(smoothed_vectors)).as_matrix()


def align_hand_action(
    keypoints_reference_camera: np.ndarray,
    keypoint_confidence: np.ndarray,
    *,
    smooth_window: int = 11,
    polynomial_order: int = 2,
) -> np.ndarray:
    """Construct the continuous hand width."""
    points = np.asarray(keypoints_reference_camera, dtype=np.float64)
    confidence = np.asarray(keypoint_confidence, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (4, 3):
        raise ValueError(f"Expected hand keypoints [N, 4, 3], got {points.shape}")
    if confidence.shape != points.shape[:2]:
        raise ValueError(
            f"Expected hand confidence {points.shape[:2]}, got {confidence.shape}"
        )

    _, thumb, index, middle = np.moveaxis(points, 1, 0)
    virtual_finger = 0.7 * index + 0.3 * middle
    jaw = thumb - virtual_finger
    width = np.linalg.norm(jaw, axis=1)
    usable_keypoints = (
        np.isfinite(points).all(axis=2)
        & np.isfinite(confidence)
        & (confidence > 0.0)
    )
    finger_valid = usable_keypoints[:, 1:].all(axis=1)
    width_valid = (
        finger_valid
        & np.isfinite(width)
        & (width > 1e-6)
    )
    return np.maximum(
        _smooth_values(
            width[:, None],
            width_valid,
            smooth_window,
            polynomial_order,
        )[:, 0],
        0.0,
    )


def _hdf5_hand_transform_keys(side: str) -> tuple[str, ...]:
    return tuple(
        f"transforms/{side}{name}"
        for name in HAND_KEYPOINT_NAMES
    )


def _hdf5_hand_landmark_transform_keys(side: str) -> tuple[str, ...]:
    return tuple(
        f"transforms/{side}{name}"
        for name in HAND_LANDMARK_NAMES
    )


def _hdf5_optional_confidence(
    handle: h5py.File,
    *,
    key: str,
    frame_count: int,
) -> np.ndarray:
    """Load confidence when available, otherwise trust finite transforms."""

    if key not in handle:
        return np.ones(frame_count, dtype=np.float64)
    confidence = np.asarray(handle[key], dtype=np.float64)
    if confidence.shape != (frame_count,):
        raise ValueError(
            f"{key} must have shape {(frame_count,)}, got {confidence.shape}"
        )
    return confidence


def _hdf5_hand_landmarks_reference_camera(
    handle: h5py.File,
    *,
    side: str,
    reference_world_to_camera: np.ndarray,
    frame_count: int,
) -> ArticulatedHandTrajectory | None:
    """Load a complete articulated track when all 25 landmarks are present."""

    if not all(
        key in handle for key in _hdf5_hand_landmark_transform_keys(side)
    ):
        return None

    for name in HAND_LANDMARK_NAMES:
        transform = handle[f"transforms/{side}{name}"]
        if transform.shape != (frame_count, 4, 4):
            raise ValueError(
                f"transforms/{side}{name} must have shape "
                f"{(frame_count, 4, 4)}, got {transform.shape}"
            )
        _hdf5_optional_confidence(
            handle,
            key=f"confidences/{side}{name}",
            frame_count=frame_count,
        )

    transforms_world = np.stack(
        [
            np.asarray(
                handle[f"transforms/{side}{name}"],
                dtype=np.float64,
            )
            for name in HAND_LANDMARK_NAMES
        ],
        axis=1,
    )
    confidence = np.stack(
        [
            _hdf5_optional_confidence(
                handle,
                key=f"confidences/{side}{name}",
                frame_count=frame_count,
            )
            for name in HAND_LANDMARK_NAMES
        ],
        axis=1,
    )
    transforms_reference_camera = np.matmul(
        reference_world_to_camera[None, None],
        transforms_world,
    )

    hand_index = HAND_LANDMARK_NAMES.index("Hand")
    hand_transforms = transforms_reference_camera[:, hand_index]
    valid_hand = np.isfinite(hand_transforms).all(axis=(1, 2))
    determinants = np.full(frame_count, np.nan, dtype=np.float64)
    determinants[valid_hand] = np.linalg.det(
        hand_transforms[valid_hand, :3, :3]
    )
    valid_hand &= np.isfinite(determinants)
    valid_hand &= np.abs(determinants) > 1e-12

    reference_to_palm = np.full(
        (frame_count, 4, 4),
        np.nan,
        dtype=np.float64,
    )
    reference_to_palm[valid_hand] = np.linalg.inv(
        hand_transforms[valid_hand]
    )
    transforms_palm = np.einsum(
        "nij,nkjl->nkil",
        reference_to_palm,
        transforms_reference_camera,
    )
    return ArticulatedHandTrajectory(
        names=HAND_LANDMARK_NAMES,
        transforms_reference_camera=transforms_reference_camera,
        transforms_palm=transforms_palm,
        confidence=confidence,
    )


def _hdf5_hand_usable(handle: h5py.File, *, side: str) -> bool:
    if (
        "transforms/camera" not in handle
        or not all(
            key in handle for key in _hdf5_hand_transform_keys(side)
        )
    ):
        return False
    frame_count = int(handle["transforms/camera"].shape[0])
    valid = np.ones(frame_count, dtype=bool)
    for name in HAND_KEYPOINT_NAMES:
        transform = handle[f"transforms/{side}{name}"]
        if transform.shape != (frame_count, 4, 4):
            return False
        valid &= np.isfinite(transform[:, :3, 3]).all(axis=1)
        confidence_key = f"confidences/{side}{name}"
        if confidence_key in handle:
            confidence = handle[confidence_key]
            if confidence.shape != (frame_count,):
                return False
            confidence_values = np.asarray(confidence, dtype=np.float64)
            valid &= np.isfinite(confidence_values)
            valid &= confidence_values > 0.0
    return int(valid.sum()) >= 2


def _hdf5_rigid_palm_usable(handle: h5py.File, *, side: str) -> bool:
    transform_key = f"transforms/{side}Hand"
    confidence_key = f"confidences/{side}Hand"
    if (
        "transforms/camera" not in handle
        or transform_key not in handle
    ):
        return False
    frame_count = int(handle["transforms/camera"].shape[0])
    transform = handle[transform_key]
    if transform.shape != (frame_count, 4, 4):
        return False
    transform_values = np.asarray(transform, dtype=np.float64)
    valid = np.isfinite(transform_values[:, :3, :]).all(axis=(1, 2))
    if confidence_key in handle:
        confidence = handle[confidence_key]
        if confidence.shape != (frame_count,):
            return False
        confidence_values = np.asarray(confidence, dtype=np.float64)
        valid &= np.isfinite(confidence_values)
        valid &= confidence_values > 0.0
    return int(valid.sum()) >= 2


def _hdf5_hand_target_mode(handle: h5py.File) -> str:
    raw_mode = handle.attrs.get(
        HAND_TARGET_MODE_ATTRIBUTE,
        FINGERTIP_GRASP_TARGET,
    )
    if isinstance(raw_mode, bytes):
        raw_mode = raw_mode.decode("utf-8")
    mode = str(raw_mode)
    if mode not in HAND_TARGET_MODES:
        raise ValueError(
            f"Unsupported {HAND_TARGET_MODE_ATTRIBUTE}={mode!r}; "
            f"expected one of {sorted(HAND_TARGET_MODES)}"
        )
    return mode


def hdf5_hand_target_mode(hdf5_path: str | Path) -> str:
    with h5py.File(hdf5_path, "r") as handle:
        return _hdf5_hand_target_mode(handle)


def _hdf5_keypoints_reference_camera(
    handle: h5py.File,
    *,
    side: str,
    reference_world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transforms = np.stack(
        [
            np.asarray(handle[f"transforms/{side}{name}"], dtype=np.float64)
            for name in HAND_KEYPOINT_NAMES
        ],
        axis=1,
    )
    confidence = np.stack(
        [
            _hdf5_optional_confidence(
                handle,
                key=f"confidences/{side}{name}",
                frame_count=transforms.shape[0],
            )
            for name in HAND_KEYPOINT_NAMES
        ],
        axis=1,
    )
    points_world = transforms[..., :3, 3]
    repeated = np.repeat(
        reference_world_to_camera[None],
        points_world.shape[0],
        axis=0,
    )
    return _transform_points(repeated, points_world), confidence


def load_hand_trajectory(
    hdf5_path: str | Path,
    *,
    smooth_window: int = 11,
    orientation_smooth_window: int = 15,
    polynomial_order: int = 2,
) -> HandTrajectory:
    """Load hand targets, defaulting absent confidence tracks to one."""

    path = Path(hdf5_path)
    with h5py.File(path, "r") as handle:
        if "transforms/camera" not in handle:
            raise KeyError(f"{path} is missing dataset: transforms/camera")
        camera_to_world = np.asarray(
            handle["transforms/camera"], dtype=np.float64
        )
        if camera_to_world.ndim != 3 or camera_to_world.shape[1:] != (4, 4):
            raise ValueError(
                "transforms/camera must have shape [frame_count, 4, 4]"
            )
        frame_count = len(camera_to_world)
        reference_world_to_camera = np.linalg.inv(camera_to_world[0])
        target_mode = _hdf5_hand_target_mode(handle)
        target_availability = (
            _hdf5_rigid_palm_usable
            if target_mode == RIGID_PALM_TARGET
            else _hdf5_hand_usable
        )
        availability = {
            side: target_availability(handle, side=side)
            for side in ("left", "right")
        }
        missing_sides = [
            side for side, usable in availability.items() if not usable
        ]
        if missing_sides:
            if target_mode == RIGID_PALM_TARGET:
                raise ValueError(
                    f"{path} lacks usable rigid palm transforms for "
                    f"{', '.join(missing_sides)} hand(s). Required "
                    "leftHand/rightHand transforms with at least two valid "
                    "frames; confidence is optional."
                )
            required_names = ", ".join(HAND_KEYPOINT_NAMES)
            raise ValueError(
                f"{path} lacks usable annotated keypoints for "
                f"{', '.join(missing_sides)} hand(s). Required transforms: "
                f"{required_names}, with at least two valid frames; "
                "confidence is optional."
            )
        keypoints = (
            {
                side: _hdf5_keypoints_reference_camera(
                    handle,
                    side=side,
                    reference_world_to_camera=reference_world_to_camera,
                )
                for side in ("left", "right")
            }
            if target_mode == FINGERTIP_GRASP_TARGET
            else None
        )
        rigid_body_tracks = {
            side: (
                np.asarray(
                    handle[f"transforms/{side}Hand"],
                    dtype=np.float64,
                ),
                _hdf5_optional_confidence(
                    handle,
                    key=f"confidences/{side}Hand",
                    frame_count=frame_count,
                ),
            )
            for side in ("left", "right")
        }
        hand_landmark_tracks = {
            side: _hdf5_hand_landmarks_reference_camera(
                handle,
                side=side,
                reference_world_to_camera=reference_world_to_camera,
                frame_count=frame_count,
            )
            for side in ("left", "right")
        }

    aligned = (
        {
            side: align_hand_action(
                *keypoints[side],
                smooth_window=smooth_window,
                polynomial_order=polynomial_order,
            )
            for side in ("left", "right")
        }
        if keypoints is not None
        else None
    )
    ik_targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    repeated_reference_transform = np.repeat(
        reference_world_to_camera[None],
        frame_count,
        axis=0,
    )
    for side in ("left", "right"):
        transforms, confidence = rigid_body_tracks[side]
        position_valid = (
            np.isfinite(transforms[:, :3, 3]).all(axis=1)
            & np.isfinite(confidence)
            & (confidence > 0.2)
        )
        reference_position = _smooth_values(
            _transform_points(
                repeated_reference_transform,
                transforms[:, :3, 3],
            ),
            position_valid,
            smooth_window,
            polynomial_order,
        )
        rotation_valid = (
            np.isfinite(confidence)
            & (confidence > 0.2)
            & np.isfinite(transforms[:, :3, :3]).all(axis=(1, 2))
        )
        world_rotation = _smooth_rigid_body_rotations(
            transforms[:, :3, :3],
            rotation_valid,
            orientation_smooth_window,
            polynomial_order,
        )
        reference_rotation = np.einsum(
            "ij,njk->nik",
            reference_world_to_camera[:3, :3],
            world_rotation,
        )
        ik_targets[side] = (
            reference_position,
            reference_rotation,
        )

    left_position, left_rotation = ik_targets["left"]
    right_position, right_rotation = ik_targets["right"]
    if aligned is None:
        left_width = np.full(frame_count, np.nan, dtype=np.float64)
        right_width = np.full(frame_count, np.nan, dtype=np.float64)
    else:
        left_width = aligned["left"]
        right_width = aligned["right"]
    return HandTrajectory(
        left_action_reference_camera=left_position,
        right_action_reference_camera=right_position,
        left_action_reference_rotations=left_rotation,
        right_action_reference_rotations=right_rotation,
        left_widths=left_width,
        right_widths=right_width,
        target_mode=target_mode,
        left_articulated_hand=hand_landmark_tracks["left"],
        right_articulated_hand=hand_landmark_tracks["right"],
    )
