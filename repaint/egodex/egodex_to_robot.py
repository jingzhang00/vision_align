#!/usr/bin/env python3
"""Retarget EgoDex directly to G1 and write LeRobot v3.

The input is a directory of same-stem EgoDex ``.hdf5``/``.mp4`` pairs.  For
each episode this script:

1. maps the tracked body and hand landmarks into a canonical Z-up frame,
2. combines it with a static G1-derived standing target,
3. solves the body with Nymeria's 29-DoF ``unitree_g1`` GMR path,
4. inserts the independently retargeted 22 O6 hand joints, and
5. writes 51-DoF references, camera calibration, and frame-aligned RGB.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import mujoco as mj
import numpy as np
import pandas as pd
import pinocchio as pin
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
STREAM_ROOT = WORKSPACE_ROOT / "stream"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(STREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM_ROOT))

from general_motion_retargeting import (  # noqa: E402
    GeneralMotionRetargeting as GMR,
)
from repaint.g1_hand import (  # noqa: E402
    O6HandRetargetTrajectory,
    TRACKED_HAND_TO_G1_PALM_ROTATIONS,
    build_pinocchio_g1,
    retarget_o6_hands,
)
from repaint.nymeria.nymeria_to_robot import (  # noqa: E402
    merge_body_with_open_hands,
)
from repaint.robot_camera import (  # noqa: E402
    HEAD_CAMERA_PARENT_FRAME,
    retarget_scene_camera_poses_cv,
    robot_frame_poses_world,
)
from repaint.vda_pipeline_common import (  # noqa: E402
    natural_key,
    remove_exact,
    write_json_atomic,
)


VIDEO_KEY = "observation.images.ego"
DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH_TEMPLATE = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
EPISODES_PATH_TEMPLATE = (
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
)
CHUNKS_SIZE = 1000
DATA_FILE_SIZE_MB = 100
VIDEO_FILE_SIZE_MB = 200
STAT_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
G1_URDF = (
    WORKSPACE_ROOT
    / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
)
SMOOTH_WINDOW = 7
ACTION_OFFSET = 1
IMAGE_STAT_SAMPLES = 24
BODY_TRACK_KEYS = (
    "hip",
    "leftArm",
    "leftForearm",
    "leftHand",
    "rightArm",
    "rightForearm",
    "rightHand",
)

TRACK_KEYS = BODY_TRACK_KEYS
GMR_BODY_LINKS = {
    "pelvis": "pelvis",
    "left_hip": "left_hip_roll_link",
    "left_knee": "left_knee_link",
    "left_foot": "left_toe_link",
    "right_hip": "right_hip_roll_link",
    "right_knee": "right_knee_link",
    "right_foot": "right_toe_link",
    "spine3": "torso_link",
    "left_shoulder": "left_shoulder_yaw_link",
    "left_elbow": "left_elbow_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_shoulder": "right_shoulder_yaw_link",
    "right_elbow": "right_elbow_link",
    "right_wrist": "right_wrist_yaw_link",
}
GMR_UPPER_BODY = (
    "spine3",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    pixel_format: str
    has_audio: bool


@dataclass(frozen=True)
class SourcePair:
    stem: str
    hdf5_path: Path
    video_path: Path


@dataclass
class EgoDexEpisode:
    stem: str
    hdf5_path: Path
    video_path: Path
    video: VideoInfo
    task: str
    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    source_spine3_to_world: np.ndarray
    points: dict[str, np.ndarray]
    source_to_robot_world_rotation: np.ndarray

    @property
    def frame_count(self) -> int:
        return self.video.frame_count


@dataclass
class RetargetedEpisode:
    source: EgoDexEpisode
    state: np.ndarray
    action: np.ndarray
    root_pose: np.ndarray
    camera_pose: np.ndarray
    source_camera_pose_in_spine3: np.ndarray
    source_camera_rotation_world: np.ndarray
    source_to_robot_world_rotation: np.ndarray
    source_spine3_to_robot_torso_rotation: np.ndarray
    camera_intrinsics: np.ndarray
    scene_scale: float
    metrics: dict[str, float | int]
    hand_diagnostics: dict[str, dict[str, object]]


def probe_video(path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required but was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffprobe failed for {path}:\n{exc.stderr.strip()}"
        ) from exc

    payload = json.loads(result.stdout)
    video_streams = [
        stream
        for stream in payload["streams"]
        if stream["codec_type"] == "video"
    ]
    if len(video_streams) != 1:
        raise ValueError(
            f"{path} must contain exactly one video stream, found "
            f"{len(video_streams)}"
        )
    stream = video_streams[0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = int(numerator) / int(denominator)
    frame_count = int(stream["nb_read_frames"])
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Invalid video timing in {path}")

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=float(fps),
        frame_count=frame_count,
        codec=str(stream["codec_name"]),
        pixel_format=str(stream["pix_fmt"]),
        has_audio=any(
            item["codec_type"] == "audio"
            for item in payload["streams"]
        ),
    )


def discover_source_pairs(src_root: Path) -> list[SourcePair]:
    if not src_root.is_dir():
        raise FileNotFoundError(f"EgoDex source directory does not exist: {src_root}")

    hdf5_files = list(src_root.rglob("*.hdf5"))
    hdf5_files = sorted(
        set(path.resolve() for path in hdf5_files),
        key=lambda path: natural_key(str(path.relative_to(src_root.resolve()))),
    )
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {src_root}")

    pairs: list[SourcePair] = []
    missing: list[Path] = []
    for hdf5_path in hdf5_files:
        video_path = hdf5_path.with_suffix(".mp4")
        if not video_path.is_file():
            missing.append(video_path)
            continue
        relative = hdf5_path.relative_to(src_root.resolve()).with_suffix("")
        pairs.append(
            SourcePair(
                stem=relative.as_posix(),
                hdf5_path=hdf5_path,
                video_path=video_path,
            )
        )
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Every EgoDex HDF5 file must have a same-stem MP4 file. Missing:\n"
            f"{formatted}"
        )
    return pairs


def _smooth_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("Track positions must be finite")
    if len(points) < SMOOTH_WINDOW:
        return points
    return savgol_filter(
        points,
        window_length=SMOOTH_WINDOW,
        polyorder=2,
        axis=0,
        mode="interp",
    )


def _validate_transforms(
    name: str,
    transforms: np.ndarray,
    frame_count: int,
) -> None:
    if transforms.shape != (frame_count, 4, 4):
        raise ValueError(
            f"{name} must have shape {(frame_count, 4, 4)}, got "
            f"{transforms.shape}"
        )
    if not np.isfinite(transforms).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(
        transforms[:, 3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-4,
    ):
        raise ValueError(f"{name} contains invalid homogeneous transforms")


def load_egodex_episode(
    pair: SourcePair,
) -> EgoDexEpisode:
    video = probe_video(pair.video_path)
    with h5py.File(pair.hdf5_path, "r") as handle:
        if (
            "camera/intrinsic" not in handle
            or "transforms/camera" not in handle
            or "transforms/spine3" not in handle
        ):
            raise KeyError(
                f"{pair.hdf5_path} must contain camera/intrinsic, "
                "transforms/camera, and transforms/spine3"
            )
        frame_count = int(handle["transforms/camera"].shape[0])
        if frame_count != video.frame_count:
            raise ValueError(
                f"Frame mismatch for {pair.stem}: HDF5={frame_count}, "
                f"MP4={video.frame_count}"
            )

        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError(
                f"camera/intrinsic must be a finite 3x3 matrix, got "
                f"{intrinsic.shape}"
            )
        camera_to_world = np.asarray(
            handle["transforms/camera"],
            dtype=np.float64,
        )
        _validate_transforms(
            "transforms/camera",
            camera_to_world,
            frame_count,
        )
        source_spine3 = np.asarray(
            handle["transforms/spine3"],
            dtype=np.float64,
        )
        _validate_transforms(
            "transforms/spine3",
            source_spine3,
            frame_count,
        )

        points: dict[str, np.ndarray] = {}
        hip_rotation: np.ndarray | None = None
        for key in TRACK_KEYS:
            transform_key = f"transforms/{key}"
            transforms = np.asarray(handle[transform_key], dtype=np.float64)
            _validate_transforms(
                transform_key,
                transforms,
                frame_count,
            )
            points[key] = _smooth_points(transforms[:, :3, 3])
            if key == "hip":
                hip_rotation = transforms[:, :3, :3]
        task = str(handle.attrs["llm_description"]).strip()

    # EgoDex hip local axes are +X left, +Y up, +Z forward.  They are more
    # stable than reconstructing a frame from occlusion-prone shoulder points.
    # Reorder the initial columns into the G1 convention (+X forward, +Y
    # left, +Z up), then invert that one source-world alignment. Later human
    # turns must not change the fixed world basis.
    assert hip_rotation is not None
    canonical_to_source = np.stack(
        [
            hip_rotation[:, :, 2],
            hip_rotation[:, :, 0],
            hip_rotation[:, :, 1],
        ],
        axis=2,
    )
    u, _, vt = np.linalg.svd(canonical_to_source[0])
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    source_to_robot = (u @ correction @ vt).T
    if not np.allclose(source_to_robot @ source_to_robot.T, np.eye(3), atol=1e-5):
        raise ValueError(f"Computed basis is not orthonormal for {pair.stem}")
    if np.linalg.det(source_to_robot) < 0.999:
        raise ValueError(f"Computed basis is left-handed for {pair.stem}")

    return EgoDexEpisode(
        stem=pair.stem,
        hdf5_path=pair.hdf5_path,
        video_path=pair.video_path,
        video=video,
        task=task,
        intrinsic=intrinsic,
        camera_to_world=camera_to_world,
        source_spine3_to_world=source_spine3,
        points=points,
        source_to_robot_world_rotation=source_to_robot,
    )


def _copy_human_frame(
    frame: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: (position.copy(), quaternion.copy())
        for name, (position, quaternion) in frame.items()
    }


def _standing_gmr_frame(
    retarget: GMR,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Invert GMR preprocessing at the robot's neutral standing pose."""

    mj.mj_forward(retarget.model, retarget.configuration.data)
    data = retarget.configuration.data
    root_link = GMR_BODY_LINKS["pelvis"]
    root_id = mj.mj_name2id(
        retarget.model,
        mj.mjtObj.mjOBJ_BODY,
        root_link,
    )
    root_target_rotation = Rotation.from_matrix(
        data.xmat[root_id].reshape(3, 3)
    )
    root_target_position = data.xpos[root_id].copy()
    root_offset = retarget.pos_offsets1["pelvis"]
    scaled_root_position = (
        root_target_position
        - root_target_rotation.apply(root_offset)
    )
    raw_root_position = (
        scaled_root_position
        / float(retarget.human_scale_table["pelvis"])
    )

    frame: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for human_name, robot_link in GMR_BODY_LINKS.items():
        body_id = mj.mj_name2id(
            retarget.model,
            mj.mjtObj.mjOBJ_BODY,
            robot_link,
        )
        target_rotation = Rotation.from_matrix(
            data.xmat[body_id].reshape(3, 3)
        )
        target_position = data.xpos[body_id].copy()
        raw_rotation = (
            target_rotation
            * retarget.rot_offsets1[human_name].inv()
        )
        scaled_position = (
            target_position
            - target_rotation.apply(retarget.pos_offsets1[human_name])
        )
        if human_name == "pelvis":
            raw_position = raw_root_position
        else:
            raw_position = raw_root_position + (
                scaled_position - scaled_root_position
            ) / float(retarget.human_scale_table[human_name])
        frame[human_name] = (
            raw_position,
            raw_rotation.as_quat(scalar_first=True),
        )

    retarget.update_targets(_copy_human_frame(frame))
    if retarget.error1() > 1e-6:
        raise RuntimeError("Failed to reconstruct the neutral G1 standing target")
    return frame


def _egodex_gmr_frames(
    episode: EgoDexEpisode,
    standing: dict[str, tuple[np.ndarray, np.ndarray]],
    retarget: GMR,
    hand: O6HandRetargetTrajectory,
) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    if hand.source_hand_transforms_reference_camera.shape != (
        episode.frame_count,
        2,
        4,
        4,
    ):
        raise ValueError(
            f"Tracked Hand trajectory length differs for {episode.stem}"
        )

    basis = episode.source_to_robot_world_rotation
    reference_camera_to_source = episode.camera_to_world[0]
    robot_from_reference_camera = (
        basis @ reference_camera_to_source[:3, :3]
    )
    raw_root_position = standing["pelvis"][0]
    scaled_root_position = (
        float(retarget.human_scale_table["pelvis"]) * raw_root_position
    )
    ground_offset = np.asarray(
        [0.0, 0.0, float(retarget.ground_offset)],
        dtype=np.float64,
    )
    upper_positions: dict[str, np.ndarray] = {}
    upper_quaternions: dict[str, np.ndarray] = {}
    upper_positions["spine3"] = np.repeat(
        standing["spine3"][0][None],
        episode.frame_count,
        axis=0,
    )
    upper_quaternions["spine3"] = np.repeat(
        standing["spine3"][1][None],
        episode.frame_count,
        axis=0,
    )

    data = retarget.configuration.data
    for side_index, side in enumerate(("left", "right")):
        shoulder_name = f"{side}_shoulder"
        elbow_name = f"{side}_elbow"
        wrist_name = f"{side}_wrist"
        source_shoulder = episode.points[f"{side}Arm"]
        source_elbow = episode.points[f"{side}Forearm"]
        source_wrist = episode.points[f"{side}Hand"]
        raw_target_shoulder = standing[shoulder_name][0]
        upper_positions[shoulder_name] = np.repeat(
            raw_target_shoulder[None],
            episode.frame_count,
            axis=0,
        )
        upper_positions[elbow_name] = (
            raw_target_shoulder
            + (basis @ (source_elbow - source_shoulder).T).T
        )

        body_ids = {
            name: mj.mj_name2id(
                retarget.model,
                mj.mjtObj.mjOBJ_BODY,
                GMR_BODY_LINKS[name],
            )
            for name in (shoulder_name, elbow_name, wrist_name)
        }
        neutral_upper = (
            data.xpos[body_ids[elbow_name]]
            - data.xpos[body_ids[shoulder_name]]
        )
        neutral_forearm = (
            data.xpos[body_ids[wrist_name]]
            - data.xpos[body_ids[elbow_name]]
        )
        source_upper = (basis @ (source_elbow - source_shoulder).T).T
        source_forearm = (basis @ (source_wrist - source_elbow).T).T
        alignments = {
            shoulder_name: _direction_alignments(neutral_upper, source_upper),
            elbow_name: _direction_alignments(neutral_forearm, source_forearm),
        }
        for human_name, alignment in alignments.items():
            neutral_rotation = Rotation.from_matrix(
                data.xmat[body_ids[human_name]].reshape(3, 3)
            )
            target_rotation = alignment * neutral_rotation
            raw_rotation = (
                target_rotation
                * retarget.rot_offsets1[human_name].inv()
            )
            upper_quaternions[human_name] = raw_rotation.as_quat(
                scalar_first=True
            )

        # Finger landmarks drive only O6 articulation. The GMR wrist target is
        # the tracked Hand pose with one fixed side-specific local-axis map.
        source_hand = (
            hand.source_hand_transforms_reference_camera[:, side_index]
        )
        source_hand_position = (
            reference_camera_to_source[:3, 3]
            + (
                reference_camera_to_source[:3, :3]
                @ source_hand[:, :3, 3].T
            ).T
        )
        desired_wrist_rotation = np.einsum(
            "ij,njk,kl->nil",
            robot_from_reference_camera,
            source_hand[:, :3, :3],
            TRACKED_HAND_TO_G1_PALM_ROTATIONS[side_index],
        )
        # GMR appends its configured SMPL-X-to-G1 wrist rotation offset.
        # Remove it here so the solved robot frame, not the pre-offset human
        # task frame, matches the tracked Hand orientation.
        raw_wrist_rotation = (
            Rotation.from_matrix(desired_wrist_rotation)
            * retarget.rot_offsets1[wrist_name].inv()
        )
        if not np.allclose(
            (
                raw_wrist_rotation
                * retarget.rot_offsets1[wrist_name]
            ).as_matrix(),
            desired_wrist_rotation,
            atol=1e-9,
        ):
            raise RuntimeError(f"Failed to compensate GMR {wrist_name} offset")
        wrist_scale = float(retarget.human_scale_table[wrist_name])
        standing_shoulder_position = np.asarray(
            retarget.scaled_human_data[shoulder_name][0],
            dtype=np.float64,
        )
        wrist_position = (
            standing_shoulder_position
            + wrist_scale
            * (basis @ (source_hand_position - source_shoulder).T).T
        )
        wrist_position_offset = retarget.pos_offsets1[wrist_name]
        upper_positions[wrist_name] = (
            raw_root_position
            + (
                wrist_position
                - np.einsum(
                    "nij,j->ni",
                    desired_wrist_rotation,
                    wrist_position_offset,
                )
                + ground_offset
                - scaled_root_position
            )
            / wrist_scale
        )
        upper_quaternions[wrist_name] = raw_wrist_rotation.as_quat(
            scalar_first=True
        )

    frames: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    for frame_index in range(episode.frame_count):
        frame = _copy_human_frame(standing)
        for human_name in GMR_UPPER_BODY:
            frame[human_name] = (
                upper_positions[human_name][frame_index],
                upper_quaternions[human_name][frame_index],
            )
        frames.append(frame)
    return frames


def _direction_alignments(
    neutral_direction: np.ndarray,
    target_directions: np.ndarray,
) -> Rotation:
    source = np.asarray(neutral_direction, dtype=np.float64)
    source /= np.linalg.norm(source)
    targets = np.asarray(target_directions, dtype=np.float64)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    rotations = [
        Rotation.align_vectors(target[None], source[None])[0].as_rotvec()
        for target in targets
    ]
    return Rotation.from_rotvec(np.asarray(rotations))


def _gmr_body_trajectory(
    episode: EgoDexEpisode,
    model: pin.Model,
) -> tuple[
    GMR,
    np.ndarray,
    np.ndarray,
    O6HandRetargetTrajectory,
    float,
]:
    """Solve EgoDex with the same 29-DoF GMR body path as Nymeria."""

    retarget = GMR(
        actual_human_height=None,
        src_human="smplx",
        tgt_robot="unitree_g1",
        verbose=False,
    )
    hand_scale = float(
        np.mean(
            [
                retarget.human_scale_table["left_wrist"],
                retarget.human_scale_table["right_wrist"],
            ]
        )
    )
    hand = retarget_o6_hands(
        model,
        episode.hdf5_path,
        scale=hand_scale,
    )
    frames = _egodex_gmr_frames(
        episode,
        _standing_gmr_frame(retarget),
        retarget,
        hand,
    )
    body_qpos = np.empty(
        (episode.frame_count, retarget.model.nq),
        dtype=np.float64,
    )
    task_errors = np.empty(episode.frame_count, dtype=np.float64)
    for frame_index, human_frame in enumerate(frames):
        body_qpos[frame_index] = retarget.retarget(human_frame)
        task_errors[frame_index] = float(retarget.error2())
    return (
        retarget,
        body_qpos,
        task_errors,
        hand,
        float(retarget.human_scale_table["spine3"]),
    )



def _merge_body_and_hands(
    model: pin.Model,
    retarget: GMR,
    body_qpos: np.ndarray,
    hand_joint_names: tuple[str, ...],
    hand_joint_positions: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    state, joint_names, _, open_hand_names = merge_body_with_open_hands(
        model,
        retarget,
        body_qpos,
    )
    if set(open_hand_names) != set(hand_joint_names):
        raise ValueError("EgoDex O6 joints differ from the G1 hand joints")
    hand_index = {name: index for index, name in enumerate(hand_joint_names)}
    for joint_name in open_hand_names:
        joint_id = model.getJointId(joint_name)
        state[:, model.idx_qs[joint_id]] = hand_joint_positions[
            :,
            hand_index[joint_name],
        ]
    if np.any(state < model.lowerPositionLimit - 1e-6) or np.any(
        state > model.upperPositionLimit + 1e-6
    ):
        raise ValueError("Retargeted trajectory exceeds the 51-DoF URDF limits")
    return state.astype(np.float32), joint_names



def retarget_episode(
    episode: EgoDexEpisode,
    model: pin.Model,
) -> tuple[RetargetedEpisode, list[str]]:
    (
        retarget,
        body_qpos,
        task_errors,
        hand,
        scene_scale,
    ) = _gmr_body_trajectory(episode, model)
    state, joint_names = _merge_body_and_hands(
        model,
        retarget,
        body_qpos,
        hand.joint_names,
        hand.joint_positions,
    )
    action_indices = np.minimum(
        np.arange(episode.frame_count) + ACTION_OFFSET,
        episode.frame_count - 1,
    )
    action = state[action_indices].copy()
    root_pose = np.concatenate(
        [
            body_qpos[:, :3],
            body_qpos[:, 3:7][:, [1, 2, 3, 0]],
        ],
        axis=1,
    ).astype(np.float32)
    robot_torso_poses_world = robot_frame_poses_world(
        urdf_path=G1_URDF,
        joint_names=tuple(joint_names),
        joint_positions=state,
        root_pose_xyzw=root_pose,
        frame_name=HEAD_CAMERA_PARENT_FRAME,
    )
    (
        camera_pose,
        source_camera_pose_in_spine3,
        source_camera_rotation_world,
        source_spine3_to_robot_torso_rotation,
    ) = retarget_scene_camera_poses_cv(
        source_camera_to_world=episode.camera_to_world,
        source_spine3_to_world=episode.source_spine3_to_world,
        source_to_robot_world_rotation=episode.source_to_robot_world_rotation,
        robot_torso_pose_world_xyzw=robot_torso_poses_world,
        scene_scale=scene_scale,
    )
    intrinsic = np.repeat(
        episode.intrinsic.reshape(1, 9),
        episode.frame_count,
        axis=0,
    ).astype(np.float32)
    joint_step = (
        np.abs(np.diff(state.astype(np.float64), axis=0))
        if len(state) > 1
        else np.zeros((1, state.shape[1]), dtype=np.float64)
    )
    hand_diagnostics = dict(hand.diagnostics)
    hand_diagnostics["target_adapter"] = {
        "body_retargeter": "unitree_g1 (same as Nymeria)",
        "source_action_mode": hand.action_mode,
        "wrist_rotation_source": "tracked EgoDex Hand orientation",
        "wrist_position_source": "tracked EgoDex Hand wrist-root position",
    }
    return (
        RetargetedEpisode(
            source=episode,
            state=state,
            action=action,
            root_pose=root_pose,
            camera_pose=camera_pose.astype(np.float32),
            source_camera_pose_in_spine3=(
                source_camera_pose_in_spine3.astype(np.float32)
            ),
            source_camera_rotation_world=(
                source_camera_rotation_world.astype(np.float32)
            ),
            source_to_robot_world_rotation=(
                episode.source_to_robot_world_rotation.astype(np.float64)
            ),
            source_spine3_to_robot_torso_rotation=(
                source_spine3_to_robot_torso_rotation
            ),
            camera_intrinsics=intrinsic,
            scene_scale=float(scene_scale),
            metrics={
                "gmr_task_error_mean": float(np.mean(task_errors)),
                "gmr_task_error_max": float(np.max(task_errors)),
                "max_joint_step_rad": float(np.max(joint_step)),
                "mean_joint_step_rad": float(np.mean(joint_step)),
            },
            hand_diagnostics=hand_diagnostics,
        ),
        joint_names,
    )



def _numeric_stats(values: np.ndarray) -> dict[str, list[Any]]:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or len(array) == 0:
        raise ValueError(f"Expected a non-empty [N,D] array, got {array.shape}")
    float_array = array.astype(np.float64)
    stats: dict[str, list[Any]] = {
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(float_array, axis=0).tolist(),
        "std": np.std(float_array, axis=0).tolist(),
        "count": [int(len(array))],
    }
    for quantile in STAT_QUANTILES:
        stats[f"q{int(quantile * 100):02d}"] = np.quantile(
            float_array,
            quantile,
            axis=0,
        ).tolist()
    return stats


def _video_stats(
    path: Path,
    frame_count: int,
    sample_count: int,
) -> dict[str, list[Any]]:
    if sample_count <= 0:
        raise ValueError("Image statistic sample count must be positive")
    wanted = set(
        np.linspace(
            0,
            frame_count - 1,
            num=min(frame_count, sample_count),
        ).round().astype(int).tolist()
    )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path} for image statistics")

    samples: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        if index in wanted:
            height, width = frame_bgr.shape[:2]
            scale = min(1.0, 256.0 / max(height, width))
            if scale < 1.0:
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            samples.append(
                frame_rgb.reshape(-1, 3).astype(np.float32) / 255.0
            )
        index += 1
    capture.release()
    if index != frame_count:
        raise ValueError(
            f"OpenCV decoded {index} frames from {path}, expected {frame_count}"
        )
    if len(samples) != len(wanted):
        raise ValueError(
            f"Decoded {len(samples)} requested statistic frames from {path}, "
            f"expected {len(wanted)}"
        )

    pixels = np.concatenate(samples, axis=0).astype(np.float64)

    def image_shape(values: np.ndarray) -> list[list[list[float]]]:
        return values.reshape(3, 1, 1).tolist()

    stats: dict[str, Any] = {
        "min": image_shape(np.min(pixels, axis=0)),
        "max": image_shape(np.max(pixels, axis=0)),
        "mean": image_shape(np.mean(pixels, axis=0)),
        "std": image_shape(np.std(pixels, axis=0)),
        "count": [len(samples)],
    }
    for quantile in STAT_QUANTILES:
        stats[f"q{int(quantile * 100):02d}"] = image_shape(
            np.quantile(pixels, quantile, axis=0)
        )
    return stats


def _aggregate_feature_stats(
    feature_stats: list[dict[str, list[Any]]],
) -> dict[str, list[Any]]:
    counts = np.asarray(
        [float(stats["count"][0]) for stats in feature_stats],
        dtype=np.float64,
    )
    total_count = float(np.sum(counts))
    means = np.stack(
        [np.asarray(stats["mean"], dtype=np.float64) for stats in feature_stats]
    )
    variances = np.stack(
        [
            np.square(np.asarray(stats["std"], dtype=np.float64))
            for stats in feature_stats
        ]
    )
    weight_shape = (len(counts),) + (1,) * (means.ndim - 1)
    weights = counts.reshape(weight_shape)
    total_mean = np.sum(means * weights, axis=0) / total_count
    total_variance = np.sum(
        (variances + np.square(means - total_mean)) * weights,
        axis=0,
    ) / total_count
    result: dict[str, Any] = {
        "min": np.min(
            np.stack(
                [np.asarray(stats["min"]) for stats in feature_stats]
            ),
            axis=0,
        ).tolist(),
        "max": np.max(
            np.stack(
                [np.asarray(stats["max"]) for stats in feature_stats]
            ),
            axis=0,
        ).tolist(),
        "mean": total_mean.tolist(),
        "std": np.sqrt(np.maximum(total_variance, 0.0)).tolist(),
        "count": [int(total_count)],
    }
    for key in ("q01", "q10", "q50", "q90", "q99"):
        quantiles = np.stack(
            [
                np.asarray(stats[key], dtype=np.float64)
                for stats in feature_stats
            ]
        )
        result[key] = (
            np.sum(quantiles * weights, axis=0) / total_count
        ).tolist()
    return result


def aggregate_stats(
    episode_stats: list[dict[str, dict[str, list[Any]]]],
) -> dict[str, dict[str, list[Any]]]:
    feature_names = sorted(
        {feature for stats in episode_stats for feature in stats}
    )
    return {
        feature: _aggregate_feature_stats(
            [stats[feature] for stats in episode_stats if feature in stats]
        )
        for feature in feature_names
    }


def _episode_columns(
    episode: RetargetedEpisode,
    episode_index: int,
    task_index: int,
    global_from_index: int,
) -> dict[str, np.ndarray]:
    length = episode.source.frame_count
    return {
        "observation.state": episode.state,
        "action": episode.action,
        "observation.root_pose": episode.root_pose,
        "observation.camera_pose": episode.camera_pose,
        "observation.camera_intrinsics": episode.camera_intrinsics,
        "observation.source_camera_pose_in_spine3_xyzw": (
            episode.source_camera_pose_in_spine3
        ),
        "observation.source_camera_rotation_world_xyzw": (
            episode.source_camera_rotation_world
        ),
        "timestamp": (
            np.arange(length, dtype=np.float32)
            / np.float32(episode.source.video.fps)
        ),
        "frame_index": np.arange(length, dtype=np.int64),
        "episode_index": np.full(length, episode_index, dtype=np.int64),
        "index": np.arange(
            global_from_index,
            global_from_index + length,
            dtype=np.int64,
        ),
        "task_index": np.full(length, task_index, dtype=np.int64),
    }


def _arrow_column(values: np.ndarray) -> pa.Array:
    values = np.asarray(values)
    if values.ndim == 1:
        if np.issubdtype(values.dtype, np.floating):
            return pa.array(values, type=pa.float32())
        if np.issubdtype(values.dtype, np.integer):
            return pa.array(values, type=pa.int64())
        raise TypeError(f"Unsupported scalar dtype: {values.dtype}")
    if values.ndim == 2:
        if np.issubdtype(values.dtype, np.floating):
            value_type = pa.float32()
            normalized = values.astype(np.float32, copy=False)
        elif np.issubdtype(values.dtype, np.integer):
            value_type = pa.int64()
            normalized = values.astype(np.int64, copy=False)
        else:
            raise TypeError(f"Unsupported vector dtype: {values.dtype}")
        return pa.array(
            normalized.tolist(),
            type=pa.list_(value_type, list_size=values.shape[1]),
        )
    raise ValueError(f"Only scalar/vector columns are supported, got {values.shape}")


def _write_data_parquet(columns: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [_arrow_column(values) for values in columns.values()],
        names=list(columns),
    )
    pq.write_table(
        table,
        path,
        compression="snappy",
        use_dictionary=True,
    )


def _feature_schema(
    joint_names: list[str],
    video: VideoInfo,
) -> dict[str, dict[str, Any]]:
    pose_names = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    intrinsic_names = [
        "k00",
        "k01",
        "k02",
        "k10",
        "k11",
        "k12",
        "k20",
        "k21",
        "k22",
    ]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(joint_names)],
            "names": joint_names,
            "fps": int(round(video.fps)),
        },
        "action": {
            "dtype": "float32",
            "shape": [len(joint_names)],
            "names": joint_names,
            "fps": int(round(video.fps)),
        },
        "observation.root_pose": {
            "dtype": "float32",
            "shape": [7],
            "names": pose_names,
            "fps": int(round(video.fps)),
        },
        "observation.camera_pose": {
            "dtype": "float32",
            "shape": [7],
            "names": pose_names,
            "fps": int(round(video.fps)),
        },
        "observation.camera_intrinsics": {
            "dtype": "float32",
            "shape": [9],
            "names": intrinsic_names,
            "fps": int(round(video.fps)),
        },
        "observation.source_camera_pose_in_spine3_xyzw": {
            "dtype": "float32",
            "shape": [7],
            "names": pose_names,
            "fps": int(round(video.fps)),
        },
        "observation.source_camera_rotation_world_xyzw": {
            "dtype": "float32",
            "shape": [4],
            "names": ["qx", "qy", "qz", "qw"],
            "fps": int(round(video.fps)),
        },
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [video.height, video.width, 3],
            "names": ["height", "width", "channels"],
            "fps": int(round(video.fps)),
            "info": {
                "video.height": video.height,
                "video.width": video.width,
                "video.codec": video.codec,
                "video.pix_fmt": video.pixel_format,
                "video.fps": int(round(video.fps)),
                "video.channels": 3,
                "has_audio": video.has_audio,
                "video.is_depth_map": False,
            },
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
    }


def _flatten_episode_stats(
    row: dict[str, Any],
    stats: dict[str, dict[str, list[Any]]],
) -> None:
    for feature, feature_stats in stats.items():
        for stat_name in STAT_KEYS:
            row[f"stats/{feature}/{stat_name}"] = feature_stats[stat_name]


def _commit_output(temporary_root: Path, output_root: Path) -> None:
    """Atomically replace output_root, restoring it if the swap fails."""
    backup_root: Path | None = None
    if output_root.exists() or output_root.is_symlink():
        backup_root = output_root.parent / (
            f".{output_root.name}.backup-{os.getpid()}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        output_root.rename(backup_root)
    try:
        temporary_root.rename(output_root)
    except Exception:
        if backup_root is not None:
            backup_root.rename(output_root)
        raise
    else:
        if backup_root is not None:
            remove_exact(backup_root)


def write_lerobot_v3(
    retargeted: list[RetargetedEpisode],
    joint_names: list[str],
    output_root: Path,
    overwrite: bool = True,
) -> None:
    if not retargeted:
        raise ValueError("No retargeted episodes to write")
    if (output_root.exists() or output_root.is_symlink()) and not overwrite:
        raise FileExistsError(
            f"Output already exists and --no-overwrite was requested: "
            f"{output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = output_root.parent / (
        f".{output_root.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    temporary_root.mkdir(parents=False, exist_ok=False)

    task_to_index: dict[str, int] = {}
    for episode in retargeted:
        if episode.source.task not in task_to_index:
            task_to_index[episode.source.task] = len(task_to_index)

    first_video = retargeted[0].source.video
    features = _feature_schema(joint_names, first_video)
    episode_rows: list[dict[str, Any]] = []
    all_episode_stats: list[dict[str, dict[str, list[Any]]]] = []
    manifest_episodes: list[dict[str, Any]] = []
    global_index = 0

    try:
        for episode_index, episode in enumerate(retargeted):
            source = episode.source
            task_index = task_to_index[source.task]
            chunk_index = episode_index // CHUNKS_SIZE
            file_index = episode_index % CHUNKS_SIZE
            columns = _episode_columns(
                episode,
                episode_index=episode_index,
                task_index=task_index,
                global_from_index=global_index,
            )
            data_relative = DATA_PATH_TEMPLATE.format(
                chunk_index=chunk_index,
                file_index=file_index,
            )
            _write_data_parquet(columns, temporary_root / data_relative)

            video_relative = VIDEO_PATH_TEMPLATE.format(
                video_key=VIDEO_KEY,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            video_destination = temporary_root / video_relative
            video_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.video_path, video_destination)

            stats = {
                feature: _numeric_stats(values)
                for feature, values in columns.items()
            }
            stats[VIDEO_KEY] = _video_stats(
                source.video_path,
                source.frame_count,
                sample_count=IMAGE_STAT_SAMPLES,
            )
            all_episode_stats.append(stats)

            length = source.frame_count
            row: dict[str, Any] = {
                "episode_index": episode_index,
                "tasks": [source.task],
                "length": length,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + length,
                f"videos/{VIDEO_KEY}/chunk_index": chunk_index,
                f"videos/{VIDEO_KEY}/file_index": file_index,
                f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
                f"videos/{VIDEO_KEY}/to_timestamp": (
                    length / source.video.fps
                ),
            }
            _flatten_episode_stats(row, stats)
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0
            episode_rows.append(row)

            manifest_episodes.append(
                {
                    "episode_index": episode_index,
                    "source_stem": source.stem,
                    "source_hdf5": str(source.hdf5_path),
                    "source_video": str(source.video_path),
                    "task": source.task,
                    "frame_count": length,
                    "scene_scale": episode.scene_scale,
                    "scene_scale_source": "gmr_spine3_scale",
                    "source_spine3_to_robot_torso_rotation": (
                        episode.source_spine3_to_robot_torso_rotation.tolist()
                    ),
                    "source_to_robot_world_rotation": (
                        episode.source_to_robot_world_rotation.tolist()
                    ),
                    "local_basis_source": (
                        "frame0_source_spine3_axes_mapped_through_initial_"
                        "egodex_hip_world_basis_into_g1_torso_fk"
                    ),
                    "local_basis_reference_frame": 0,
                    "scene_camera_position_anchor": "g1_torso_fk",
                    "scene_camera_rotation_anchor": "raw_source_world",
                    "retarget_metrics": episode.metrics,
                    "hand_retarget": episode.hand_diagnostics,
                }
            )
            global_index += length

        episodes_path = temporary_root / EPISODES_PATH_TEMPLATE.format(
            chunk_index=0,
            file_index=0,
        )
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(episode_rows),
            episodes_path,
            compression="snappy",
            use_dictionary=True,
        )

        task_frame = pd.DataFrame(
            {"task_index": list(task_to_index.values())},
            index=pd.Index(list(task_to_index), name="task"),
        )
        task_path = temporary_root / "meta/tasks.parquet"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_frame.to_parquet(task_path)

        info = {
            "codebase_version": "v3.0",
            "robot_type": "unitree_g1_with_hands",
            "total_episodes": len(retargeted),
            "total_frames": global_index,
            "total_tasks": len(task_to_index),
            "chunks_size": CHUNKS_SIZE,
            "data_files_size_in_mb": DATA_FILE_SIZE_MB,
            "video_files_size_in_mb": VIDEO_FILE_SIZE_MB,
            "fps": int(round(first_video.fps)),
            "splits": {"train": f"0:{len(retargeted)}"},
            "data_path": DATA_PATH_TEMPLATE,
            "video_path": VIDEO_PATH_TEMPLATE,
            "features": features,
        }
        write_json_atomic(temporary_root / "meta/info.json", info)
        write_json_atomic(
            temporary_root / "meta/stats.json",
            aggregate_stats(all_episode_stats),
        )
        write_json_atomic(
            temporary_root / "meta/egodex.json",
            {
                "schema_version": 10,
                "source_format": "EgoDex annotated HDF5 + MP4",
                "robot": "unitree_g1_with_hands",
                "retargeter": "general_motion_retargeting",
                "body_solver": "unitree_g1 (same as Nymeria)",
                "body_config": "smplx_to_g1.json",
                "lower_body_target": "G1 neutral standing FK",
                "hand_retargeter": (
                    "tracked Hand pose to GMR wrist target plus 25-landmark "
                    "articulated finger mapping"
                ),
                "hand_target_adapter": (
                    "tracked Hand pose -> fixed side-specific local-axis map "
                    "-> GMR wrist target"
                ),
                "robot_urdf": str(G1_URDF),
                "joint_order": joint_names,
                "action_offset_frames": ACTION_OFFSET,
                "camera": (
                    "source camera world rotation with XYZ anchored to G1 "
                    "torso FK; G1 capture camera remains independent"
                ),
                "episodes": manifest_episodes,
            },
        )

        _commit_output(temporary_root, output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise



def _validate_uniform_video_schema(episodes: list[EgoDexEpisode]) -> None:
    first = episodes[0].video
    if not np.isclose(first.fps, round(first.fps), atol=1e-6):
        raise ValueError(
            f"LeRobot v3 recording FPS must be integral; got {first.fps}"
        )
    for episode in episodes[1:]:
        video = episode.video
        fields = (
            video.width == first.width,
            video.height == first.height,
            np.isclose(video.fps, first.fps, atol=1e-6),
            video.codec == first.codec,
            video.pixel_format == first.pixel_format,
            video.has_audio == first.has_audio,
        )
        if not all(fields):
            raise ValueError(
                "All source videos must share resolution, FPS, codec, pixel "
                f"format, and audio layout. {episode.stem} differs from the "
                "first episode."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget EgoDex tracks directly to a 51-DoF Unitree G1 and "
            "write frame-aligned LeRobot v3."
        )
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=WORKSPACE_ROOT / "dataset/basic_pick_place",
        help="Directory recursively containing same-stem .hdf5 and .mp4 files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "dataset/basic_pick_place_g1_lerobot_v3",
        help="LeRobot v3 output directory; overwritten by default.",
    )
    parser.add_argument(
        "--stem",
        help="Convert only this exact source stem relative to --src-root.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace an existing output dataset (default: enabled).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not G1_URDF.is_file():
        raise FileNotFoundError(f"G1 URDF not found: {G1_URDF}")
    model = build_pinocchio_g1(G1_URDF)

    pairs = discover_source_pairs(args.src_root.resolve())
    if args.stem is not None:
        pairs = [pair for pair in pairs if pair.stem == args.stem]
        if len(pairs) != 1:
            raise ValueError(f"EgoDex stem {args.stem!r} matched {len(pairs)} episodes")
    episodes = [
        load_egodex_episode(pair)
        for pair in pairs
    ]
    _validate_uniform_video_schema(episodes)

    retargeted: list[RetargetedEpisode] = []
    joint_names: list[str] | None = None
    for episode in episodes:
        converted, episode_joint_names = retarget_episode(
            episode,
            model=model,
        )
        if joint_names is None:
            joint_names = episode_joint_names
        elif joint_names != episode_joint_names:
            raise ValueError("Robot joint order changed between episodes")
        retargeted.append(converted)
        print(
            f"{episode.stem}: frames={episode.frame_count}, "
            f"GMR_error={converted.metrics['gmr_task_error_mean']:.4f}, "
            f"max_joint_step={converted.metrics['max_joint_step_rad']:.4f} rad"
        )

    assert joint_names is not None
    write_lerobot_v3(
        retargeted,
        joint_names=joint_names,
        output_root=args.output_root.resolve(),
        overwrite=args.overwrite,
    )
    total_frames = sum(episode.frame_count for episode in episodes)
    print(
        f"Wrote {len(episodes)} episodes / {total_frames} frames to "
        f"{args.output_root.resolve()}."
    )


if __name__ == "__main__":
    main()
