#!/usr/bin/env python3
"""Retarget EgoDex directly to G1 and write LeRobot v3.

The input is a directory of same-stem EgoDex ``.hdf5``/``.mp4`` pairs.  For
each episode this script:

1. maps the tracked body and hand landmarks into a canonical Z-up frame,
2. combines it with a static G1-derived standing target,
3. builds GMR from the same complete G1+O6 URDF used by the output,
4. solves the floating-base body trajectory with all 22 O6 hand joints while
   auditing, but not constraining, cross-hand collisions, and
5. writes 51-DoF references, camera calibration, and frame-aligned RGB.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import mink
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
from human_to_robot.retarget_g1_pinocchio import (  # noqa: E402
    HAND_JOINTS,
    O6HandRetargetTrajectory,
    build_pinocchio_g1,
    retarget_o6_hands,
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
IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
G1_URDF = (
    WORKSPACE_ROOT
    / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
)
SMOOTH_WINDOW = 7
ACTION_OFFSET = 1
IMAGE_STAT_SAMPLES = 24
CROSS_HAND_COLLISION_TOLERANCE_M = 1e-4
CROSS_HAND_NEAR_CLEARANCE_M = 2e-3
GMR_COLLISION_PHASE1_MAX_ITERATIONS = 3
GMR_COLLISION_PHASE2_MAX_ITERATIONS = 5
GMR_NEUTRAL_PELVIS_HEIGHT_M = 0.793
GMR_TOE_FRAME_OFFSET_M = np.array([0.1, 0.0, -0.02], dtype=np.float64)
GMR_MODEL_KIND = "main_with_hand_new.urdf:floating_o6"
HAND_COLLISION_LINK_PARTS = (
    "hand_base",
    "index",
    "middle",
    "ring",
    "pinky",
    "thumb",
)
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
    points: dict[str, np.ndarray]
    source_to_robot_rotation: np.ndarray

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
    camera_intrinsics: np.ndarray
    source_to_robot_scale: float
    metrics: dict[str, float | int]
    hand_diagnostics: dict[str, dict[str, object]]


@dataclass(frozen=True)
class CrossHandCollisionChecker:
    geometry_model: pin.GeometryModel
    pair_names: tuple[tuple[str, str], ...]
    palm_pair_index: int


def _is_o6_hand_body(body_name: str, side: str) -> bool:
    return body_name == f"{side}_hand_base_link" or any(
        body_name.startswith(f"{side}_{part}")
        for part in ("thumb", "index", "middle", "ring", "pinky")
    )


def _name_o6_collision_geometries(spec: mj.MjSpec) -> None:
    """Give the imported URDF hand geoms stable names for Mink constraints."""

    counts = {"left": 0, "right": 0}
    for body in spec.bodies:
        for side in ("left", "right"):
            if not _is_o6_hand_body(body.name, side):
                continue
            if len(body.geoms) != 1:
                raise ValueError(
                    f"Expected one collision geom on {body.name}, found "
                    f"{len(body.geoms)}"
                )
            body.geoms[0].name = f"{body.name}_collision"
            counts[side] += 1
    if counts != {"left": 12, "right": 12}:
        raise ValueError(
            "Complete O6 URDF must provide 12 collision geoms per hand; "
            f"found {counts}"
        )


def _restore_right_o6_meshes(spec: mj.MjSpec, urdf_path: Path) -> None:
    """Undo MuJoCo URDF asset deduplication for same-named L/R hand STLs."""

    right_mesh_root = (
        urdf_path.parent.parent / "hand/right_hand/meshes"
    ).resolve()
    existing_meshes = {mesh.name: mesh for mesh in spec.meshes}
    restored = 0
    for body in spec.bodies:
        if not _is_o6_hand_body(body.name, "right"):
            continue
        if len(body.geoms) != 1:
            raise ValueError(
                f"Expected one right-hand collision geom on {body.name}"
            )
        geom = body.geoms[0]
        source_asset = existing_meshes.get(geom.meshname)
        if source_asset is None:
            raise ValueError(
                f"Missing imported mesh asset {geom.meshname!r} for {body.name}"
            )
        right_mesh = right_mesh_root / Path(source_asset.file).name
        if not right_mesh.is_file():
            raise FileNotFoundError(
                f"Missing right O6 collision mesh: {right_mesh}"
            )
        unique_name = f"{body.name}_collision_mesh"
        spec.add_mesh(name=unique_name, file=str(right_mesh))
        geom.meshname = unique_name
        restored += 1
    if restored != 12:
        raise ValueError(
            f"Expected to restore 12 right O6 meshes, restored {restored}"
        )


def _build_gmr_with_hand_model(urdf_path: Path) -> mj.MjModel:
    """Compile the complete G1+O6 URDF into GMR's floating-base model."""

    resolved = urdf_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"G1+O6 URDF not found: {resolved}")

    spec = mj.MjSpec.from_file(str(resolved))
    # MuJoCo otherwise fuses fixed links such as hand_base_link into the wrist,
    # which prevents the actual O6 frames and geometry from entering GMR.
    spec.compiler.fusestatic = False
    pelvis = spec.body("pelvis")
    if pelvis is None:
        raise ValueError(f"{resolved} has no pelvis root body")
    if pelvis.joints:
        raise ValueError("G1+O6 URDF pelvis must be fixed before adding free root")
    pelvis.pos = [0.0, 0.0, GMR_NEUTRAL_PELVIS_HEIGHT_M]
    pelvis.add_freejoint(name="pelvis")

    # smplx_to_g1.json targets these operation frames. They are present in the
    # upstream GMR MJCF but not represented as links in the complete URDF.
    for side in ("left", "right"):
        toe_name = f"{side}_toe_link"
        if spec.body(toe_name) is not None:
            raise ValueError(f"Unexpected duplicate GMR frame {toe_name}")
        ankle = spec.body(f"{side}_ankle_roll_link")
        if ankle is None:
            raise ValueError(f"{resolved} has no {side} ankle-roll body")
        ankle.add_body(name=toe_name, pos=GMR_TOE_FRAME_OFFSET_M)

    _restore_right_o6_meshes(spec, resolved)
    _name_o6_collision_geometries(spec)
    model = spec.compile()

    expected_dimensions = (58, 57, 52)
    actual_dimensions = (model.nq, model.nv, model.njnt)
    if actual_dimensions != expected_dimensions:
        raise ValueError(
            "Unexpected complete G1+O6 MuJoCo dimensions: "
            f"expected nq/nv/njnt={expected_dimensions}, got "
            f"{actual_dimensions}"
        )
    hand_mesh_ids: dict[str, set[int]] = {}
    for side in ("left", "right"):
        mesh_ids = {
            int(model.geom_dataid[geom_id])
            for geom_id in range(model.ngeom)
            if (
                (
                    body_name := mj.mj_id2name(
                        model,
                        mj.mjtObj.mjOBJ_BODY,
                        int(model.geom_bodyid[geom_id]),
                    )
                )
                is not None
                and _is_o6_hand_body(body_name, side)
            )
        }
        if len(mesh_ids) != 12 or -1 in mesh_ids:
            raise ValueError(
                f"Complete G1+O6 model must retain 12 distinct {side} "
                f"hand collision meshes, found {sorted(mesh_ids)}"
            )
        hand_mesh_ids[side] = mesh_ids
    if hand_mesh_ids["left"] & hand_mesh_ids["right"]:
        raise ValueError(
            "Complete G1+O6 model reused collision mesh assets across hands"
        )
    root_joint = mj.mj_name2id(
        model,
        mj.mjtObj.mjOBJ_JOINT,
        "pelvis",
    )
    if (
        root_joint < 0
        or model.jnt_type[root_joint] != mj.mjtJoint.mjJNT_FREE
        or model.jnt_qposadr[root_joint] != 0
        or not np.allclose(
            model.qpos0[:7],
            [0.0, 0.0, GMR_NEUTRAL_PELVIS_HEIGHT_M, 1.0, 0.0, 0.0, 0.0],
            atol=1e-9,
        )
    ):
        raise ValueError("Complete G1+O6 GMR model has an invalid floating root")

    missing_bodies = [
        body_name
        for body_name in (
            *GMR_BODY_LINKS.values(),
            "left_hand_base_link",
            "right_hand_base_link",
        )
        if mj.mj_name2id(
            model,
            mj.mjtObj.mjOBJ_BODY,
            body_name,
        )
        < 0
    ]
    missing_joints = [
        joint_name
        for joint_name in HAND_JOINTS
        if mj.mj_name2id(
            model,
            mj.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        < 0
    ]
    if missing_bodies or missing_joints:
        raise ValueError(
            "Complete G1+O6 GMR model is incomplete: "
            f"missing bodies={missing_bodies}, missing joints={missing_joints}"
        )
    return model


def _configure_gmr_hand_constraints(
    retarget: GMR,
    hand_joint_names: tuple[str, ...],
) -> tuple[dict[str, object], tuple[tuple[int, int], ...]]:
    """Freeze the retargeted fingers without constraining the two hands."""

    if tuple(hand_joint_names) != tuple(HAND_JOINTS):
        raise ValueError("O6 hand joint order differs from the complete URDF")
    retarget.ik_limits.append(
        mink.VelocityLimit(
            retarget.model,
            {joint_name: 0.0 for joint_name in hand_joint_names},
        )
    )
    diagnostics = {
        "model_kind": GMR_MODEL_KIND,
        "source_urdf": str(G1_URDF.resolve()),
        "floating_root": True,
        "fixed_link_fusion": False,
        "hand_joint_count": len(hand_joint_names),
        "cross_hand_geometry_pairs": 0,
        "cross_hand_collision_avoidance": False,
        "finger_dofs_frozen_during_body_ik": True,
        "collision_solver": "disabled",
        "post_gmr_arm_refinement": False,
        "phase_iterations": [
            GMR_COLLISION_PHASE1_MAX_ITERATIONS,
            GMR_COLLISION_PHASE2_MAX_ITERATIONS,
        ],
    }
    return diagnostics, ()


def _solve_complete_gmr_phase(
    retarget: GMR,
    tasks: list[mink.FrameTask],
    *,
    iterations: int,
) -> None:
    """Run a fixed number of constrained differential-IK iterations."""

    for _ in range(iterations):
        dt = retarget.configuration.model.opt.timestep
        velocity = mink.solve_ik(
            retarget.configuration,
            tasks,
            dt,
            retarget.solver,
            damping=retarget.damping,
            limits=retarget.ik_limits,
        )
        retarget.configuration.integrate_inplace(velocity, dt)


def _complete_gmr_retarget(
    retarget: GMR,
    human_frame: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Solve both GMR phases with frozen O6 joints."""

    retarget.update_targets(human_frame)
    if retarget.use_ik_match_table1:
        _solve_complete_gmr_phase(
            retarget,
            retarget.tasks1,
            iterations=GMR_COLLISION_PHASE1_MAX_ITERATIONS,
        )
    if retarget.use_ik_match_table2:
        _solve_complete_gmr_phase(
            retarget,
            retarget.tasks2,
            iterations=GMR_COLLISION_PHASE2_MAX_ITERATIONS,
        )
    return retarget.configuration.data.qpos.copy()


def _set_gmr_hand_configuration(
    retarget: GMR,
    hand: O6HandRetargetTrajectory,
    frame_index: int,
) -> None:
    """Inject the exact O6 state whose geometry GMR must avoid."""

    values = np.asarray(hand.joint_positions[frame_index], dtype=np.float64)
    if values.shape != (len(hand.joint_names),) or not np.isfinite(values).all():
        raise ValueError(f"Invalid O6 joint state at frame {frame_index}")
    for joint_name, value in zip(hand.joint_names, values, strict=True):
        joint_id = mj.mj_name2id(
            retarget.model,
            mj.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise KeyError(f"GMR model is missing O6 joint {joint_name!r}")
        qpos_address = int(retarget.model.jnt_qposadr[joint_id])
        retarget.configuration.data.qpos[qpos_address] = value
    mj.mj_forward(retarget.model, retarget.configuration.data)


def _validate_o6_mimic_trajectory(
    model: pin.Model,
    hand: O6HandRetargetTrajectory,
) -> dict[str, object]:
    """Verify the URDF mimic equations that MuJoCo does not import."""

    mimic_joints = getattr(model, "vision_align_mimic_joints", None)
    if not isinstance(mimic_joints, dict) or len(mimic_joints) != 10:
        raise ValueError(
            "Complete G1+O6 Pinocchio model must expose all 10 URDF mimic "
            "joint equations"
        )
    joint_index = {
        joint_name: index
        for index, joint_name in enumerate(hand.joint_names)
    }
    maximum_residual = 0.0
    for child, (master, multiplier, offset) in mimic_joints.items():
        if child not in joint_index or master not in joint_index:
            raise ValueError(
                f"O6 hand trajectory is missing mimic pair {master!r} -> "
                f"{child!r}"
            )
        child_values = hand.joint_positions[:, joint_index[child]]
        master_values = hand.joint_positions[:, joint_index[master]]
        residual = child_values - (
            float(multiplier) * master_values + float(offset)
        )
        maximum_residual = max(
            maximum_residual,
            float(np.max(np.abs(residual))),
        )
    if maximum_residual > 1e-9:
        raise ValueError(
            "Retargeted O6 trajectory violates URDF mimic equations; "
            f"maximum residual={maximum_residual:.3e} rad"
        )
    return {
        "source": "main_with_hand_new.urdf",
        "equation_count": len(mimic_joints),
        "enforcement": "validated input plus frozen GMR hand DoFs",
        "maximum_residual_rad": maximum_residual,
    }


def _natural_key(value: str) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


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
        key=lambda path: _natural_key(str(path.relative_to(src_root.resolve()))),
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
        if "camera/intrinsic" not in handle or "transforms/camera" not in handle:
            raise KeyError(
                f"{pair.hdf5_path} must contain camera/intrinsic and "
                "transforms/camera"
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
    # Reorder those columns into the G1 convention (+X forward, +Y left,
    # +Z up), average the episode frame on SO(3), then invert it.
    assert hip_rotation is not None
    canonical_to_source = np.stack(
        [
            hip_rotation[:, :, 2],
            hip_rotation[:, :, 0],
            hip_rotation[:, :, 1],
        ],
        axis=2,
    )
    u, _, vt = np.linalg.svd(np.mean(canonical_to_source, axis=0))
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
        points=points,
        source_to_robot_rotation=source_to_robot,
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
    if (
        hand.source_action_transforms_reference_camera.shape
        != (episode.frame_count, 2, 4, 4)
        or hand.wrist_from_action_transforms.shape
        != (episode.frame_count, 2, 4, 4)
    ):
        raise ValueError(
            f"Hand action trajectory length differs for {episode.stem}"
        )

    basis = episode.source_to_robot_rotation
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

        # EgoDex's tracked Hand transform is the wrist root. For articulated
        # input this pipeline deliberately derives a fingertip grasp action
        # frame, then removes the state-dependent O6 wrist-to-action transform.
        # This preserves the source task point without pretending that the
        # fingertips and wrist_yaw_link share an origin.
        source_action = (
            hand.source_action_transforms_reference_camera[:, side_index]
        )
        source_action_position = (
            reference_camera_to_source[:3, 3]
            + (
                reference_camera_to_source[:3, :3]
                @ source_action[:, :3, 3].T
            ).T
        )
        action_rotation = np.einsum(
            "ij,njk->nik",
            robot_from_reference_camera,
            source_action[:, :3, :3],
        )
        wrist_from_action = (
            hand.wrist_from_action_transforms[:, side_index]
        )
        wrist_rotation = np.einsum(
            "nij,nkj->nik",
            action_rotation,
            wrist_from_action[:, :3, :3],
        )
        wrist_scale = float(retarget.human_scale_table[wrist_name])
        standing_shoulder_position = np.asarray(
            retarget.scaled_human_data[shoulder_name][0],
            dtype=np.float64,
        )
        action_position = (
            standing_shoulder_position
            + wrist_scale
            * (basis @ (source_action_position - source_shoulder).T).T
        )
        wrist_position = action_position - np.einsum(
            "nij,nj->ni",
            wrist_rotation,
            wrist_from_action[:, :3, 3],
        )
        wrist_position_offset = retarget.pos_offsets1[wrist_name]
        upper_positions[wrist_name] = (
            raw_root_position
            + (
                wrist_position
                - np.einsum(
                    "nij,j->ni",
                    wrist_rotation,
                    wrist_position_offset,
                )
                + ground_offset
                - scaled_root_position
            )
            / wrist_scale
        )
        upper_quaternions[wrist_name] = (
            Rotation.from_matrix(wrist_rotation)
            * retarget.rot_offsets1[wrist_name].inv()
        ).as_quat(scalar_first=True)

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
    gmr_model: mj.MjModel,
) -> tuple[
    GMR,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    O6HandRetargetTrajectory,
    float,
    dict[str, object],
]:
    retarget = GMR(
        actual_human_height=None,
        # The config key supplies GMR's G1 joint/task schema. The injected
        # model below replaces its legacy 43-DoF hand asset with the complete
        # 51-joint G1+O6 URDF used by this pipeline.
        src_human="smplx",
        tgt_robot="unitree_g1_with_hands",
        verbose=False,
        robot_model=gmr_model,
        robot_model_source=str(G1_URDF.resolve()),
    )
    source_to_robot_scale = float(
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
        scale=source_to_robot_scale,
    )
    mimic_diagnostics = _validate_o6_mimic_trajectory(model, hand)
    (
        gmr_model_diagnostics,
        _,
    ) = _configure_gmr_hand_constraints(
        retarget,
        hand.joint_names,
    )
    gmr_model_diagnostics["mimic_constraints"] = mimic_diagnostics
    standing = _standing_gmr_frame(retarget)
    frames = _egodex_gmr_frames(
        episode,
        standing,
        retarget,
        hand,
    )
    qpos = np.empty(
        (episode.frame_count, retarget.model.nq),
        dtype=np.float64,
    )
    position_errors = np.empty(
        (episode.frame_count, len(GMR_UPPER_BODY)),
        dtype=np.float64,
    )
    task_errors = np.empty(episode.frame_count, dtype=np.float64)
    maximum_hand_solver_drift = 0.0
    for frame_index, human_frame in enumerate(frames):
        _set_gmr_hand_configuration(retarget, hand, frame_index)
        _complete_gmr_retarget(retarget, human_frame)
        for joint_index, joint_name in enumerate(hand.joint_names):
            joint_id = mj.mj_name2id(
                retarget.model,
                mj.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            qpos_address = int(retarget.model.jnt_qposadr[joint_id])
            drift = abs(
                float(retarget.configuration.data.qpos[qpos_address])
                - float(hand.joint_positions[frame_index, joint_index])
            )
            maximum_hand_solver_drift = max(
                maximum_hand_solver_drift,
                drift,
            )
            if drift > 1e-5:
                raise RuntimeError(
                    f"GMR failed to freeze O6 joint {joint_name!r} at frame "
                    f"{frame_index}; drift={drift:.3e} rad"
                )
        # The QP backend may leave sub-microradian feasibility residuals on
        # zero-velocity joints. Reapply the exact retargeted O6 state before
        # recording the frame.
        _set_gmr_hand_configuration(retarget, hand, frame_index)
        qpos[frame_index] = retarget.configuration.data.qpos
        mj.mj_forward(retarget.model, retarget.configuration.data)
        for target_index, human_name in enumerate(GMR_UPPER_BODY):
            body_id = mj.mj_name2id(
                retarget.model,
                mj.mjtObj.mjOBJ_BODY,
                GMR_BODY_LINKS[human_name],
            )
            position_errors[frame_index, target_index] = np.linalg.norm(
                retarget.configuration.data.xpos[body_id]
                - retarget.scaled_human_data[human_name][0]
            )
        task_errors[frame_index] = retarget.error2()
    gmr_model_diagnostics["solver_step_count"] = int(
        episode.frame_count
        * (
            GMR_COLLISION_PHASE1_MAX_ITERATIONS
            + GMR_COLLISION_PHASE2_MAX_ITERATIONS
        )
    )
    gmr_model_diagnostics["maximum_frozen_hand_solver_drift_rad"] = (
        maximum_hand_solver_drift
    )
    return (
        retarget,
        qpos,
        position_errors,
        task_errors,
        hand,
        source_to_robot_scale,
        gmr_model_diagnostics,
    )


def _merge_body_and_hands(
    model: pin.Model,
    retarget: GMR,
    body_qpos: np.ndarray,
    hand_joint_names: tuple[str, ...],
    hand_joint_positions: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    joint_names = list(model.names[1:])
    state = np.repeat(
        pin.neutral(model)[None],
        len(body_qpos),
        axis=0,
    )
    hand_index = {
        name: index for index, name in enumerate(hand_joint_names)
    }
    for joint_name in joint_names:
        robot_joint_id = model.getJointId(joint_name)
        state_index = model.idx_qs[robot_joint_id]
        gmr_joint_id = mj.mj_name2id(
            retarget.model,
            mj.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if gmr_joint_id < 0:
            raise KeyError(f"Complete GMR model is missing joint {joint_name!r}")
        gmr_qpos_index = retarget.model.jnt_qposadr[gmr_joint_id]
        state[:, state_index] = body_qpos[:, gmr_qpos_index]
        if joint_name in hand_index and not np.allclose(
            state[:, state_index],
            hand_joint_positions[:, hand_index[joint_name]],
            atol=1e-8,
        ):
            raise RuntimeError(
                f"Complete GMR state disagrees with O6 retarget for "
                f"{joint_name!r}"
            )

    if np.any(state < model.lowerPositionLimit - 1e-6) or np.any(
        state > model.upperPositionLimit + 1e-6
    ):
        raise ValueError("Retargeted trajectory exceeds the 51-DoF URDF limits")
    return state.astype(np.float32), joint_names


def _build_cross_hand_collision_checker(
    model: pin.Model,
    urdf_path: Path,
) -> CrossHandCollisionChecker:
    geometry_model = pin.buildGeomFromUrdf(
        model,
        str(urdf_path),
        pin.GeometryType.COLLISION,
        None,
        [str(urdf_path.parent)],
    )

    def side_geometry_ids(side: str) -> list[int]:
        result: list[int] = []
        for geometry_id, geometry in enumerate(
            geometry_model.geometryObjects
        ):
            link_name = model.frames[geometry.parentFrame].name
            if not link_name.startswith(f"{side}_"):
                continue
            if any(part in link_name for part in HAND_COLLISION_LINK_PARTS):
                result.append(geometry_id)
        return result

    left_ids = side_geometry_ids("left")
    right_ids = side_geometry_ids("right")
    if len(left_ids) != 12 or len(right_ids) != 12:
        raise ValueError(
            "Expected 12 collision geometries per O6 hand, found "
            f"left={len(left_ids)} and right={len(right_ids)}"
        )

    pair_names: list[tuple[str, str]] = []
    palm_pair_index = -1
    for left_id in left_ids:
        for right_id in right_ids:
            left_name = geometry_model.geometryObjects[left_id].name
            right_name = geometry_model.geometryObjects[right_id].name
            geometry_model.addCollisionPair(
                pin.CollisionPair(left_id, right_id)
            )
            pair_names.append((left_name, right_name))
            if (
                "hand_base_link" in left_name
                and "hand_base_link" in right_name
            ):
                palm_pair_index = len(pair_names) - 1
    if len(pair_names) != 144 or palm_pair_index < 0:
        raise RuntimeError(
            "Failed to construct all 144 cross-hand collision pairs"
        )
    return CrossHandCollisionChecker(
        geometry_model=geometry_model,
        pair_names=tuple(pair_names),
        palm_pair_index=palm_pair_index,
    )


def _cross_hand_collision_validation(
    model: pin.Model,
    checker: CrossHandCollisionChecker,
    state: np.ndarray,
) -> tuple[dict[str, float | int], dict[str, object]]:
    geometry_model = checker.geometry_model
    if len(geometry_model.collisionPairs) != len(checker.pair_names):
        raise RuntimeError("Cross-hand collision pair metadata is inconsistent")

    model_data = model.createData()
    geometry_data = pin.GeometryData(geometry_model)
    collision_frames: list[int] = []
    palm_collision_frames: list[int] = []
    near_clearance_frames: list[int] = []
    maximum_penetration = 0.0
    minimum_signed_distance = np.inf
    worst_frame = -1
    worst_pair = ("", "")

    for frame_index, qpos in enumerate(state.astype(np.float64)):
        pin.computeCollisions(
            model,
            model_data,
            geometry_model,
            geometry_data,
            qpos,
            False,
        )
        frame_penetration = 0.0
        frame_worst_pair = ("", "")
        palm_penetration = 0.0
        for pair_index, result in enumerate(
            geometry_data.collisionResults
        ):
            if not result.isCollision():
                continue
            penetration = max(
                (
                    float(contact.penetration_depth)
                    for contact in result.getContacts()
                ),
                default=0.0,
            )
            if penetration > frame_penetration:
                frame_penetration = penetration
                frame_worst_pair = checker.pair_names[pair_index]
            if pair_index == checker.palm_pair_index:
                palm_penetration = max(palm_penetration, penetration)

        pin.computeDistances(
            model,
            model_data,
            geometry_model,
            geometry_data,
            qpos,
        )
        frame_clearance = min(
            float(result.min_distance)
            for result in geometry_data.distanceResults
        )
        signed_distance = (
            -frame_penetration
            if frame_penetration > 0.0
            else frame_clearance
        )
        minimum_signed_distance = min(
            minimum_signed_distance,
            signed_distance,
        )
        if frame_clearance < CROSS_HAND_NEAR_CLEARANCE_M:
            near_clearance_frames.append(frame_index)
        if frame_penetration > CROSS_HAND_COLLISION_TOLERANCE_M:
            collision_frames.append(frame_index)
            if frame_penetration > maximum_penetration:
                maximum_penetration = frame_penetration
                worst_frame = frame_index
                worst_pair = frame_worst_pair
        if palm_penetration > CROSS_HAND_COLLISION_TOLERANCE_M:
            palm_collision_frames.append(frame_index)

    metrics: dict[str, float | int] = {
        "cross_hand_collision_pair_count": len(checker.pair_names),
        "cross_hand_collision_frame_count": len(collision_frames),
        "cross_hand_collision_fraction": (
            len(collision_frames) / len(state)
        ),
        "cross_hand_palm_collision_frame_count": len(
            palm_collision_frames
        ),
        "cross_hand_palm_collision_fraction": (
            len(palm_collision_frames) / len(state)
        ),
        "cross_hand_max_penetration_m": maximum_penetration,
        "cross_hand_min_signed_distance_m": float(
            minimum_signed_distance
        ),
        "cross_hand_near_clearance_frame_count": len(
            near_clearance_frames
        ),
        "cross_hand_collision_tolerance_m": (
            CROSS_HAND_COLLISION_TOLERANCE_M
        ),
        "cross_hand_near_clearance_threshold_m": (
            CROSS_HAND_NEAR_CLEARANCE_M
        ),
    }
    diagnostics: dict[str, object] = {
        "status": "passed" if not collision_frames else "failed",
        "checked_pair_count": len(checker.pair_names),
        "collision_frame_indices": collision_frames,
        "palm_collision_frame_indices": palm_collision_frames,
        "near_clearance_frame_indices": near_clearance_frames,
        "worst_frame_index": worst_frame,
        "worst_geometry_pair": list(worst_pair),
    }
    return metrics, diagnostics


def _minimum_hand_distance(
    model: pin.Model,
    state: np.ndarray,
) -> float:
    frame_ids = [
        model.getFrameId("left_hand_base_link"),
        model.getFrameId("right_hand_base_link"),
    ]
    if any(frame_id >= model.nframes for frame_id in frame_ids):
        raise KeyError("G1 URDF is missing a hand base frame")
    data = model.createData()
    minimum = np.inf
    for qpos in state:
        pin.forwardKinematics(model, data, qpos)
        pin.updateFramePlacements(model, data)
        distance = np.linalg.norm(
            data.oMf[frame_ids[0]].translation
            - data.oMf[frame_ids[1]].translation
        )
        minimum = min(minimum, float(distance))
    return minimum


def _mapped_camera_poses(
    episode: EgoDexEpisode,
    robot_root_positions: np.ndarray,
    source_to_robot_scale: float,
) -> np.ndarray:
    if source_to_robot_scale <= 0:
        raise ValueError("Camera/source-to-robot scale must be positive")
    basis = episode.source_to_robot_rotation
    source_position = episode.camera_to_world[:, :3, 3]
    relative = source_position - episode.points["hip"]
    mapped_position = (
        robot_root_positions
        + source_to_robot_scale * (basis @ relative.T).T
    )
    mapped_rotation = np.einsum(
        "ij,njk->nik",
        basis,
        episode.camera_to_world[:, :3, :3],
    )
    mapped_quaternion_xyzw = Rotation.from_matrix(mapped_rotation).as_quat()
    for frame_index in range(1, len(mapped_quaternion_xyzw)):
        if (
            np.dot(
                mapped_quaternion_xyzw[frame_index - 1],
                mapped_quaternion_xyzw[frame_index],
            )
            < 0.0
        ):
            mapped_quaternion_xyzw[frame_index] *= -1.0
    return np.concatenate(
        [mapped_position, mapped_quaternion_xyzw],
        axis=1,
    ).astype(np.float32)


def retarget_episode(
    episode: EgoDexEpisode,
    model: pin.Model,
    gmr_model: mj.MjModel,
    collision_checker: CrossHandCollisionChecker,
) -> tuple[RetargetedEpisode, list[str]]:
    (
        retarget,
        body_qpos,
        position_errors,
        task_errors,
        hand,
        source_to_robot_scale,
        gmr_model_diagnostics,
    ) = (
        _gmr_body_trajectory(episode, model, gmr_model)
    )
    state, joint_names = _merge_body_and_hands(
        model,
        retarget,
        body_qpos,
        hand.joint_names,
        hand.joint_positions,
    )
    if not np.isfinite(state).all():
        raise ValueError(f"GMR produced non-finite state for {episode.stem}")
    collision_metrics, collision_diagnostics = (
        _cross_hand_collision_validation(
            model,
            collision_checker,
            state,
        )
    )
    action_indices = np.minimum(
        np.arange(episode.frame_count) + ACTION_OFFSET,
        episode.frame_count - 1,
    )
    action = state[action_indices].copy()

    root_positions = body_qpos[:, :3]
    root_quaternion_xyzw = body_qpos[:, 3:7][:, [1, 2, 3, 0]]
    root_pose = np.concatenate(
        (root_positions, root_quaternion_xyzw),
        axis=1,
    ).astype(np.float32)
    camera_pose = _mapped_camera_poses(
        episode,
        root_positions,
        source_to_robot_scale=source_to_robot_scale,
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
    metrics = {
        "ik_position_rmse_m": float(
            np.sqrt(np.mean(np.square(position_errors)))
        ),
        "ik_position_max_error_m": float(np.max(position_errors)),
        "gmr_task_error_mean": float(np.mean(task_errors)),
        "gmr_task_error_max": float(np.max(task_errors)),
        "min_hand_center_distance_m": _minimum_hand_distance(model, state),
        "max_joint_step_rad": float(np.max(joint_step)),
        "mean_joint_step_rad": float(np.mean(joint_step)),
        "max_leg_deviation_from_neutral_rad": float(
            np.max(np.abs(state[:, :12]))
        ),
        **collision_metrics,
    }
    hand_diagnostics = dict(hand.diagnostics)
    hand_diagnostics["target_adapter"] = {
        "source_action_mode": hand.action_mode,
        "mapping": (
            "source action frame -> articulated O6 wrist/action transform "
            "-> GMR wrist target"
        ),
        "wrist_rotation_source": "source semantic hand action orientation",
        "wrist_position_source": (
            "source semantic hand action position with O6 geometry removed"
        ),
    }
    hand_diagnostics["cross_hand_collision_validation"] = (
        collision_diagnostics
    )
    hand_diagnostics["gmr_model"] = gmr_model_diagnostics
    return (
        RetargetedEpisode(
            source=episode,
            state=state,
            action=action,
            root_pose=root_pose,
            camera_pose=camera_pose,
            camera_intrinsics=intrinsic,
            source_to_robot_scale=float(source_to_robot_scale),
            metrics=metrics,
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


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
            _remove_path(backup_root)


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
                    "source_to_robot_rotation": (
                        source.source_to_robot_rotation.tolist()
                    ),
                    "source_to_robot_camera_translation_scale": (
                        episode.source_to_robot_scale
                    ),
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
        _write_json(temporary_root / "meta/info.json", info)
        _write_json(
            temporary_root / "meta/stats.json",
            aggregate_stats(all_episode_stats),
        )
        _write_json(
            temporary_root / "meta/egodex.json",
            {
                "schema_version": 6,
                "source_format": "EgoDex annotated HDF5 + MP4",
                "robot": "unitree_g1_with_hands",
                "retargeter": "general_motion_retargeting",
                "body_solver": "mink_complete_g1_o6",
                "body_config": "smplx_to_g1.json",
                "gmr_robot_model": {
                    "kind": GMR_MODEL_KIND,
                    "source_urdf": str(G1_URDF.resolve()),
                    "floating_root": True,
                    "fixed_link_fusion": False,
                    "robot_joint_count": 51,
                    "o6_hand_joint_count": len(HAND_JOINTS),
                    "o6_mimic_constraints": {
                        "equation_count": 10,
                        "policy": (
                            "validate retargeted values and freeze all hand "
                            "DoFs during GMR"
                        ),
                    },
                    "cross_hand_collision_avoidance": {
                        "enabled": False,
                        "geometry_pairs": 0,
                        "solver": "disabled",
                        "final_validation_penetration_tolerance_m": (
                            CROSS_HAND_COLLISION_TOLERANCE_M
                        ),
                    },
                },
                "lower_body_target": "G1 neutral standing FK",
                "hand_retargeter": (
                    "semantic action-frame to O6 wrist target plus "
                    "25-landmark articulated finger mapping"
                ),
                "hand_target_adapter": (
                    "source action -> inverse articulated O6 wrist/action "
                    "transform -> GMR wrist pose"
                ),
                "cross_hand_collision_validation": {
                    "geometry_pairs": 144,
                    "penetration_tolerance_m": (
                        CROSS_HAND_COLLISION_TOLERANCE_M
                    ),
                    "near_clearance_threshold_m": (
                        CROSS_HAND_NEAR_CLEARANCE_M
                    ),
                    "failure_policy": "audit_only",
                },
                "robot_urdf": str(G1_URDF),
                "joint_order": joint_names,
                "action_offset_frames": ACTION_OFFSET,
                "camera": (
                    "T_robot_world_from_camera_cv, xyzw; Isaac rotation is "
                    "R_cv @ diag(1,-1,-1)"
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
    gmr_model = _build_gmr_with_hand_model(G1_URDF)
    collision_checker = _build_cross_hand_collision_checker(
        model,
        G1_URDF,
    )

    pairs = discover_source_pairs(args.src_root.resolve())
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
            gmr_model=gmr_model,
            collision_checker=collision_checker,
        )
        if joint_names is None:
            joint_names = episode_joint_names
        elif joint_names != episode_joint_names:
            raise ValueError("Robot joint order changed between episodes")
        retargeted.append(converted)
        print(
            f"{episode.stem}: frames={episode.frame_count}, "
            f"IK_RMSE={converted.metrics['ik_position_rmse_m']:.4f} m, "
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
