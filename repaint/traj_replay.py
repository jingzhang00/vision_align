#!/usr/bin/env python3
"""Fast human-to-G1 trajectory comparison using Pinocchio FK."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import pinocchio as pin
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from utils.visualize_g1_trajectory import (  # noqa: E402
    BACKGROUND,
    FOREGROUND,
    LEFT,
    RIGHT,
    TARGET,
    draw_pose,
    edge_color,
    set_equal_3d_limits,
    style_3d_axis,
    style_axis,
    urdf_edges,
)
from repaint.egodex.inspect_egodex_human import (  # noqa: E402
    display_coordinates,
    load_human_trajectory,
)
from repaint.nymeria.inspect_nymeria_human import (  # noqa: E402
    SMPL_EDGES,
    SMPL_JOINT_NAMES,
)
from repaint.nymeria.nymeria_adapter import (  # noqa: E402
    NymeriaClip,
    discover_nymeria_clips,
    load_scaled_intrinsic,
)
from repaint.nymeria.nymeria_camera_alignment import (  # noqa: E402
    CAMERA_CALIBRATION_RELATIVE_PATH,
    CAMERA_TRAJECTORY_RELATIVE_PATH,
    HAND_EYE_SAMPLE_FPS,
    HEAD_BODY_INDEX,
    _world_from_device,
    estimate_session_hand_eye,
    load_clip_motion,
    load_recording_camera,
    load_slam_trajectory,
)
from repaint.rgbd_geometry import camera_points_to_world  # noqa: E402
from repaint.robot_camera import (  # noqa: E402
    CAMERA_LINK_FROM_CV_OPTICAL,
    _proper_rotation_matrix,
    head_camera_poses_world,
)
from repaint.trajectory_io import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_G1_URDF,
    SUPPORTED_FIELDS,
    Trajectory,
    find_dataset_roots,
    list_episode_indices,
    load_trajectory,
)
from repaint.vda_scene_isaac import (  # noqa: E402
    D435_INTRINSIC,
    D435_REFERENCE_SIZE,
)


SCENE_CAMERA = "#d2a8ff"
ROBOT_CAMERA = "#f778ba"
EGO_DISPLAY_FROM_WORLD = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
@dataclass(frozen=True)
class ComparisonData:
    human_label: str
    human_positions: np.ndarray
    human_names: tuple[str, ...]
    human_edges: tuple[tuple[int, int], ...]
    human_camera_poses_cv_xyzw: np.ndarray
    source_intrinsic: np.ndarray
    scene_camera_poses_cv_xyzw: np.ndarray
    robot_camera_poses_cv_xyzw: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _one_episode(
    episodes: Any,
    episode_index: int,
    *,
    label: str,
) -> dict[str, Any]:
    matching = [
        item
        for item in episodes
        if isinstance(item, dict)
        and int(item.get("episode_index", -1)) == episode_index
    ]
    if len(matching) != 1:
        raise ValueError(
            f"{label} episode {episode_index} was found {len(matching)} times"
        )
    return matching[0]


def _metadata_path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _selected_frames(
    values: np.ndarray,
    frame_indices: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0 or len(values) <= int(frame_indices[-1]):
        raise ValueError(f"{label} does not cover every selected source frame")
    selected = values[frame_indices]
    if not np.isfinite(selected).all():
        raise ValueError(f"{label} contains non-finite values")
    return selected


def _transforms_to_poses_xyzw(
    transforms: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    transforms = np.asarray(transforms, dtype=np.float64)
    if (
        transforms.ndim != 3
        or transforms.shape[1:] != (4, 4)
        or not np.isfinite(transforms).all()
        or not np.allclose(
            transforms[:, 3],
            [0.0, 0.0, 0.0, 1.0],
            atol=1e-6,
        )
    ):
        raise ValueError(f"{label} must contain finite rigid 4x4 transforms")
    return np.concatenate(
        [
            transforms[:, :3, 3],
            Rotation.from_matrix(transforms[:, :3, :3]).as_quat(),
        ],
        axis=1,
    )


def _align_nymeria_positions(
    positions_motion_world: np.ndarray,
    aria_world_from_motion_world: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions_motion_world, dtype=np.float64)
    world_motion = np.asarray(aria_world_from_motion_world, dtype=np.float64)
    if (
        positions.ndim != 3
        or positions.shape[2] != 3
        or world_motion.shape != (4, 4)
        or not np.isfinite(positions).all()
        or not np.isfinite(world_motion).all()
    ):
        raise ValueError("Nymeria body and camera transforms are invalid")
    return (
        positions @ world_motion[:3, :3].T
        + world_motion[:3, 3]
    )


def _anchored_world_transform(
    rotation: np.ndarray,
    source_anchor: np.ndarray,
    target_anchor: np.ndarray,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _proper_rotation_matrix(rotation, "world rotation")
    transform[:3, 3] = target_anchor - transform[:3, :3] @ source_anchor
    return transform


def _egodex_source_and_scene_camera(
    trajectory: Trajectory,
) -> tuple[Path, np.ndarray, str]:
    manifest = _read_json(trajectory.dataset_root / "meta/egodex.json")
    episode = _one_episode(
        manifest.get("episodes", ()),
        trajectory.episode_index,
        label="EgoDex",
    )
    source_hdf5 = _metadata_path(
        episode["source_hdf5"],
        trajectory.dataset_root,
    )

    info = _read_json(trajectory.dataset_root / "meta/info.json")
    episode_rows: list[dict[str, Any]] = []
    for path in sorted(
        (trajectory.dataset_root / "meta/episodes").glob("chunk-*/*.parquet")
    ):
        episode_rows.extend(pq.read_table(path).to_pylist())
    row = _one_episode(
        episode_rows,
        trajectory.episode_index,
        label="LeRobot metadata",
    )
    data_path = trajectory.dataset_root / str(info["data_path"]).format(
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )
    table = pq.read_table(
        data_path,
        columns=[
            "observation.camera_pose",
            "episode_index",
            "frame_index",
        ],
    )
    episode_column = np.asarray(table["episode_index"].to_pylist(), dtype=int)
    rows = np.flatnonzero(episode_column == trajectory.episode_index)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=int)[rows]
    order = np.argsort(frame_indices)
    frame_indices = frame_indices[order]
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise ValueError("EgoDex scene-camera frames are not contiguous")
    scene_camera = np.asarray(
        table["observation.camera_pose"].to_pylist(),
        dtype=np.float64,
    )[rows][order]
    return (
        source_hdf5,
        _selected_frames(
            scene_camera,
            trajectory.frame_indices,
            label="EgoDex scene camera",
        ),
        "EgoDex human",
    )


def _nymeria_source_and_scene_camera(
    trajectory: Trajectory,
) -> tuple[dict[str, Any], dict[str, Any] | None, np.ndarray, str]:
    manifest = _read_json(trajectory.dataset_root / "manifest.json")
    episode = _one_episode(
        manifest.get("episodes", ()),
        trajectory.episode_index,
        label="Nymeria",
    )
    quality_gate = manifest.get("camera_alignment_quality_gate")
    if not isinstance(quality_gate, dict) or quality_gate.get("status") != "passed":
        raise ValueError("Nymeria comparison requires a passed camera alignment")
    reports = []
    for item in quality_gate.get("camera_trajectory_reports", ()):
        if not isinstance(item, dict) or "path" not in item:
            continue
        path = Path(str(item["path"])).expanduser()
        if not path.is_absolute():
            path = trajectory.dataset_root / path
        if not path.is_file():
            continue
        report = _read_json(path)
        if report.get("episode") == episode.get("session"):
            reports.append(report)
    if len(reports) > 1:
        raise ValueError(
            f"Nymeria session {episode.get('session')} has {len(reports)} "
            "matching camera-alignment reports"
        )
    trajectory_path = _metadata_path(
        episode["trajectory_path"],
        trajectory.dataset_root,
    )
    with np.load(trajectory_path) as archive:
        if "camera_pose_cv_xyzw" not in archive:
            raise KeyError(f"{trajectory_path} has no retargeted scene camera")
        scene_camera = np.asarray(
            archive["camera_pose_cv_xyzw"],
            dtype=np.float64,
        )
    return (
        episode,
        reports[0] if reports else None,
        _selected_frames(
            scene_camera,
            trajectory.frame_indices,
            label="Nymeria scene camera",
        ),
        "Nymeria 22-joint human · G1 display basis",
    )


def _egodex_human_tracks(
    source_hdf5: Path,
    trajectory: Trajectory,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    tuple[tuple[int, int], ...],
    np.ndarray,
    np.ndarray,
]:
    indices = trajectory.frame_indices
    with h5py.File(source_hdf5, "r") as handle:
        if "transforms/camera" not in handle or "camera/intrinsic" not in handle:
            raise KeyError(f"{source_hdf5} lacks camera transforms or intrinsics")
        camera_transforms = _selected_frames(
            handle["transforms/camera"],
            indices,
            label="Human camera",
        )
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        if (
            intrinsic.shape != (3, 3)
            or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0.0
            or intrinsic[1, 1] <= 0.0
        ):
            raise ValueError("Human camera intrinsic must be a valid 3x3 matrix")
    start = int(indices[0])
    stop = int(indices[-1]) + 1
    if not np.array_equal(indices, np.arange(start, stop)):
        raise ValueError("EgoDex comparison requires contiguous source frames")
    human = load_human_trajectory(source_hdf5, start, stop)
    positions = display_coordinates(human.positions_world)
    world_basis = np.eye(4, dtype=np.float64)
    world_basis[:3, :3] = EGO_DISPLAY_FROM_WORLD
    camera_transforms = np.einsum(
        "ij,njk->nik",
        world_basis,
        camera_transforms,
    )

    return (
        positions,
        human.names,
        human.edges,
        _transforms_to_poses_xyzw(camera_transforms, label="Human camera"),
        intrinsic,
    )


def _nymeria_human_tracks(
    episode: dict[str, Any],
    report: dict[str, Any] | None,
    trajectory: Trajectory,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    tuple[tuple[int, int], ...],
    np.ndarray,
    np.ndarray,
]:
    source_video = _metadata_path(
        episode["source_video"], trajectory.dataset_root
    )
    session_root = source_video.parent.parent
    intrinsic_path = _metadata_path(
        report["intrinsics"]["path"]
        if report is not None
        else session_root / "undistorted/camera_intrinsics.json",
        trajectory.dataset_root,
    )
    clip = NymeriaClip(
        session=str(episode["session"]),
        stem=str(episode["source_stem"]),
        video_path=source_video,
        motion_208_path=_metadata_path(
            episode["source_motion_208"], trajectory.dataset_root
        ),
        text_path=_metadata_path(
            episode["source_text"], trajectory.dataset_root
        ),
        smpl_path=_metadata_path(
            episode["source_smpl_metadata"], trajectory.dataset_root
        ),
        session_metadata_path=session_root / "metadata.json",
        camera_intrinsics_path=intrinsic_path,
    )
    calibration_path = _metadata_path(
        report["recording_head"]["calibration"]
        if report is not None
        else session_root / CAMERA_CALIBRATION_RELATIVE_PATH,
        trajectory.dataset_root,
    )
    slam_path = _metadata_path(
        report["recording_head"]["trajectory"]
        if report is not None
        else session_root / CAMERA_TRAJECTORY_RELATIVE_PATH,
        trajectory.dataset_root,
    )
    slam = load_slam_trajectory(slam_path)
    if report is None:
        session_motions = [
            load_clip_motion(candidate)
            for candidate in discover_nymeria_clips(session_root.parent)
            if candidate.session == clip.session
        ]
        hand_eye, _ = estimate_session_hand_eye(
            clips=session_motions,
            slam=slam,
            sample_fps=HAND_EYE_SAMPLE_FPS,
        )
        device_from_motion_head = hand_eye.device_from_motion_head
        clip_motion = next(
            item for item in session_motions if item.clip.stem == clip.stem
        )
    else:
        device_from_motion_head = np.asarray(
            report["hand_eye"]["T_Device_MotionHead"],
            dtype=np.float64,
        )
        clip_motion = load_clip_motion(clip)
    camera = load_recording_camera(
        calibration_path,
        clip.camera_intrinsics_path,
    )
    world_device, _, _ = _world_from_device(
        slam,
        clip_motion.aria_times_us,
    )
    resampled = clip_motion.resampled
    positions_motion = (
        resampled.root_position_y_up[:, None, :]
        + np.einsum(
            "nij,nkj->nki",
            resampled.root_rotation.as_matrix(),
            resampled.relative_joint_positions,
        )
    )
    motion_head = clip_motion.body_transforms[:, HEAD_BODY_INDEX]
    world_motion = (
        world_device[0]
        @ device_from_motion_head
        @ np.linalg.inv(motion_head[0])
    )
    alignment = episode.get("camera_alignment")
    if not isinstance(alignment, dict) or alignment.get("status") != "validated":
        raise ValueError("Nymeria episode lacks validated camera alignment")
    source_to_robot = np.asarray(
        alignment["source_to_robot_world_rotation"],
        dtype=np.float64,
    )
    anchor_frame = int(trajectory.frame_indices[0])
    pelvis_aria = (
        world_motion[:3, :3] @ positions_motion[anchor_frame, 0]
        + world_motion[:3, 3]
    )
    robot_from_aria = _anchored_world_transform(
        source_to_robot,
        pelvis_aria,
        trajectory.root_pose_xyzw[0, :3],
    )
    display_world_motion = robot_from_aria @ world_motion
    positions = _align_nymeria_positions(
        positions_motion,
        display_world_motion,
    )
    head_from_camera = (
        np.linalg.inv(device_from_motion_head)
        @ camera.device_from_video_camera
    )
    camera_transforms = np.einsum(
        "ij,njk,kl->nil",
        display_world_motion,
        motion_head,
        head_from_camera,
    )
    recorded_camera_transforms = np.einsum(
        "ij,njk,kl->nil",
        robot_from_aria,
        world_device,
        camera.device_from_video_camera,
    )
    camera_transforms[:, :3, :3] = recorded_camera_transforms[:, :3, :3]
    indices = trajectory.frame_indices
    return (
        _selected_frames(positions, indices, label="Nymeria 22-joint human"),
        SMPL_JOINT_NAMES,
        SMPL_EDGES,
        _transforms_to_poses_xyzw(
            _selected_frames(camera_transforms, indices, label="Human camera"),
            label="Human camera",
        ),
        load_scaled_intrinsic(
            clip,
            video_width=clip_motion.video.width,
            video_height=clip_motion.video.height,
        ),
    )


def load_comparison_data(
    trajectory: Trajectory,
    urdf_path: Path,
) -> ComparisonData:
    if trajectory.field != "observation.state":
        raise ValueError("Camera comparison requires --field observation.state")
    nymeria = (trajectory.dataset_root / "manifest.json").is_file()
    if nymeria:
        episode, report, scene_camera, human_label = (
            _nymeria_source_and_scene_camera(trajectory)
        )
        human_tracks = _nymeria_human_tracks(
            episode,
            report,
            trajectory,
        )
    else:
        source_hdf5, scene_camera, human_label = (
            _egodex_source_and_scene_camera(trajectory)
        )
        human_tracks = _egodex_human_tracks(source_hdf5, trajectory)
    (
        human_positions,
        human_names,
        human_edges,
        human_camera,
        source_intrinsic,
    ) = human_tracks
    robot_camera = head_camera_poses_world(
        urdf_path=urdf_path,
        joint_names=trajectory.joint_names,
        joint_positions=trajectory.joint_positions,
        root_pose_xyzw=trajectory.root_pose_xyzw,
    )
    robot_camera[:, 3:7] = Rotation.from_matrix(
        Rotation.from_quat(robot_camera[:, 3:7]).as_matrix()
        @ CAMERA_LINK_FROM_CV_OPTICAL
    ).as_quat()
    expected_shape = (trajectory.frame_count, 7)
    if (
        human_camera.shape != expected_shape
        or scene_camera.shape != expected_shape
        or robot_camera.shape != expected_shape
        or human_positions.shape[0] != trajectory.frame_count
    ):
        raise ValueError("Human, robot, and camera tracks are not frame aligned")
    return ComparisonData(
        human_label=human_label,
        human_positions=human_positions,
        human_names=human_names,
        human_edges=human_edges,
        human_camera_poses_cv_xyzw=human_camera,
        source_intrinsic=source_intrinsic,
        scene_camera_poses_cv_xyzw=scene_camera,
        robot_camera_poses_cv_xyzw=robot_camera,
    )


def camera_frustum_segments(
    camera_pose_cv_xyzw: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    depth_m: float,
) -> np.ndarray:
    pose = np.asarray(camera_pose_cv_xyzw, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    width, height = (int(value) for value in image_size)
    if (
        pose.shape != (7,)
        or intrinsic.shape != (3, 3)
        or not np.isfinite(pose).all()
        or not np.isfinite(intrinsic).all()
        or np.linalg.norm(pose[3:7]) <= 1e-8
        or min(width, height) <= 1
        or not np.isfinite(depth_m)
        or depth_m <= 0.0
    ):
        raise ValueError("Camera pose, intrinsic, image size, and depth must be valid")
    pixels = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [width, 0.0, 1.0],
            [width, height, 1.0],
            [0.0, height, 1.0],
        ],
        dtype=np.float64,
    )
    try:
        rays = np.linalg.solve(intrinsic, pixels.T).T
    except np.linalg.LinAlgError as error:
        raise ValueError("Camera intrinsic must be invertible") from error
    if np.any(np.abs(rays[:, 2]) <= 1e-8):
        raise ValueError("Camera corner rays do not intersect positive camera Z")
    corners_cv = rays * (depth_m / rays[:, 2:3])
    corners_world = camera_points_to_world(corners_cv, pose)
    origin = np.repeat(pose[None, :3], 4, axis=0)
    sides = np.stack([origin, corners_world], axis=1)
    rim = np.stack(
        [corners_world, corners_world[[1, 2, 3, 0]]],
        axis=1,
    )
    return np.concatenate([sides, rim], axis=0)


def pinocchio_link_positions(
    trajectory: Trajectory,
    urdf_path: Path,
) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    model = pin.buildModelFromUrdf(str(urdf_path.resolve()))
    if tuple(model.names[1:]) != trajectory.joint_names:
        raise ValueError(
            "Trajectory joint order does not match the Pinocchio G1 URDF"
        )
    frame_ids = [
        frame_id
        for frame_id, frame in enumerate(model.frames)
        if frame.type == pin.FrameType.BODY
    ]
    names = [model.frames[frame_id].name for frame_id in frame_ids]
    positions = np.empty(
        (trajectory.frame_count, len(frame_ids), 3),
        dtype=np.float64,
    )
    data = model.createData()
    for index, qpos in enumerate(trajectory.joint_positions):
        pin.forwardKinematics(model, data, qpos)
        pin.updateFramePlacements(model, data)
        local = np.stack(
            [data.oMf[frame_id].translation for frame_id in frame_ids]
        )
        rotation = Rotation.from_quat(
            trajectory.root_pose_xyzw[index, 3:7]
        ).as_matrix()
        positions[index] = (
            local @ rotation.T
            + trajectory.root_pose_xyzw[index, :3]
        )
    return positions, names, urdf_edges(urdf_path, names)


def plot_overview(
    trajectory: Trajectory,
    positions: np.ndarray,
    names: list[str],
    edges: list[tuple[int, int]],
    output_path: Path,
) -> None:
    frames = trajectory.frame_count
    x = trajectory.frame_indices
    fig = plt.figure(
        figsize=(16, 9),
        facecolor=BACKGROUND,
        constrained_layout=True,
    )
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1.0))
    pose_axis = fig.add_subplot(grid[:, 0], projection="3d")
    joint_axis = fig.add_subplot(grid[0, 1])
    step_axis = fig.add_subplot(grid[1, 1])
    style_3d_axis(pose_axis)
    style_axis(joint_axis)
    style_axis(step_axis)

    samples = np.unique(np.linspace(0, frames - 1, 5, dtype=int))
    for frame in samples[:-1]:
        draw_pose(
            pose_axis,
            positions[frame],
            edges,
            names,
            alpha=0.16,
            linewidth=1.2,
        )
    draw_pose(
        pose_axis,
        positions[samples[-1]],
        edges,
        names,
        linewidth=2.4,
    )
    set_equal_3d_limits(pose_axis, positions.reshape(-1, 3))
    pose_axis.set(
        title="Pinocchio FK trajectory (five sampled poses)",
        xlabel="World X (m)",
        ylabel="World Y (m)",
        zlabel="World Z (m)",
    )

    arm_indices = [
        index
        for index, name in enumerate(trajectory.joint_names)
        if "shoulder" in name or "elbow" in name or "wrist" in name
    ]
    for index in arm_indices:
        name = trajectory.joint_names[index]
        joint_axis.plot(
            x,
            trajectory.joint_positions[:, index],
            color=LEFT if name.startswith("left_") else RIGHT,
            alpha=0.62,
            linewidth=1.0,
        )
    joint_axis.set(
        title="Arm joint trajectory",
        xlabel="Frame",
        ylabel="Position (rad)",
    )

    step = np.max(
        np.abs(np.diff(trajectory.joint_positions, axis=0)),
        axis=1,
    )
    step_axis.plot(x[1:], step, color=TARGET, linewidth=1.5)
    step_axis.set(
        title="Maximum joint step per frame",
        xlabel="Frame",
        ylabel="|Δq| max (rad)",
    )
    fig.suptitle(
        f"G1 trajectory · episode {trajectory.episode_index} · "
        f"{frames} frames · {trajectory.fps:g} FPS\n",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def plot_comparison_animation(
    trajectory: Trajectory,
    robot_positions: np.ndarray,
    robot_names: list[str],
    robot_edges: list[tuple[int, int]],
    comparison: ComparisonData,
    output_path: Path,
    frustum_depth_m: float,
) -> None:
    if not np.isfinite(frustum_depth_m) or frustum_depth_m <= 0.0:
        raise ValueError("--frustum-depth must be positive and finite")
    fig = plt.figure(
        figsize=(16, 8),
        dpi=100,
        facecolor=BACKGROUND,
        constrained_layout=True,
    )
    human_axis = fig.add_subplot(121, projection="3d")
    robot_axis = fig.add_subplot(122, projection="3d")
    style_3d_axis(human_axis)
    style_3d_axis(robot_axis)
    set_equal_3d_limits(
        human_axis,
        np.concatenate(
            [
                comparison.human_positions.reshape(-1, 3),
                comparison.human_camera_poses_cv_xyzw[:, :3],
            ],
            axis=0,
        ),
    )
    set_equal_3d_limits(
        robot_axis,
        np.concatenate(
            [
                robot_positions.reshape(-1, 3),
                comparison.scene_camera_poses_cv_xyzw[:, :3],
                comparison.robot_camera_poses_cv_xyzw[:, :3],
            ],
            axis=0,
        ),
    )
    human_axis.set(
        title=f"{comparison.human_label} + source camera",
        xlabel="World X (m)",
        ylabel="World Y (m)",
        zlabel="World Z (m)",
    )
    robot_axis.set(
        title="Retargeted G1 + scene camera + G1 camera",
        xlabel="World X (m)",
        ylabel="World Y (m)",
        zlabel="World Z (m)",
    )

    def skeleton_lines(
        axis: plt.Axes,
        names: tuple[str, ...] | list[str],
        edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    ) -> list[tuple[Any, int, int]]:
        lines: list[tuple[Any, int, int]] = []
        for parent, child in edges:
            line, = axis.plot(
                [],
                [],
                [],
                color=edge_color(names[parent], names[child]),
                linewidth=2.4,
                solid_capstyle="round",
            )
            lines.append((line, parent, child))
        return lines

    human_lines = skeleton_lines(
        human_axis,
        comparison.human_names,
        comparison.human_edges,
    )
    robot_lines = skeleton_lines(robot_axis, robot_names, robot_edges)
    human_axis.plot(
        *comparison.human_camera_poses_cv_xyzw[:, :3].T,
        color=TARGET,
        linewidth=0.8,
        linestyle=":",
        alpha=0.35,
    )
    robot_axis.plot(
        *comparison.scene_camera_poses_cv_xyzw[:, :3].T,
        color=SCENE_CAMERA,
        linewidth=0.8,
        linestyle=":",
        alpha=0.35,
    )
    robot_axis.plot(
        *comparison.robot_camera_poses_cv_xyzw[:, :3].T,
        color=ROBOT_CAMERA,
        linewidth=0.8,
        linestyle=":",
        alpha=0.35,
    )
    human_frustum = Line3DCollection(
        camera_frustum_segments(
            comparison.human_camera_poses_cv_xyzw[0],
            comparison.source_intrinsic,
            trajectory.source_size,
            frustum_depth_m,
        ),
        colors=TARGET,
        linewidths=1.8,
    )
    scene_frustum = Line3DCollection(
        camera_frustum_segments(
            comparison.scene_camera_poses_cv_xyzw[0],
            comparison.source_intrinsic,
            trajectory.source_size,
            frustum_depth_m,
        ),
        colors=SCENE_CAMERA,
        linewidths=1.8,
    )
    robot_frustum = Line3DCollection(
        camera_frustum_segments(
            comparison.robot_camera_poses_cv_xyzw[0],
            D435_INTRINSIC,
            D435_REFERENCE_SIZE,
            frustum_depth_m,
        ),
        colors=ROBOT_CAMERA,
        linewidths=1.8,
    )
    human_axis.add_collection3d(human_frustum)
    robot_axis.add_collection3d(scene_frustum)
    robot_axis.add_collection3d(robot_frustum)
    human_axis.legend(
        handles=[Line2D([0], [0], color=TARGET, label="Human camera")],
        loc="upper left",
        frameon=False,
        labelcolor=FOREGROUND,
    )
    robot_axis.legend(
        handles=[
            Line2D([0], [0], color=SCENE_CAMERA, label="Retargeted scene camera"),
            Line2D([0], [0], color=ROBOT_CAMERA, label="G1 D435 camera"),
        ],
        loc="upper left",
        frameon=False,
        labelcolor=FOREGROUND,
    )
    fig.suptitle(
        f"Human → G1 retarget comparison · episode {trajectory.episode_index}",
        color=FOREGROUND,
        fontsize=15,
        fontweight="bold",
    )
    frame_text = fig.text(
        0.5,
        0.015,
        "",
        horizontalalignment="center",
        color=FOREGROUND,
        fontsize=11,
    )

    def update_skeleton(
        lines: list[tuple[Any, int, int]],
        current: np.ndarray,
    ) -> None:
        for line, parent, child in lines:
            segment = current[[parent, child]]
            line.set_data(segment[:, 0], segment[:, 1])
            line.set_3d_properties(segment[:, 2])

    def update(frame: int) -> list[Any]:
        update_skeleton(human_lines, comparison.human_positions[frame])
        update_skeleton(robot_lines, robot_positions[frame])
        human_frustum.set_segments(
            camera_frustum_segments(
                comparison.human_camera_poses_cv_xyzw[frame],
                comparison.source_intrinsic,
                trajectory.source_size,
                frustum_depth_m,
            )
        )
        scene_frustum.set_segments(
            camera_frustum_segments(
                comparison.scene_camera_poses_cv_xyzw[frame],
                comparison.source_intrinsic,
                trajectory.source_size,
                frustum_depth_m,
            )
        )
        robot_frustum.set_segments(
            camera_frustum_segments(
                comparison.robot_camera_poses_cv_xyzw[frame],
                D435_INTRINSIC,
                D435_REFERENCE_SIZE,
                frustum_depth_m,
            )
        )
        frame_text.set_text(
            f"Episode {trajectory.episode_index} · "
            f"source frame {int(trajectory.frame_indices[frame])} · "
            f"{frame + 1}/{trajectory.frame_count}"
        )
        return [
            *(line for line, _, _ in human_lines),
            *(line for line, _, _ in robot_lines),
            human_frustum,
            scene_frustum,
            robot_frustum,
            frame_text,
        ]

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=trajectory.frame_count,
        interval=1000.0 / trajectory.fps,
        blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=trajectory.fps,
        codec="libx264",
        bitrate=6000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "Human and G1 camera trajectory comparison"},
    )
    movie.save(output_path, writer=writer)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fast G1 FK overview and a side-by-side human/robot "
            "camera-trajectory comparison MP4."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Native Nymeria episode/dataset, LeRobot root, or folder "
            "recursively containing either format (default: canonical "
            "egodex/basic_pick_place/lerobot)."
        ),
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to render; omit to render every episode.",
    )
    parser.add_argument(
        "--field",
        choices=SUPPORTED_FIELDS,
        default="observation.state",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_G1_URDF)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--frustum-depth",
        type=float,
        default=0.25,
        help="Displayed camera-frustum depth in meters (default: 0.25).",
    )
    parser.add_argument("--overview-only", action="store_true")
    return parser.parse_args()


def render_episode(args: argparse.Namespace, episode_index: int) -> None:
    trajectory = load_trajectory(
        args.dataset,
        episode_index,
        args.field,
        args.start_frame,
        args.end_frame,
    )
    positions, names, edges = pinocchio_link_positions(
        trajectory,
        args.urdf,
    )
    episode_dir = f"episode_{trajectory.episode_index:06d}"
    if args.output_dir is None:
        output_dir = trajectory.dataset_root / "trajectory_check" / episode_dir
    elif args.episode is None:
        output_dir = args.output_dir / episode_dir
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = output_dir / "g1_trajectory_overview.png"
    plot_overview(trajectory, positions, names, edges, overview)
    print(f"Wrote {overview}")
    if not args.overview_only:
        comparison = load_comparison_data(trajectory, args.urdf)
        video = output_dir / "human_g1_camera_comparison.mp4"
        plot_comparison_animation(
            trajectory,
            positions,
            names,
            edges,
            comparison,
            video,
            args.frustum_depth,
        )
        print(f"Wrote {video}")


def main() -> None:
    args = parse_args()
    input_root = args.dataset.expanduser().resolve()
    output_root = args.output_dir
    dataset_roots = find_dataset_roots(input_root)
    if (
        args.episode is None
        and (input_root / "trajectory.npz").is_file()
        and input_root.name.startswith("episode_")
    ):
        args.episode = int(input_root.name.removeprefix("episode_"))
    for dataset_root in dataset_roots:
        args.dataset = dataset_root
        if output_root is not None and len(dataset_roots) > 1:
            args.output_dir = (
                output_root / dataset_root.relative_to(input_root)
            )
        episode_indices = (
            (args.episode,)
            if args.episode is not None
            else list_episode_indices(dataset_root)
        )
        print(
            f"{dataset_root}: processing {len(episode_indices)} episode(s): "
            + ", ".join(str(index) for index in episode_indices)
        )
        for episode_index in episode_indices:
            render_episode(args, episode_index)


if __name__ == "__main__":
    main()
