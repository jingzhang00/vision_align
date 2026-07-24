from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .trajectory import HandTrajectory
from .video_io import read_video, write_video


CV_TO_G1_WORLD = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

LEG_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

WRIST_JOINTS = (
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class RetargetResult:
    qpos: np.ndarray
    left_targets: np.ndarray
    right_targets: np.ndarray
    left_target_rotations: np.ndarray
    right_target_rotations: np.ndarray
    left_errors: np.ndarray
    right_errors: np.ndarray
    left_orientation_errors: np.ndarray
    right_orientation_errors: np.ndarray
    left_visible: np.ndarray
    right_visible: np.ndarray
    base_position: np.ndarray
    scale: float
    search_results: list[dict[str, float | list[float]]]


def build_g1_model(mjcf_path: str | Path, fovy_degrees: float) -> mujoco.MjModel:
    path = Path(mjcf_path).resolve()
    tree = ET.parse(path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"No <compiler> element in {path}")
    compiler.set("meshdir", str((path.parent / "../meshes/g1").resolve()))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> element in {path}")
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "synthesis_camera",
            "pos": "0.1 0 1.3",
            "quat": "0.5 0.5 0.5 0.5",
            "fovy": f"{fovy_degrees:.8f}",
        },
    )
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1920")
    global_visual.set("offheight", "1080")
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml)


def _visible(
    pixels: np.ndarray,
    confidence: np.ndarray,
    width: int,
    height: int,
    margin: float = 0.0,
) -> np.ndarray:
    return (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= -margin)
        & (pixels[:, 0] < width + margin)
        & (pixels[:, 1] >= -margin)
        & (pixels[:, 1] < height + margin)
        & (confidence >= 0.5)
    )


def _tracked(points: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Return frames whose HDF5 hand point is valid, including off-screen points."""

    return np.isfinite(points).all(axis=1) & (confidence >= 0.5)


def map_targets(
    trajectory: HandTrajectory,
    scale: float,
    camera_origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = camera_origin + (
        CV_TO_G1_WORLD @ (scale * trajectory.left_reference_camera).T
    ).T
    right = camera_origin + (
        CV_TO_G1_WORLD @ (scale * trajectory.right_reference_camera).T
    ).T
    return left, right


def map_target_rotations(
    trajectory: HandTrajectory,
    left_calibration: np.ndarray,
    right_calibration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer relative HDF5 hand rotation onto each calibrated G1 palm."""

    targets: list[np.ndarray] = []
    for source, calibration in (
        (trajectory.left_reference_rotations, left_calibration),
        (trajectory.right_reference_rotations, right_calibration),
    ):
        mapped = np.einsum("ij,njk->nik", CV_TO_G1_WORLD, source)
        hand_to_palm = mapped[0].T @ calibration
        targets.append(np.einsum("nij,jk->nik", mapped, hand_to_palm))
    return targets[0], targets[1]


def camera_poses(
    trajectory: HandTrajectory,
    scale: float,
    camera_origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return world positions, OpenCV rotations, and MuJoCo camera quaternions."""

    positions: list[np.ndarray] = []
    cv_rotations: list[np.ndarray] = []
    gl_quaternions: list[np.ndarray] = []
    cv_to_gl = np.diag([1.0, -1.0, -1.0])
    for motion in trajectory.camera_motion:
        position = camera_origin + CV_TO_G1_WORLD @ (scale * motion[:3, 3])
        world_from_cv = CV_TO_G1_WORLD @ motion[:3, :3]
        world_from_gl = world_from_cv @ cv_to_gl
        xyzw = Rotation.from_matrix(world_from_gl).as_quat()
        wxyz = np.roll(xyzw, 1)
        positions.append(position)
        cv_rotations.append(world_from_cv)
        gl_quaternions.append(wxyz)
    return (
        np.asarray(positions),
        np.asarray(cv_rotations),
        np.asarray(gl_quaternions),
    )


def _dof_index(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = model.joint(joint_name).id
    return int(model.jnt_dofadr[joint_id])


def _make_position_tasks(
    model: mujoco.MjModel, home_q: np.ndarray
) -> tuple[mink.FrameTask, mink.FrameTask, mink.PostureTask, mink.DofFreezingTask]:
    left = mink.FrameTask(
        frame_name="left_palm",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.0,
        gain=0.8,
        lm_damping=1e-4,
    )
    right = mink.FrameTask(
        frame_name="right_palm",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.0,
        gain=0.8,
        lm_damping=1e-4,
    )
    posture_cost = np.full(model.nv, 0.025, dtype=np.float64)
    posture_cost[:6] = 0.0
    posture_cost[18:21] = 0.08
    posture = mink.PostureTask(
        model=model, cost=posture_cost, gain=0.25, lm_damping=1e-4
    )
    posture.set_target(home_q)
    frozen_dofs = (
        list(range(6))
        + [_dof_index(model, name) for name in LEG_JOINTS]
        + [_dof_index(model, name) for name in WRIST_JOINTS]
    )
    frozen = mink.DofFreezingTask(model=model, dof_indices=frozen_dofs)
    return left, right, posture, frozen


def _make_pose_tasks(
    model: mujoco.MjModel, seed_q: np.ndarray
) -> tuple[
    mink.FrameTask,
    mink.FrameTask,
    mink.PostureTask,
    mink.DampingTask,
    mink.DofFreezingTask,
    mink.ConfigurationLimit,
]:
    left = mink.FrameTask(
        frame_name="left_palm",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.1,
        gain=0.5,
        lm_damping=5e-4,
    )
    right = mink.FrameTask(
        frame_name="right_palm",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.1,
        gain=0.5,
        lm_damping=5e-4,
    )
    posture_cost = np.full(model.nv, 0.02, dtype=np.float64)
    posture_cost[:6] = 0.0
    posture_cost[18:21] = 0.05
    for joint_name in WRIST_JOINTS:
        posture_cost[_dof_index(model, joint_name)] = 0.01
    posture = mink.PostureTask(
        model=model, cost=posture_cost, gain=0.15, lm_damping=1e-4
    )
    posture.set_target(seed_q)

    damping_cost = np.full(model.nv, 0.05, dtype=np.float64)
    damping_cost[:6] = 0.0
    damping = mink.DampingTask(model=model, cost=damping_cost)
    frozen_dofs = list(range(6)) + [
        _dof_index(model, name) for name in LEG_JOINTS
    ]
    frozen = mink.DofFreezingTask(model=model, dof_indices=frozen_dofs)
    configuration_limit = mink.ConfigurationLimit(
        model=model,
        gain=0.95,
        min_distance_from_limits=0.03,
    )
    return left, right, posture, damping, frozen, configuration_limit


def _solve_frame(
    configuration: mink.Configuration,
    left_task: mink.FrameTask,
    right_task: mink.FrameTask,
    posture_task: mink.PostureTask,
    freeze_task: mink.DofFreezingTask,
    left_target: np.ndarray,
    right_target: np.ndarray,
    left_active: bool,
    right_active: bool,
    iterations: int,
) -> tuple[float, float]:
    tasks: list[mink.Task] = [posture_task]
    if left_active:
        left_task.set_target(mink.SE3.from_translation(left_target))
        tasks.append(left_task)
    if right_active:
        right_task.set_target(mink.SE3.from_translation(right_target))
        tasks.append(right_task)

    for _ in range(iterations):
        velocity = mink.solve_ik(
            configuration,
            tasks,
            dt=0.05,
            solver="daqp",
            damping=1e-5,
            constraints=[freeze_task],
        )
        configuration.integrate_inplace(velocity, 0.05)

    left_position = configuration.get_transform_frame_to_world(
        "left_palm", "site"
    ).translation()
    right_position = configuration.get_transform_frame_to_world(
        "right_palm", "site"
    ).translation()
    return (
        float(np.linalg.norm(left_position - left_target))
        if left_active
        else float("nan"),
        float(np.linalg.norm(right_position - right_target))
        if right_active
        else float("nan"),
    )


def _solve_pose_frame(
    configuration: mink.Configuration,
    left_task: mink.FrameTask,
    right_task: mink.FrameTask,
    posture_task: mink.PostureTask,
    damping_task: mink.DampingTask,
    freeze_task: mink.DofFreezingTask,
    configuration_limit: mink.ConfigurationLimit,
    left_target: np.ndarray,
    right_target: np.ndarray,
    left_target_rotation: np.ndarray,
    right_target_rotation: np.ndarray,
    left_active: bool,
    right_active: bool,
    iterations: int,
) -> tuple[float, float, float, float]:
    tasks: list[mink.Task] = [posture_task, damping_task]
    if left_active:
        left_task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(left_target_rotation),
                left_target,
            )
        )
        tasks.append(left_task)
    if right_active:
        right_task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(right_target_rotation),
                right_target,
            )
        )
        tasks.append(right_task)

    for _ in range(iterations):
        velocity = mink.solve_ik(
            configuration,
            tasks,
            dt=0.05,
            solver="daqp",
            damping=1e-5,
            limits=[configuration_limit],
            constraints=[freeze_task],
        )
        configuration.integrate_inplace(velocity, 0.05)

    left_pose = configuration.get_transform_frame_to_world("left_palm", "site")
    right_pose = configuration.get_transform_frame_to_world("right_palm", "site")
    left_orientation_error = Rotation.from_matrix(
        left_pose.rotation().as_matrix().T @ left_target_rotation
    ).magnitude()
    right_orientation_error = Rotation.from_matrix(
        right_pose.rotation().as_matrix().T @ right_target_rotation
    ).magnitude()
    return (
        float(np.linalg.norm(left_pose.translation() - left_target))
        if left_active
        else float("nan"),
        float(np.linalg.norm(right_pose.translation() - right_target))
        if right_active
        else float("nan"),
        float(left_orientation_error) if left_active else float("nan"),
        float(right_orientation_error) if right_active else float("nan"),
    )


def _base_candidates() -> Iterable[tuple[np.ndarray, float]]:
    # 0.84 is the G1 shoulder-to-palm / source Arm-to-Hand length ratio.
    scale = 0.84
    for x in (-0.12, 0.0, 0.12):
        for y in (-0.10, 0.0, 0.10):
            for z in (0.70, 0.793, 0.88):
                yield np.array([x, y, z], dtype=np.float64), scale


def _evaluate_candidate(
    model: mujoco.MjModel,
    trajectory: HandTrajectory,
    base_position: np.ndarray,
    scale: float,
    camera_origin: np.ndarray,
    left_tracked: np.ndarray,
    right_tracked: np.ndarray,
) -> dict[str, float | list[float]]:
    left_targets, right_targets = map_targets(trajectory, scale, camera_origin)
    home_q = model.qpos0.copy()
    home_q[:3] = base_position
    configuration = mink.Configuration(model, q=home_q)
    left_task, right_task, posture_task, freeze_task = _make_position_tasks(
        model, home_q
    )

    sample_ids = np.unique(
        np.concatenate(
            [
                np.arange(0, trajectory.frame_count, 12),
                np.flatnonzero(np.diff(right_tracked.astype(np.int8), prepend=0) != 0),
            ]
        )
    )
    sample_ids = sample_ids[(sample_ids >= 0) & (sample_ids < trajectory.frame_count)]
    errors: list[float] = []
    feasible = 0
    targets = 0
    for frame_id in sample_ids:
        left_error, right_error = _solve_frame(
            configuration,
            left_task,
            right_task,
            posture_task,
            freeze_task,
            left_targets[frame_id],
            right_targets[frame_id],
            bool(left_tracked[frame_id]),
            bool(right_tracked[frame_id]),
            iterations=45,
        )
        for active, error in (
            (left_tracked[frame_id], left_error),
            (right_tracked[frame_id], right_error),
        ):
            if active:
                targets += 1
                errors.append(error)
                feasible += int(error < 0.035)

    mean_error = float(np.mean(errors)) if errors else float("inf")
    feasible_fraction = feasible / max(targets, 1)
    base_prior = float(
        np.linalg.norm((base_position - np.array([0.0, 0.0, 0.793])) / 0.2)
    )
    # If several placements are fully feasible, prefer an anatomically plausible
    # pelvis-to-head-camera relation instead of moving the pelvis into the camera.
    score = 10.0 * feasible_fraction - mean_error - 0.005 * base_prior
    return {
        "base_position": base_position.tolist(),
        "scale": float(scale),
        "feasible_fraction": float(feasible_fraction),
        "mean_error_m": mean_error,
        "score": float(score),
    }


def retarget_to_g1(
    model: mujoco.MjModel,
    trajectory: HandTrajectory,
    *,
    source_width: int,
    source_height: int,
    camera_origin: np.ndarray | None = None,
) -> RetargetResult:
    if camera_origin is None:
        camera_origin = np.array([0.10, 0.0, 1.30], dtype=np.float64)
    left_visible = _visible(
        trajectory.left_pixels,
        trajectory.left_confidence,
        source_width,
        source_height,
    )
    right_visible = _visible(
        trajectory.right_pixels,
        trajectory.right_confidence,
        source_width,
        source_height,
    )
    # Visibility is only a rendering/diagnostic property.  The HDF5 poses remain
    # valid while a hand is outside the image, so keep those targets active to
    # avoid a discontinuity when the hand enters or leaves the frame.
    left_tracked = _tracked(
        trajectory.left_reference_camera, trajectory.left_confidence
    )
    right_tracked = _tracked(
        trajectory.right_reference_camera, trajectory.right_confidence
    )

    search_results = [
        _evaluate_candidate(
            model,
            trajectory,
            base_position,
            scale,
            camera_origin,
            left_tracked,
            right_tracked,
        )
        for base_position, scale in _base_candidates()
    ]
    search_results.sort(key=lambda item: float(item["score"]), reverse=True)
    best = search_results[0]
    base_position = np.asarray(best["base_position"], dtype=np.float64)
    scale = float(best["scale"])
    left_targets, right_targets = map_targets(trajectory, scale, camera_origin)

    home_q = model.qpos0.copy()
    home_q[:3] = base_position
    configuration = mink.Configuration(model, q=home_q)
    left_seed_task, right_seed_task, seed_posture_task, seed_freeze_task = (
        _make_position_tasks(model, home_q)
    )
    _solve_frame(
        configuration,
        left_seed_task,
        right_seed_task,
        seed_posture_task,
        seed_freeze_task,
        left_targets[0],
        right_targets[0],
        bool(left_tracked[0]),
        bool(right_tracked[0]),
        iterations=100,
    )
    seed_q = configuration.q.copy()
    left_calibration = configuration.get_transform_frame_to_world(
        "left_palm", "site"
    ).rotation().as_matrix()
    right_calibration = configuration.get_transform_frame_to_world(
        "right_palm", "site"
    ).rotation().as_matrix()
    left_target_rotations, right_target_rotations = map_target_rotations(
        trajectory,
        left_calibration,
        right_calibration,
    )
    (
        left_task,
        right_task,
        posture_task,
        damping_task,
        freeze_task,
        configuration_limit,
    ) = _make_pose_tasks(model, seed_q)
    qpos = np.zeros((trajectory.frame_count, model.nq), dtype=np.float64)
    left_errors = np.full(trajectory.frame_count, np.nan)
    right_errors = np.full(trajectory.frame_count, np.nan)
    left_orientation_errors = np.full(trajectory.frame_count, np.nan)
    right_orientation_errors = np.full(trajectory.frame_count, np.nan)
    for frame_id in range(trajectory.frame_count):
        active = (bool(left_tracked[frame_id]), bool(right_tracked[frame_id]))
        iterations = 40 if frame_id == 0 else 8
        (
            left_errors[frame_id],
            right_errors[frame_id],
            left_orientation_errors[frame_id],
            right_orientation_errors[frame_id],
        ) = _solve_pose_frame(
            configuration,
            left_task,
            right_task,
            posture_task,
            damping_task,
            freeze_task,
            configuration_limit,
            left_targets[frame_id],
            right_targets[frame_id],
            left_target_rotations[frame_id],
            right_target_rotations[frame_id],
            active[0],
            active[1],
            iterations=iterations,
        )
        qpos[frame_id] = configuration.q

    return RetargetResult(
        qpos=qpos,
        left_targets=left_targets,
        right_targets=right_targets,
        left_target_rotations=left_target_rotations,
        right_target_rotations=right_target_rotations,
        left_errors=left_errors,
        right_errors=right_errors,
        left_orientation_errors=left_orientation_errors,
        right_orientation_errors=right_orientation_errors,
        left_visible=left_visible,
        right_visible=right_visible,
        base_position=base_position,
        scale=scale,
        search_results=search_results,
    )


def _set_camera(
    model: mujoco.MjModel,
    camera_id: int,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> None:
    model.cam_pos[camera_id] = position
    model.cam_quat[camera_id] = quaternion


def _project_world_point(
    point: np.ndarray,
    camera_position: np.ndarray,
    world_from_cv: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    point_cv = world_from_cv.T @ (point - camera_position)
    if point_cv[2] <= 1e-6:
        return np.array([np.nan, np.nan])
    return np.array(
        [
            intrinsic[0, 0] * point_cv[0] / point_cv[2] + intrinsic[0, 2],
            intrinsic[1, 1] * point_cv[1] / point_cv[2] + intrinsic[1, 2],
        ]
    )


def render_g1(
    model: mujoco.MjModel,
    trajectory: HandTrajectory,
    result: RetargetResult,
    *,
    output_root: str | Path,
    width: int,
    height: int,
    source_video: str | Path | None = None,
) -> dict[str, float | int | str]:
    output_root = Path(output_root)
    rgb_dir = output_root / "robot_rgb"
    mask_dir = output_root / "robot_masks"
    depth_dir = output_root / "robot_depth_mm"
    for directory in (rgb_dir, mask_dir, depth_dir):
        directory.mkdir(parents=True, exist_ok=True)

    camera_origin = np.array([0.10, 0.0, 1.30], dtype=np.float64)
    positions, cv_rotations, gl_quaternions = camera_poses(
        trajectory, result.scale, camera_origin
    )
    camera_id = model.camera("synthesis_camera").id
    original_groups = model.geom_group.copy()
    model.geom_group[original_groups == 2] = 0
    for body_name in ("left_shoulder_pitch_link", "right_shoulder_pitch_link"):
        body_id = model.body(body_name).id
        geom_ids = np.asarray(mink.get_subtree_geom_ids(model, body_id), dtype=int)
        visual_ids = geom_ids[original_groups[geom_ids] == 2]
        model.geom_group[visual_ids] = 2
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[2] = 1

    robot_frames: list[np.ndarray] = []
    diagnostic_points: list[tuple[np.ndarray, np.ndarray]] = []
    scaled_intrinsic = trajectory.intrinsic.copy()
    scaled_intrinsic[0] *= width / (2.0 * trajectory.intrinsic[0, 2])
    scaled_intrinsic[1] *= height / (2.0 * trajectory.intrinsic[1, 2])

    for frame_id, q in enumerate(result.qpos):
        data.qpos[:] = q
        _set_camera(
            model,
            camera_id,
            positions[frame_id],
            gl_quaternions[frame_id],
        )
        mujoco.mj_forward(model, data)

        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(data, camera=camera_id, scene_option=option)
        rgb = renderer.render().copy()

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera_id, scene_option=option)
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        mask = np.any(segmentation >= 0, axis=-1).astype(np.uint8) * 255

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera_id, scene_option=option)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        depth_mm = np.where(mask > 0, np.clip(depth * 1000.0, 0, 65535), 0).astype(
            np.uint16
        )

        cv2.imwrite(
            str(rgb_dir / f"{frame_id:05d}.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(mask_dir / f"{frame_id:05d}.png"), mask)
        cv2.imwrite(str(depth_dir / f"{frame_id:05d}.png"), depth_mm)
        robot_frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        left_world = data.site("left_palm").xpos.copy()
        right_world = data.site("right_palm").xpos.copy()
        diagnostic_points.append(
            (
                _project_world_point(
                    left_world,
                    positions[frame_id],
                    cv_rotations[frame_id],
                    scaled_intrinsic,
                ),
                _project_world_point(
                    right_world,
                    positions[frame_id],
                    cv_rotations[frame_id],
                    scaled_intrinsic,
                ),
            )
        )
    renderer.close()
    write_video(output_root / "robot_only.mp4", robot_frames, 30.0)

    if source_video is not None:
        source_frames, fps = read_video(source_video)
        if len(source_frames) != len(robot_frames):
            raise ValueError("Source video and retarget result have different lengths")
        diagnostics: list[np.ndarray] = []
        source_scale = np.array(
            [
                width / (2.0 * trajectory.intrinsic[0, 2]),
                height / (2.0 * trajectory.intrinsic[1, 2]),
            ]
        )
        for frame_id, frame in enumerate(source_frames):
            canvas = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            desired = (
                trajectory.left_pixels[frame_id] * source_scale,
                trajectory.right_pixels[frame_id] * source_scale,
            )
            actual = diagnostic_points[frame_id]
            for target, achieved, color in zip(
                desired, actual, ((255, 100, 20), (20, 20, 255)), strict=True
            ):
                if np.isfinite(target).all():
                    cv2.drawMarker(
                        canvas,
                        tuple(np.rint(target).astype(int)),
                        color,
                        cv2.MARKER_CROSS,
                        20,
                        3,
                    )
                if np.isfinite(achieved).all():
                    cv2.circle(
                        canvas,
                        tuple(np.rint(achieved).astype(int)),
                        7,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )
            diagnostics.append(canvas)
        write_video(output_root / "ik_projection_overlay.mp4", diagnostics, fps)

    joint_steps = np.abs(np.diff(result.qpos, axis=0))
    wrist_motion: dict[str, dict[str, float]] = {}
    for joint_name in WRIST_JOINTS:
        joint_id = model.joint(joint_name).id
        qpos_index = int(model.jnt_qposadr[joint_id])
        values = result.qpos[:, qpos_index]
        steps = np.abs(np.diff(values))
        wrist_motion[joint_name] = {
            "range_rad": float(np.ptp(values)),
            "max_step_rad": float(np.max(steps)) if steps.size else 0.0,
        }
    report = {
        "frame_count": int(result.qpos.shape[0]),
        "base_position": result.base_position.tolist(),
        "scale": result.scale,
        "hand_representation": (
            "single HDF5 hand-origin position plus rigid hand-frame rotation; "
            "no MANO or grasp state"
        ),
        "tracking_policy": "confidence-valid targets stay active off-screen",
        "left_tracked_frames": int((trajectory.left_confidence >= 0.5).sum()),
        "right_tracked_frames": int((trajectory.right_confidence >= 0.5).sum()),
        "left_visible_frames": int(result.left_visible.sum()),
        "right_visible_frames": int(result.right_visible.sum()),
        "left_mean_error_m": float(np.nanmean(result.left_errors)),
        "left_max_error_m": float(np.nanmax(result.left_errors)),
        "right_mean_error_m": float(np.nanmean(result.right_errors)),
        "right_max_error_m": float(np.nanmax(result.right_errors)),
        "left_mean_orientation_error_deg": float(
            np.degrees(np.nanmean(result.left_orientation_errors))
        ),
        "left_max_orientation_error_deg": float(
            np.degrees(np.nanmax(result.left_orientation_errors))
        ),
        "right_mean_orientation_error_deg": float(
            np.degrees(np.nanmean(result.right_orientation_errors))
        ),
        "right_max_orientation_error_deg": float(
            np.degrees(np.nanmax(result.right_orientation_errors))
        ),
        "max_joint_step_rad": float(np.max(joint_steps)) if joint_steps.size else 0.0,
        "wrist_motion": wrist_motion,
        "top_base_search_results": result.search_results[:10],
    }
    (output_root / "ik_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_root / "g1_trajectory.npz",
        qpos=result.qpos,
        left_targets=result.left_targets,
        right_targets=result.right_targets,
        left_target_rotations=result.left_target_rotations,
        right_target_rotations=result.right_target_rotations,
        left_errors=result.left_errors,
        right_errors=result.right_errors,
        left_orientation_errors=result.left_orientation_errors,
        right_orientation_errors=result.right_orientation_errors,
        left_visible=result.left_visible,
        right_visible=result.right_visible,
        base_position=result.base_position,
        scale=result.scale,
        camera_positions=positions,
        camera_cv_rotations=cv_rotations,
        camera_gl_quaternions=gl_quaternions,
    )
    return report
