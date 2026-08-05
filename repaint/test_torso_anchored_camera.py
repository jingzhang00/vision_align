"""Focused regressions for the torso-anchored virtual camera contract."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from repaint.nymeria.nymeria_adapter import frame0_anchored_spine3
from repaint.robot_camera import (
    TORSO_ANCHORED_CAMERA_MAPPING,
    source_camera_poses_in_spine3_cv,
    torso_anchored_camera_poses_cv,
)
from repaint.rgbd_prepare import build_parser as build_prepare_parser
from repaint.vda_scene_isaac import (
    _resolve_mesh_camera_poses,
    _validate_torso_anchored_camera_chain,
    build_parser as build_render_parser,
)


def test_nymeria_spine_anchor_uses_every_frame_spine_pose() -> None:
    frame_count = 3
    aligned_head = np.repeat(np.eye(4)[None], frame_count, axis=0)
    aligned_head[:, :3, :3] = Rotation.from_euler(
        "z", [30.0, 45.0, 60.0], degrees=True
    ).as_matrix()
    aligned_head[:, :3, 3] = [
        [4.0, 5.0, 6.0],
        [5.0, 5.0, 6.0],
        [6.0, 5.0, 6.0],
    ]
    motion_head = np.repeat(np.eye(4)[None], frame_count, axis=0)
    motion_head[:, :3, 3] = [
        [0.0, 0.8, 0.0],
        [1.0, 0.8, 0.0],
        [2.0, 0.8, 0.0],
    ]
    motion_spine = np.repeat(np.eye(4)[None], frame_count, axis=0)
    motion_spine[:, :3, :3] = Rotation.from_euler(
        "y", [0.0, 20.0, 40.0], degrees=True
    ).as_matrix()
    motion_spine[:, :3, 3] = motion_head[:, :3, 3] + [0.0, -0.4, 0.0]

    anchored = frame0_anchored_spine3(
        aligned_head_to_world=aligned_head,
        motion_head_to_world=motion_head,
        motion_spine3_to_world=motion_spine,
    )

    world_from_motion = aligned_head[0, :3, :3]
    assert np.allclose(
        anchored[:, :3, :3],
        np.einsum("ij,njk->nik", world_from_motion, motion_spine[:, :3, :3]),
    )
    assert np.allclose(
        anchored[:, :3, 3],
        aligned_head[:, :3, 3]
        + np.einsum(
            "ij,nj->ni",
            world_from_motion,
            motion_spine[:, :3, 3] - motion_head[:, :3, 3],
        ),
    )


def _source_camera_poses_in_spine3(frame_count: int) -> np.ndarray:
    positions = np.column_stack(
        [
            np.linspace(0.15, 0.25, frame_count),
            np.full(frame_count, -0.05),
            np.linspace(0.45, 0.35, frame_count),
        ]
    )
    rotations = Rotation.from_euler(
        "zyx",
        np.column_stack(
            [
                np.linspace(0.0, 30.0, frame_count),
                np.linspace(0.0, -15.0, frame_count),
                np.zeros(frame_count),
            ]
        ),
        degrees=True,
    ).as_quat()
    return np.concatenate([positions, rotations], axis=1)


def _robot_torso_poses(frame_count: int) -> np.ndarray:
    positions = np.column_stack(
        [
            np.linspace(-0.2, 0.4, frame_count),
            np.linspace(0.3, 0.1, frame_count),
            np.full(frame_count, 0.85),
        ]
    )
    rotations = Rotation.from_euler(
        "xyz",
        np.column_stack(
            [
                np.linspace(0.0, 20.0, frame_count),
                np.zeros(frame_count),
                np.linspace(0.0, 60.0, frame_count),
            ]
        ),
        degrees=True,
    ).as_quat()
    return np.concatenate([positions, rotations], axis=1)


def test_torso_anchor_applies_source_local_pose_to_robot_torso() -> None:
    frame_count = 7
    source = _source_camera_poses_in_spine3(frame_count)
    torso = _robot_torso_poses(frame_count)
    basis = Rotation.from_euler("z", 35.0, degrees=True).as_matrix()
    scale = 0.82

    camera = torso_anchored_camera_poses_cv(
        source_camera_pose_in_spine3_xyzw=source,
        robot_torso_pose_world_xyzw=torso,
        source_spine3_to_robot_torso_rotation=basis,
        scene_scale=scale,
    )
    torso_rotation = Rotation.from_quat(torso[:, 3:7]).as_matrix()
    source_rotation = Rotation.from_quat(source[:, 3:7]).as_matrix()
    torso_from_source = np.einsum("nij,jk->nik", torso_rotation, basis)
    expected_position = torso[:, :3] + scale * np.einsum(
        "nij,nj->ni", torso_from_source, source[:, :3]
    )
    expected_rotation = np.einsum(
        "nij,njk->nik",
        torso_from_source,
        source_rotation,
    )
    assert np.allclose(camera[:, :3], expected_position)
    assert np.allclose(
        Rotation.from_quat(camera[:, 3:7]).as_matrix(),
        expected_rotation,
    )


def test_source_torso_rotation_is_removed_before_g1_torso_is_applied() -> None:
    frame_count = 4
    source_spine = np.repeat(np.eye(4)[None], frame_count, axis=0)
    source_spine[:, :3, :3] = Rotation.from_euler(
        "z", [0.0, 30.0, 70.0, 110.0], degrees=True
    ).as_matrix()
    camera_in_spine = np.repeat(np.eye(4)[None], frame_count, axis=0)
    camera_in_spine[:, :3, 3] = [0.0, 0.0, 0.4]
    camera_in_spine[:, :3, :3] = Rotation.from_euler(
        "y", [10.0, 10.0, 10.0, 25.0], degrees=True
    ).as_matrix()
    source_camera = np.einsum(
        "nij,njk->nik", source_spine, camera_in_spine
    )

    recovered = source_camera_poses_in_spine3_cv(
        source_camera_to_world=source_camera,
        source_spine3_to_world=source_spine,
    )
    assert np.allclose(recovered[:, :3], camera_in_spine[:, :3, 3])
    assert np.allclose(
        Rotation.from_quat(recovered[:, 3:7]).as_matrix(),
        camera_in_spine[:, :3, :3],
    )


def test_exact_torso_anchor_validator_and_mesh_contract() -> None:
    frame_count = 5
    source = _source_camera_poses_in_spine3(frame_count)
    torso = _robot_torso_poses(frame_count)
    basis = Rotation.from_euler("z", -20.0, degrees=True).as_matrix()
    scale = 0.9
    camera = torso_anchored_camera_poses_cv(
        source_camera_pose_in_spine3_xyzw=source,
        robot_torso_pose_world_xyzw=torso,
        source_spine3_to_robot_torso_rotation=basis,
        scene_scale=scale,
    )
    validation = dict(
        source_camera_pose_in_spine3_xyzw=source,
        source_spine3_to_robot_torso_rotation=basis,
        scene_scale=scale,
        g1_urdf=Path("unused-in-patched-fk.urdf"),
        joint_names=("joint",),
        joint_positions=np.zeros((frame_count, 1)),
        root_poses_xyzw=np.repeat(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            frame_count,
            axis=0,
        ),
    )
    with patch(
        "repaint.vda_scene_isaac.robot_frame_poses_world",
        return_value=torso,
    ):
        _validate_torso_anchored_camera_chain(
            camera_poses_cv_xyzw=camera,
            **validation,
        )
        displaced = camera.copy()
        displaced[-1, 0] += 0.01
        try:
            _validate_torso_anchored_camera_chain(
                camera_poses_cv_xyzw=displaced,
                **validation,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid torso-anchored position was accepted")

    resolved, source_description = _resolve_mesh_camera_poses(
        capture_view="robot-camera",
        prepared_camera_poses_cv_xyzw=camera,
        camera_mapping_mode=TORSO_ANCHORED_CAMERA_MAPPING,
    )
    assert resolved is camera
    assert source_description.startswith(
        "prepared source spine3-to-camera local pose transferred onto G1"
    )
    try:
        _resolve_mesh_camera_poses(
            capture_view="robot-camera",
            prepared_camera_poses_cv_xyzw=camera,
            camera_mapping_mode="removed_legacy_mapping",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("removed legacy camera mapping was accepted")


def test_removed_camera_overrides_are_not_cli_options() -> None:
    prepare = build_prepare_parser().parse_args(
        [
            "--dataset",
            "robot",
            "--source-root",
            "source",
            "--removed-root",
            "handless",
            "--output",
            "prepared",
        ]
    )
    assert not hasattr(prepare, "scene_scale")
    assert not hasattr(prepare, "field")
    render = build_render_parser().parse_args(["--prepared", "prepared"])
    assert not hasattr(render, "scene_alignment")

    removed_arguments = (
        (build_prepare_parser, ["--scene-scale", "0.8"]),
        (build_prepare_parser, ["--field", "action"]),
        (build_render_parser, ["--scene-alignment", "none"]),
    )
    for parser_factory, argument in removed_arguments:
        parser = parser_factory()
        required = (
            [
                "--dataset",
                "robot",
                "--source-root",
                "source",
                "--removed-root",
                "handless",
                "--output",
                "prepared",
            ]
            if parser_factory is build_prepare_parser
            else ["--prepared", "prepared"]
        )
        with redirect_stderr(StringIO()):
            try:
                parser.parse_args([*required, *argument])
            except SystemExit:
                pass
            else:
                raise AssertionError(
                    f"removed CLI argument was accepted: {argument}"
                )


if __name__ == "__main__":
    test_torso_anchor_applies_source_local_pose_to_robot_torso()
    test_source_torso_rotation_is_removed_before_g1_torso_is_applied()
    test_exact_torso_anchor_validator_and_mesh_contract()
    test_removed_camera_overrides_are_not_cli_options()
