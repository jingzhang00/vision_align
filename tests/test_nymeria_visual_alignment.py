from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from human_to_robot.pipeline import build_parser as build_pipeline_parser
from human_to_robot.trajectory import (
    RIGID_PALM_TARGET,
    hdf5_action_keypoint_availability,
    hdf5_hand_target_availability,
    hdf5_hand_target_mode,
    load_hand_trajectory,
)
from nymeria.visual_alignment import (
    ActivityAnnotation,
    CAMERA_RGB_FROM_VIDEO_CAMERA,
    CameraCalibration,
    ClipPair,
    HandEyeResult,
    _fixed_base_motion_metrics,
    _match_subsequence,
    _normalize_video,
    _pipeline_command,
    build_parser as build_alignment_parser,
    camera_space_hands,
    correct_xsens_timestamps,
    discover_clip_pairs,
    load_activity_narration,
    load_recording_camera,
    solve_hand_eye,
    world_space_hands,
    write_adapter_hdf5,
)


def _camera() -> CameraCalibration:
    return CameraCalibration(
        intrinsic=np.array(
            [
                [260.0, 0.0, 191.5],
                [0.0, 260.0, 191.5],
                [0.0, 0.0, 1.0],
            ]
        ),
        resolution=(384, 384),
        source_device_from_camera=np.eye(4),
        device_from_video_camera=np.eye(4),
        source_model="FisheyeRadTanThinPrism",
        source_parameters=np.arange(15, dtype=np.float64),
        source_record_count=2,
    )


def _hand_eye(transform: np.ndarray | None = None) -> HandEyeResult:
    return HandEyeResult(
        device_from_xsens_head=(
            np.eye(4) if transform is None else transform
        ),
        sample_count=100,
        median_rotation_error_degrees=0.0,
        p95_rotation_error_degrees=0.0,
        median_translation_error_m=0.0,
        p95_translation_error_m=0.0,
        maximum_time_error_us=0.0,
        timecode_to_device_offset_us=0.0,
    )


def _write_test_video(path: Path, *, frame_count: int = 6) -> None:
    size = (64, 48)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        size,
    )
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV mp4v encoder is unavailable")
    for index in range(frame_count):
        frame = np.full(
            (size[1], size[0], 3),
            20 * index,
            dtype=np.uint8,
        )
        writer.write(frame)
    writer.release()


def _write_narration(path: Path, count: int) -> None:
    rows = [
        "request_id,gaia_id,start_time,end_time,annotator,creation_time,"
        "Describe my activity"
    ]
    rows.extend(
        f"request,gaia,{465.0 + 30.0 * index},"
        f"{495.0 + 30.0 * index},annotator,0,action {index}"
        for index in range(count)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class NymeriaPairAndTimingTests(unittest.TestCase):
    def test_cli_requires_episode_not_smpl_directory(self) -> None:
        parsed = build_alignment_parser().parse_args(
            [
                "--episode-root",
                "/data/episode",
                "--output",
                "/data/output",
            ]
        )

        self.assertEqual(parsed.episode_root, Path("/data/episode"))
        self.assertIsNone(parsed.video_dir)
        self.assertFalse(hasattr(parsed, "smpl_dir"))

    def test_pairs_are_naturally_sorted_and_selectable_by_short_name(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            narration_path = root / "activity_summarization.csv"
            _write_narration(narration_path, 3)
            narration = load_activity_narration(narration_path)
            for suffix in ("c002", "c000", "c001"):
                (videos / f"recording_{suffix}.mp4").touch()

            pairs = discover_clip_pairs(videos, narration)
            selected = discover_clip_pairs(
                videos,
                narration,
                selected=["c002"],
            )

        self.assertEqual(
            [pair.short_name for pair in pairs],
            ["c000", "c001", "c002"],
        )
        self.assertEqual([pair.short_name for pair in selected], ["c002"])

    def test_pair_discovery_rejects_unpaired_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            narration_path = root / "activity_summarization.csv"
            _write_narration(narration_path, 2)
            narration = load_activity_narration(narration_path)
            (videos / "recording_c001.mp4").touch()

            with self.assertRaisesRegex(ValueError, "not one-to-one"):
                discover_clip_pairs(videos, narration)

    def test_xsens_timestamp_glitch_is_replaced_by_nominal_period(self) -> None:
        timestamps = np.array([1000, 5166, 9332, 20_000, 21_664])

        corrected = correct_xsens_timestamps(timestamps)

        np.testing.assert_array_equal(
            np.diff(corrected),
            np.array([4166, 4166, 4166, 4166]),
        )

    def test_clip_matching_expands_past_repeated_opening_frames(self) -> None:
        repeated = [b"same"] * 5
        false_opening = repeated + [b"false"]
        clip = repeated + [b"unique-a", b"unique-b"]
        full = false_opening + [b"gap"] * 3 + clip

        start = _match_subsequence(full, clip)

        self.assertEqual(start, len(false_opening) + 3)

    def test_narration_supplies_clip_timing_and_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "activity_summarization.csv"
            _write_narration(path, 2)
            narration = load_activity_narration(path)

        self.assertEqual(narration.timecode_offset_seconds, 465.0)
        self.assertEqual(narration.annotations[1].body_start_time, 30.0)
        self.assertEqual(narration.annotations[1].body_end_time, 60.0)
        self.assertEqual(narration.annotations[1].action_text, "action 1")


class NymeriaCalibrationTests(unittest.TestCase):
    @staticmethod
    def _record(focal_length: float = 1000.0) -> dict[str, object]:
        return {
            "CameraCalibrations": [
                {
                    "Label": "camera-rgb",
                    "Projection": {
                        "Name": "FisheyeRadTanThinPrism",
                        "Params": [focal_length, *range(1, 15)],
                    },
                    "T_Device_Camera": {
                        "Translation": [0.1, 0.2, 0.3],
                        "UnitQuaternion": [1.0, [0.0, 0.0, 0.0]],
                    },
                }
            ]
        }

    def test_recording_camera_scans_constant_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "online_calibration.jsonl"
            record = self._record()
            calibration.write_text(
                f"{json.dumps(record)}\n{json.dumps(record)}\n",
                encoding="utf-8",
            )
            intrinsic = root / "camera_intrinsics.json"
            intrinsic.write_text(
                json.dumps(
                    {
                        "camera_model": "pinhole",
                        "resolution": [384, 384],
                        "K": _camera().intrinsic.tolist(),
                    }
                ),
                encoding="utf-8",
            )

            camera = load_recording_camera(calibration, intrinsic)

        self.assertEqual(camera.source_record_count, 2)
        self.assertEqual(camera.source_model, "FisheyeRadTanThinPrism")
        self.assertEqual(camera.source_parameters[0], 1000.0)
        np.testing.assert_allclose(
            camera.source_device_from_camera[:3, 3],
            [0.1, 0.2, 0.3],
        )
        np.testing.assert_allclose(
            camera.source_device_from_camera[:3, :3],
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            camera.device_from_video_camera,
            camera.source_device_from_camera
            @ CAMERA_RGB_FROM_VIDEO_CAMERA,
            atol=1e-12,
        )

    def test_recording_camera_rejects_time_varying_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "online_calibration.jsonl"
            calibration.write_text(
                f"{json.dumps(self._record())}\n"
                f"{json.dumps(self._record(1000.1))}\n",
                encoding="utf-8",
            )
            intrinsic = root / "camera_intrinsics.json"
            intrinsic.write_text(
                json.dumps(
                    {
                        "resolution": [384, 384],
                        "K": _camera().intrinsic.tolist(),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Time-varying"):
                load_recording_camera(calibration, intrinsic)

    def test_upright_video_camera_rotates_sensor_coordinates_clockwise(
        self,
    ) -> None:
        camera = CameraCalibration(
            intrinsic=_camera().intrinsic,
            resolution=(384, 384),
            source_device_from_camera=np.eye(4),
            device_from_video_camera=CAMERA_RGB_FROM_VIDEO_CAMERA.copy(),
            source_model="FisheyeRadTanThinPrism",
            source_parameters=np.arange(15, dtype=np.float64),
            source_record_count=1,
        )
        parts = np.repeat(np.eye(4)[None, None], 1, axis=0)
        parts = np.repeat(parts, 3, axis=1)
        parts[0, 1, :3, 3] = [0.2, 0.1, 1.0]
        parts[0, 2, :3, 3] = [-0.3, 0.4, 1.0]

        left, right = camera_space_hands(
            parts,
            camera=camera,
            hand_eye=_hand_eye(),
        )

        np.testing.assert_allclose(
            left[0, :3, 3],
            [-0.1, 0.2, 1.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            right[0, :3, 3],
            [-0.4, -0.3, 1.0],
            atol=1e-12,
        )

    def test_hand_eye_recovers_known_transform(self) -> None:
        generator = np.random.default_rng(4)
        frame_count = 80
        xsens = np.repeat(np.eye(4)[None], frame_count, axis=0)
        xsens[:, :3, :3] = Rotation.random(
            frame_count,
            random_state=generator,
        ).as_matrix()
        xsens[:, :3, 3] = generator.normal(
            scale=0.5,
            size=(frame_count, 3),
        )
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_rotvec(
            [0.3, -0.2, 0.4]
        ).as_matrix()
        expected[:3, 3] = [0.08, -0.03, 0.12]
        world_alignment = np.eye(4)
        world_alignment[:3, :3] = Rotation.from_rotvec(
            [-0.5, 0.1, 0.2]
        ).as_matrix()
        world_alignment[:3, 3] = [1.0, 2.0, -0.2]
        device = world_alignment @ xsens @ np.linalg.inv(expected)

        actual, metrics = solve_hand_eye(
            device[:, :3, :3],
            device[:, :3, 3],
            xsens[:, :3, :3],
            xsens[:, :3, 3],
        )

        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        self.assertLess(metrics["p95_rotation_error_degrees"], 1e-10)
        self.assertLess(metrics["p95_translation_error_m"], 1e-12)

    def test_camera_space_chain_removes_xsens_head_motion(self) -> None:
        frames = np.repeat(np.eye(4)[None, None], 2, axis=0)
        frames = np.repeat(frames, 3, axis=1)
        frames[0, 0, :3, 3] = [1.0, 0.0, 0.0]
        frames[1, 0, :3, 3] = [2.0, 0.0, 0.0]
        frames[0, 1, :3, 3] = [1.0, 0.2, 1.0]
        frames[1, 1, :3, 3] = [2.0, 0.2, 1.0]
        frames[0, 2, :3, 3] = [1.0, -0.2, 1.0]
        frames[1, 2, :3, 3] = [2.0, -0.2, 1.0]

        left, right = camera_space_hands(
            frames,
            camera=_camera(),
            hand_eye=_hand_eye(),
        )

        np.testing.assert_allclose(left[:, :3, 3], [[0, 0.2, 1]] * 2)
        np.testing.assert_allclose(right[:, :3, 3], [[0, -0.2, 1]] * 2)

    def test_world_space_chain_preserves_camera_motion(self) -> None:
        parts = np.repeat(np.eye(4)[None, None], 3, axis=0)
        parts = np.repeat(parts, 3, axis=1)
        parts[:, 0, :3, 3] = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ]
        parts[:, 1, :3, 3] = parts[:, 0, :3, 3] + [0.0, 0.2, 1.0]
        parts[:, 2, :3, 3] = parts[:, 0, :3, 3] + [0.0, -0.2, 1.0]
        world_from_device = np.repeat(np.eye(4)[None], 3, axis=0)
        world_from_device[:, :3, :3] = Rotation.from_euler(
            "z",
            [0.0, 20.0, 40.0],
            degrees=True,
        ).as_matrix()
        world_from_device[:, :3, 3] = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.2, 0.0],
        ]

        direct_left, direct_right = camera_space_hands(
            parts,
            camera=_camera(),
            hand_eye=_hand_eye(),
        )
        world_left, world_right, camera_to_world = world_space_hands(
            parts,
            world_from_device,
            camera=_camera(),
            hand_eye=_hand_eye(),
        )
        world_to_camera = np.linalg.inv(camera_to_world)
        observed_left = world_to_camera @ world_left
        observed_right = world_to_camera @ world_right

        np.testing.assert_allclose(observed_left, direct_left, atol=1e-12)
        np.testing.assert_allclose(observed_right, direct_right, atol=1e-12)
        metrics = _fixed_base_motion_metrics(
            camera_to_world=camera_to_world,
            left_hands=world_left,
            right_hands=world_right,
        )
        self.assertGreater(metrics["maximum_camera_translation_m"], 1.0)
        self.assertGreater(metrics["maximum_camera_rotation_degrees"], 30.0)


class NymeriaAdapterTests(unittest.TestCase):
    def test_normalization_keeps_visible_frames_rate_and_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            output = root / "output.mp4"
            _write_test_video(source)

            count = _normalize_video(
                source,
                output,
                hardware_encoding=False,
            )
            with av.open(str(output)) as container:
                stream = container.streams.video[0]
                decoded = list(container.decode(stream))

            self.assertEqual(count, 6)
            self.assertEqual(len(decoded), 6)
            self.assertEqual((decoded[0].width, decoded[0].height), (64, 48))
            self.assertEqual(float(stream.average_rate), 12.0)
            self.assertEqual(decoded[0].pts, 0)

    def test_adapter_declares_rigid_palm_without_fake_fingertips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "clip.hdf5"
            frame_count = 8
            left = np.repeat(np.eye(4)[None], frame_count, axis=0)
            right = left.copy()
            left[:, :3, 3] = [0.1, 0.2, 1.0]
            right[:, :3, 3] = [0.1, -0.2, 1.0]
            body_transforms = {}
            for name, position in {
                "hip": [0.0, 0.0, 0.3],
                "neck4": [0.0, 0.0, 0.9],
                "leftArm": [0.0, 0.2, 0.6],
                "leftForearm": [0.1, 0.2, 0.8],
                "rightArm": [0.0, -0.2, 0.6],
                "rightForearm": [0.1, -0.2, 0.8],
            }.items():
                transforms = np.repeat(
                    np.eye(4)[None],
                    frame_count,
                    axis=0,
                )
                transforms[:, :3, 3] = position
                body_transforms[name] = transforms
            pair = ClipPair(
                stem="recording_c000",
                video=root / "recording_c000.mp4",
                annotation=ActivityAnnotation(
                    row_index=0,
                    original_start_time=465.0,
                    original_end_time=495.0,
                    body_start_time=0.0,
                    body_end_time=30.0,
                    action_text="test action",
                ),
            )
            write_adapter_hdf5(
                path,
                pair=pair,
                narration_path=root / "activity_summarization.csv",
                camera=_camera(),
                camera_to_world=np.repeat(
                    np.eye(4)[None],
                    frame_count,
                    axis=0,
                ),
                left_hands=left,
                right_hands=right,
                body_transforms=body_transforms,
                left_confidence=np.ones(frame_count, dtype=np.float32),
                right_confidence=np.ones(frame_count, dtype=np.float32),
                body_times=np.arange(frame_count) / 15.0,
                source_frame_indices=np.arange(frame_count),
                participant_height_m=1.81,
            )

            keypoint_availability = hdf5_action_keypoint_availability(path)
            target_availability = hdf5_hand_target_availability(path)
            target_mode = hdf5_hand_target_mode(path)
            trajectory = load_hand_trajectory(path)
            with h5py.File(path, "r") as handle:
                datasets = set()
                handle.visit(datasets.add)

        self.assertEqual(
            keypoint_availability,
            {"left": False, "right": False},
        )
        self.assertEqual(
            target_availability,
            {"left": True, "right": True},
        )
        self.assertEqual(target_mode, RIGID_PALM_TARGET)
        self.assertEqual(trajectory.target_mode, RIGID_PALM_TARGET)
        self.assertIsNotNone(trajectory.body_alignment)
        assert trajectory.body_alignment is not None
        self.assertEqual(
            trajectory.body_alignment.source,
            "nymeria_xsens_calibrated_upper_body",
        )
        np.testing.assert_array_equal(
            trajectory.body_alignment.world_up,
            [0.0, 0.0, 1.0],
        )
        for side in ("left", "right"):
            for name in ("ThumbTip", "IndexFingerTip", "MiddleFingerTip"):
                self.assertNotIn(f"transforms/{side}{name}", datasets)
        self.assertEqual(trajectory.frame_count, frame_count)
        np.testing.assert_allclose(
            trajectory.left_reference_camera[:, 2],
            1.0,
        )
        np.testing.assert_allclose(
            trajectory.left_action_reference_camera,
            trajectory.left_reference_camera,
        )
        np.testing.assert_allclose(
            trajectory.right_action_reference_rotations,
            trajectory.right_reference_rotations,
        )
        self.assertTrue(np.isnan(trajectory.left_widths).all())
        self.assertTrue(np.isnan(trajectory.right_widths).all())

    def test_generated_pipeline_command_round_trips_real_parser(self) -> None:
        args = argparse.Namespace(
            arm_prompt="first-person human arm and hand",
            sam_threshold=0.4,
            hardware_encoding=False,
            isaacsim_root=Path("/opt/isaacsim"),
            debug=False,
            reuse=False,
            pipeline_arg=[],
        )

        command = _pipeline_command(
            adapter_root=Path("/tmp/adapter"),
            output=Path("/tmp/output"),
            args=args,
        )
        parsed = build_pipeline_parser().parse_args(command[3:])

        self.assertEqual(parsed.folder, Path("/tmp/adapter"))
        self.assertEqual(parsed.output, Path("/tmp/output"))
        self.assertNotIn("--width", command)
        self.assertNotIn("--height", command)
        self.assertFalse(parsed.hardware_encoding)
        self.assertFalse(parsed.reuse)


if __name__ == "__main__":
    unittest.main()
