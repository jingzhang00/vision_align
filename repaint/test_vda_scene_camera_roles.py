#!/usr/bin/env python3
"""Regression checks for shared VDA camera and output contracts."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from repaint.vda_scene_isaac import _resolve_mesh_camera_poses  # noqa: E402
from repaint.egodex.egodex_vda_pipeline import (  # noqa: E402
    _final_video_path as _egodex_final_video_path,
)
from repaint.nymeria.nymeria_vda_pipeline import (  # noqa: E402
    _final_video_path as _nymeria_final_video_path,
)


def test_robot_view_keeps_fixed_virtual_camera_extrinsic() -> None:
    prepared = np.zeros((6, 7), dtype=np.float64)
    robot = np.zeros((6, 7), dtype=np.float64)
    prepared[:, 6] = 1.0
    robot[:, 6] = 1.0
    robot[:, 0] = np.arange(6)
    prepared[:5, :3] = robot[:5, :3] + [0.0, 1.0, 0.0]
    prepared[5, :3] = 99.0
    robot[5, 3:7] = [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
    resolved, source = _resolve_mesh_camera_poses(
        capture_view="robot-camera",
        prepared_camera_poses_cv_xyzw=prepared,
        robot_camera_poses_cv_xyzw=robot,
    )
    assert np.allclose(resolved[:5], prepared[:5])
    assert np.allclose(resolved[5, :3], [4.0, 0.0, 0.0])
    assert np.allclose(resolved[5, 3:7], robot[5, 3:7])
    assert source.startswith(
        "prepared virtual source-camera pose backprojection rigidly following"
    )


def test_final_videos_keep_source_names() -> None:
    source = SimpleNamespace(video_path=Path("/source/original.mp4"))
    egodex = SimpleNamespace(
        paths=SimpleNamespace(final=Path("/output/final")),
        episodes=(source,),
    )
    nymeria = SimpleNamespace(
        config=SimpleNamespace(
            paths=SimpleNamespace(final=Path("/output/final"))
        ),
        clips=(SimpleNamespace(episode_index=0, source=source),),
    )
    assert _egodex_final_video_path(egodex, 0).name == "original.mp4"
    assert _nymeria_final_video_path(nymeria, 0).name == "original.mp4"


if __name__ == "__main__":
    test_robot_view_keeps_fixed_virtual_camera_extrinsic()
    test_final_videos_keep_source_names()
