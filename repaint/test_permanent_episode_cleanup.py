"""Regression check for permanent per-episode intermediate cleanup."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from repaint.nymeria.nymeria_vda_pipeline import (
    _clear_episode_intermediates,
)


def test_permanent_episode_cleanup() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        prepared = root / "prepared/episode_000007"
        handless_dir = root / "handless/episodes/episode_000007"
        prepared.mkdir(parents=True)
        handless_dir.mkdir(parents=True)
        (prepared / "episode_data.npz").touch()
        handless_video = handless_dir / "clip.mp4"
        handless_video.touch()
        plan = SimpleNamespace(
            config=SimpleNamespace(
                paths=SimpleNamespace(root=root, prepared=root / "prepared"),
                alignment_is_external=True,
            )
        )
        with patch(
            "repaint.nymeria.nymeria_vda_pipeline."
            "_repaint_episode_video_paths",
            return_value=(root / "source.mp4", handless_video),
        ):
            _clear_episode_intermediates(plan, 7)
        assert not prepared.exists()
        assert not handless_dir.exists()
        assert root.exists()


if __name__ == "__main__":
    test_permanent_episode_cleanup()
