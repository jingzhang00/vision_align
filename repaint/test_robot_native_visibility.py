from __future__ import annotations

import inspect
import unittest

from repaint.rgbd_prepare import prepare
from repaint.vda_scene_isaac import PreparedScene, load_prepared_scene, render


class RobotNativeVisibilityTests(unittest.TestCase):
    def test_robot_visibility_is_not_timeline_state(self) -> None:
        self.assertNotIn(
            "robot_arm_render",
            PreparedScene.__dataclass_fields__,
        )
        self.assertNotIn("robot_arm_render", inspect.getsource(prepare))
        self.assertNotIn("robot_arm_render", inspect.getsource(render))

        self.assertNotIn(
            "robot_arm_render",
            inspect.getsource(load_prepared_scene),
        )


if __name__ == "__main__":
    unittest.main()
