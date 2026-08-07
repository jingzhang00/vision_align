"""Small shared helpers for importing the G1 articulation into Isaac Sim."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def import_g1(urdf_path: Path) -> str:
    """Import a floating-base G1 URDF and return its articulation prim path."""

    import omni.kit.commands

    status, import_config = omni.kit.commands.execute(
        "URDFCreateImportConfig"
    )
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")
    import_config.set_fix_base(False)
    import_config.set_import_inertia_tensor(True)
    import_config.set_merge_fixed_joints(False)
    import_config.set_parse_mimic(False)
    import_config.set_self_collision(False)
    import_config.set_collision_from_visuals(False)
    import_config.set_create_physics_scene(False)
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path.resolve()),
        import_config=import_config,
        dest_path="",
        get_articulation_root=True,
    )
    if not status or not prim_path:
        raise RuntimeError(f"Isaac failed to import {urdf_path}")
    return str(prim_path)


def isaac_joint_order(
    isaac_names: list[str],
    trajectory_names: tuple[str, ...],
) -> np.ndarray:
    """Map trajectory joint columns into Isaac's articulation ordering."""

    if len(set(isaac_names)) != len(isaac_names):
        raise ValueError("Isaac articulation contains duplicate joint names")
    missing = sorted(set(trajectory_names) - set(isaac_names))
    extra = sorted(set(isaac_names) - set(trajectory_names))
    if missing or extra:
        raise ValueError(
            "Isaac and LeRobot joint sets differ: "
            f"missing={missing}, extra={extra}"
        )
    source_index = {
        name: index for index, name in enumerate(trajectory_names)
    }
    return np.asarray(
        [source_index[name] for name in isaac_names],
        dtype=np.int64,
    )
