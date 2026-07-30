### EgoDex upper body to Unitree G1 (LeRobot v3)

`stream/scripts/egodex_to_robot.py` consumes same-stem EgoDex `.hdf5` and `.mp4`
pairs, fits the local neutral SMPL-X model, retargets that fitted trajectory to
the 29-DoF Unitree G1, maps the moving camera into the same robot-world frame,
and writes a frame-aligned LeRobot v3.0 dataset:

```bash
uv run python stream/scripts/egodex_to_robot.py \
  --src-root dataset/pour \
  --output-root output_new/
```

The script loads SMPL-X code from `../smplx` and always uses
`../models/models/smplx/SMPLX_NEUTRAL.npz` with 16 zero shape coefficients.
The source sample has no lower-body tracks, so its fitted SMPL-X lower body is
neutral before the standard full-body SMPL-X-to-G1 GMR stage.
`observation.state` is the current G1 kinematic joint-position reference;
`action` is the next-frame reference.
Arm retargeting reuses GMR's standard SMPL-X scaling and solves from
elbow/wrist positions because EgoDex landmarks do not define G1 link-frame
orientations.
Fitted SMPL-X parameters, the original RGB video, mapped OpenCV
`T_world_camera`, and camera intrinsics are retained for later Isaac replay
and repainting. Convert camera rotation with
`R_usd = R_cv @ diag(1, -1, -1)` before assigning it to an Isaac/USD camera.

### Lightweight trajectory replay (MuJoCo, no Isaac)

`repaint/traj_replay.py` reads the saved LeRobot v3 trajectory directly. Each
frame copies `observation.root_pose` and the 29-DoF `observation.state` into the
Unitree G1 MuJoCo model and runs kinematic forward propagation. It does not run
IK, a controller, a physics rollout, or Isaac rendering.

Open the interactive MuJoCo viewer:

```bash
uv run python repaint/traj_replay.py \
  --dataset dataset/basic_pick_place_g1_lerobot_v3 \
  --episode 0
```

Viewer controls:

- `Space`: pause/resume.
- `A` or `Left`: previous frame.
- `D` or `Right`: next frame.
- `R`: restart the selected frame range.
- `Q`: quit.

Useful options:

```bash
# Loop episode 1 at half speed.
uv run python repaint/traj_replay.py --episode 1 --speed 0.5 --loop

# Replay only frames [10, 40).
uv run python repaint/traj_replay.py \
  --episode 0 --start-frame 10 --end-frame 40

# No window: run every frame through MuJoCo kinematics and exit.
uv run python repaint/traj_replay.py --episode 0 --headless

# Replay next-frame action targets instead of current states.
uv run python repaint/traj_replay.py --episode 0 --field action
```

The default dataset is
`dataset/basic_pick_place_g1_lerobot_v3`; `--dataset` is only needed for a
different LeRobot v3 root. When replaying once, the viewer holds the final
frame until `R`, `Q`, or the window close button is used.
