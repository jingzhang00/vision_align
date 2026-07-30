### EgoDex upper body to Unitree G1 (LeRobot v3)

`stream/scripts/egodex_to_robot.py` consumes same-stem EgoDex `.hdf5` and `.mp4`
pairs and retargets them directly to the 51-DoF Unitree G1 with articulated
hands. GMR solves the 29 body joints from the tracked upper body plus a static
G1-derived standing target; the existing 25-landmark O6 mapping supplies the
22 hand joints. No SMPL-X model or fitting stage is used. The moving camera is
mapped into the same robot-world frame and everything is written as a
frame-aligned LeRobot v3.0 dataset:

```bash
uv run python stream/scripts/egodex_to_robot.py \
  --src-root dataset/flip_coin \
  --output-root output_new/flip_coin
```

```bash
uv run python repaint/traj_replay.py \
  --dataset output_new/flip_coin
```

For the full-URDF replay in Isaac Sim, run Isaac with the workspace virtual
environment so both Isaac and the LeRobot dependencies are available:

```bash
PYTHONEXE="$PWD/.venv/bin/python" \
  /home/ai01/isaacsim/python.sh repaint/isaac_replay.py \
  --dataset output_new/flip_coin
```

The GUI replay loops at the dataset FPS until the window is closed. Add
`--once` to exit after one pass, or `--headless` for a one-pass startup check.
The script imports `main_with_hand_new.urdf` as a floating-base articulation
and replays all 51 body and hand joints plus `observation.root_pose`.

Useful options:

```bash
# Only create the static overview.
uv run python repaint/traj_replay.py \
  --dataset output_new/ --episode 1 --overview-only

# Visualize only frames [10, 40).
uv run python repaint/traj_replay.py \
  --dataset output_new/ --episode 0 --start-frame 10 --end-frame 40

# Visualize next-frame action targets instead of current states.
uv run python repaint/traj_replay.py \
  --dataset output_new/ --episode 0 --field action
```

The default dataset is `output_new/basic_pick_place`; `--dataset` is only
needed for a different LeRobot v3 root.
