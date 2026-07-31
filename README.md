# EgoDex / Nymeria to G1

The only supported end-to-end entrypoints are:

- `repaint.egodex.egodex_vda_pipeline`
- `repaint.nymeria.nymeria_vda_pipeline`

Both use the same VDA route:

```text
dataset motion + MP4
  -> G1 retarget + LeRobot v3
  -> shared human-arm removal
  -> Metric Video Depth Anything
  -> dataset-specific semantic scene anchor
  -> Isaac G1 + exact BeingBeyond torso/head-camera replay
```

EgoDex reads same-stem HDF5/MP4 pairs. Nymeria reads same-stem
MP4/208-D/SMPL clips while retaining the recording-level Aria trajectory and
calibration files for camera alignment.

EgoDex first levels the support surface, then aligns the exact bilateral
retarget TCP (`0.5*thumb + 0.35*index + 0.15*middle`) in XYZ. Nymeria keeps
support-plane leveling and wrist-derived scene translation disabled, preserving
the calibrated scene coordinates while replaying the G1 robot camera.

## Run the four EgoDex tasks

```bash
.venv/bin/python -m repaint.egodex.egodex_vda_pipeline \
  --tasks basic_pick_place flip_coin insert_remove_usb pour \
  --dataset-root dataset \
  --output-root egodex \
  --headless

```

## Fast trajectory-only check

```bash
.venv/bin/python -m repaint.traj_replay \
  --dataset egodex/basic_pick_place/lerobot \
  --episode 0

.venv/bin/python -m repaint.traj_replay \
  --dataset egodex
```

## Adjustable Isaac view

To inspect a prepared episode interactively:

```bash
.venv/bin/python -m repaint.vda_scene_isaac \
  --prepared egodex/basic_pick_place/prepared/episode_000000 \
  --interactive \
  --loop \
  --no-headless

.venv/bin/python -m repaint.vda_scene_isaac \
  --prepared nymeria/vda/prepared/episode_000000 \
  --interactive \
  --loop \
  --no-headless \
  --scene-leveling none \
  --scene-alignment none
```

To render one prepared episode directly:

```bash
.venv/bin/python -m repaint.vda_scene_isaac \
  --prepared egodex/basic_pick_place/prepared/episode_000000 \
  --output egodex/basic_pick_place/final/episode_000000/g1_view.mp4 \
  --headless
```


## Nymeria clips

```bash
.venv/bin/python -m repaint.nymeria.nymeria_vda_pipeline \
  --dataset-root dataset/nymeria \
  --output-root nymeria/vda \
  --session 20230814_s1_david_hall_act0_n9k3wo \
  --clip c024
```

Camera alignment and its reports are generated under
`<output-root>/camera_alignment` and `<output-root>/alignment_reports`.
Existing valid stages are resumed automatically, and the command always runs
through the final robot-camera render.
