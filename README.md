# EgoDex / Nymeria to G1

The supported end-to-end entrypoints are:

- `repaint.egodex.egodex_vda_pipeline`
- `repaint.nymeria.nymeria_vda_pipeline`
- `repaint/egodex/run_single_task.py`
- `repaint/nymeria/run_single_task.py`

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
support-plane leveling disabled, then aligns the bilateral
`leftHand/rightHand` rigid-palm action frames to the matching replayed G1
canonical palms with one XYZ scene translation.

## Run the four EgoDex tasks

```bash
.venv/bin/python -m repaint.egodex.egodex_vda_pipeline \
  --tasks basic_pick_place flip_coin insert_remove_usb pour \
  --dataset-root dataset \
  --output-root egodex \
  --headless
```

To process episode 0 of every task under `dataset/egodex_test`:

```bash
bash batch_egodex.sh
```

## Run one task and retain intermediates

These commands retain alignment (Nymeria), LeRobot, handless, prepared, and
final artifacts. Isaac is visible by default; add `--headless` for batch use.

```bash
.venv/bin/python repaint/egodex/run_single_task.py \
  --task pour \
  --dataset-root dataset \
  --output-root egodex_single_task

.venv/bin/python repaint/nymeria/run_single_task.py \
  --task c000 \
  --session 20230814_s1_david_hall_act0_n9k3wo
```

Use `--overwrite` only when the selected output should be rebuilt from
scratch.

## Fast trajectory-only check

```bash
.venv/bin/python -m repaint.traj_replay \
  --dataset egodex_single_task/pour/lerobot \
  --episode 0

.venv/bin/python -m repaint.traj_replay \
  --dataset egodex
```

## Adjustable Isaac view

To inspect a prepared episode interactively:

```bash
.venv/bin/python -m repaint.vda_scene_isaac \
  --prepared egodex_single_task/pour/prepared/episode_000000 \
  --interactive \
  --loop \
  --no-headless

.venv/bin/python -m repaint.vda_scene_isaac \
  --prepared nymeria/single_task/prepared/episode_000000 \
  --interactive \
  --loop \
  --no-headless \
  --scene-leveling none \
  --scene-alignment initial-palms
```

To render one prepared episode directly:

```bash
.venv/bin/python -m repaint.vda_scene_isaac \
  --prepared egodex_single_task/pour/prepared/episode_000000 \
  --output egodex_single_task/pour/manual/g1_view.mp4 \
  --capture-view robot-camera \
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

To run the complete Nymeria dataset with the standard output location:

```bash
bash batch_nymeria.sh
```
