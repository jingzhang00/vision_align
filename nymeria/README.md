# Nymeria visual alignment

## Write the virtual pinhole intrinsics

The activity videos are already undistorted. `batch_undistort.py` now writes
only the matching virtual pinhole matrix; it never decodes, modifies, or
creates a video.

```bash
.venv/bin/python -m nymeria.batch_undistort \
  --dataset-root dataset/nymeria
```

Run one recording:

```bash
.venv/bin/python -m nymeria.batch_undistort \
  --dataset-root dataset/nymeria \
  --episode 20230814_s1_david_hall_act0_n9k3wo
```

The only generated file is:

```text
<episode>/undistorted/camera_intrinsics.json
```

The default 384×384 matrix is:

```text
K = [[260.0,   0.0, 191.5],
     [  0.0, 260.0, 191.5],
     [  0.0,   0.0,   1.0]]
```

An existing legacy JSON containing `undistorted.K` is accepted when it
matches. `--overwrite` replaces only `camera_intrinsics.json`; sibling videos
and other files are left untouched.

## Align activity clips and run human-to-G1 synthesis

Run all activity clips in one episode:

```bash
.venv/bin/python -m nymeria.visual_alignment \
  --episode-root dataset/nymeria/20230814_s1_david_hall_act0_n9k3wo \
  --output outputs/ \
  --isaacsim-root /home/ai01/isaacsim \
  --clip c024 \
  --debug
```

Run one clip:

```bash
.venv/bin/python -m nymeria.visual_alignment \
  --episode-root dataset/nymeria/20230814_s1_david_hall_act0_n9k3wo \
  --output outputs/ \
  --isaacsim-root /home/ai01/isaacsim \
  --clip c024
```

`c024` and `c025` are useful first checks because both camera-wearer wrists
are visible for almost every decoded frame. Clips without at least two valid
frames for each wrist are skipped by default.

The current G1 retargeter has a fixed base and frozen legs. A second default
filter therefore skips clips whose camera moves more than 0.25 m or rotates
more than 45° relative to the first frame, or whose hand target moves more
than 0.75 m. This prevents walking clips from producing unreachable arm
targets. Use `--no-motion-filter` only when a future moving-base/root
retargeter is available, or override the three `--maximum-*` limits
explicitly.

Use a fast preparation-only check before starting SAM3, ProPainter, or Isaac:

```bash
.venv/bin/python -m nymeria.visual_alignment \
  --episode-root dataset/nymeria/20230814_s1_david_hall_act0_n9k3wo \
  --output outputs/nymeria_prepare_check \
  --clip c024 \
  --prepare-only \
  --no-hardware-encoding
```

Preparation requires these recording-level inputs in the parent episode:

```text
metadata.json
body/xdata.npz
narration/activity_summarization.csv
recording_head/mps/slam/closed_loop_trajectory.csv
recording_head/mps/slam/online_calibration.jsonl
video.mp4
video_384_undistord.mp4
undistorted/camera_intrinsics.json
undistorted_384_activity_summarization_clips/*.mp4
```

The alignment policy is:

- Virtual pinhole `K` comes from `camera_intrinsics.json` and describes the
  upright MP4 pixels.
- Fisheye calibration and the raw `T_Device_Camera` come only from
  `recording_head`; every RGB calibration record is checked for consistency.
  Because Project Aria exports the upright RGB view with a clockwise 90-degree
  image rotation, the effective video-camera extrinsic is
  `T_Device_Camera @ RotZ(-90 degrees)`. `K` is not rotated again.
- `narration/activity_summarization.csv` supplies clip intervals, action
  text, and the timecode offset. Clip `cNNN` maps to CSV row `NNN`.
- Activity SMPL files are not read or required.
- A recording-level hand-eye solve registers the XSens head to the Aria
  device. Per-frame `T_World_Camera` from MPS is preserved in the adapter, so
  Isaac receives the real head-camera motion rather than a stabilized-camera
  approximation.
- `participant_height_cm` is read from Nymeria's `metadata.json` and retained
  as adapter provenance. Pelvis, Head, UpperArm, Forearm, and Hand XSens
  segments are registered into the same Aria world and exported as the common
  upper-body alignment anchors. The shared G1 retargeter derives its actual
  scale from the calibrated arm chains rather than treating participant
  height as camera height.
- Fixed-base compatibility and projected wrist visibility are checked before
  a clip is re-encoded or sent to the expensive synthesis stages.
- Nymeria's body data has rigid `L_Hand`/`R_Hand` tracks but no articulated
  finger pose. The adapter writes those explicit rigid-palm targets plus the
  body-alignment anchors; it does not synthesize fingertips. The shared
  retargeter maps them through its normal base search and 6D palm IK while
  keeping both robot hands open.

Some source clips contain hidden stream-copy preroll: the container may report
460 frames while only 450 are visible. The adapter normalizes only the decoded
visible frames, preserving the visible resolution and frame rate, so video and
HDF5 frame counts agree.

In normal mode, temporary adapter inputs and intermediates are removed and the
published outputs remain MP4/HDF5 pairs. `--debug` keeps the regular pipeline
diagnostics and `visual_alignment_report.json`. `--prepare-only` keeps the
normalized adapter and its report without running synthesis. Every batch stage
prints a progress bar and elapsed time.

Advanced pipeline tokens can be appended with the equals form, for example:

```bash
--pipeline-arg=--object-prompt --pipeline-arg="television remote control"
```
