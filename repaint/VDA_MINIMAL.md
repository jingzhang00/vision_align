# Video-Depth-Anything minimal scene validation

This validation intentionally covers only `basic_pick_place` episode 0:

- handless source: 63 frames, 1920×1080, 30 FPS;
- depth: offline Metric Video-Depth-Anything-Large, one continuous pass for
  each of the original and handless streams;
- inference frames: 1280×720, model input size 518, CUDA autocast;
- Isaac mesh: nearest-neighbor resampled 480×270 metric camera-Z depth,
  calibrated per episode from HDF5 hand depth and unchanged background,
  transformed by the recorded world-from-camera poses;
- visualization: dynamic RGB-D scene plus the frame-aligned 51-DoF G1
  trajectory, viewed from a fixed third-person camera.

## Reproduce

From the workspace root:

```bash
.venv/bin/python -m repaint.vda_scene_prepare \
  --input-video \
  output_new/basic_pick_place/videos_handless/observation.images.ego/chunk-000/file-000.mp4 \
  --template-prepared \
  output_new/basic_pick_place/rgbd_trial/replay_processed_frames \
  --output \
  output_new/basic_pick_place/vda_minimal/episode_000000/prepared_auto_hand_anchor_final
```

```bash
.venv/bin/python -m repaint.vda_continuity \
  --prepared \
  output_new/basic_pick_place/vda_minimal/episode_000000/prepared_auto_hand_anchor_final \
  --output \
  output_new/basic_pick_place/vda_minimal/episode_000000/metrics/vda_auto_hand_anchor_final_continuity.json
```

```bash
.venv/bin/python -m repaint.vda_scene_isaac \
  --prepared \
  output_new/basic_pick_place/vda_minimal/episode_000000/prepared_auto_hand_anchor_final \
  --output \
  output_new/basic_pick_place/vda_minimal/episode_000000/isaac/vda_auto_hand_aligned_final.mp4 \
  --render-size 960 540 \
  --scene-eye -0.04 -1.23 1.81 \
  --scene-target 0.85 -0.23 1.25 \
  --scene-alignment initial-hands \
  --headless
```

For an adjustable Isaac viewport that plays at the prepared FPS and loops
until the window is closed:

```bash
.venv/bin/python -m repaint.vda_scene_isaac \
  --prepared \
  output_new/basic_pick_place/vda_minimal/episode_000000/prepared_auto_hand_anchor_final \
  --render-size 960 540 \
  --scene-eye -0.04 -1.23 1.81 \
  --scene-target 0.85 -0.23 1.25 \
  --scene-alignment initial-hands \
  --interactive \
  --loop \
  --no-headless
```

The G1 root pose and all 51 joints are read from `episode_data.npz` and applied
on the same local frame as each RGB-D mesh update. The interactive mode does
not write an MP4. Its camera arguments define only the initial view; the free
Isaac viewport camera remains under mouse control.
Use Alt+left-drag to orbit, middle-drag to pan, right-drag to look, and the
wheel or Alt+right-drag to zoom. Without `--loop`, playback stops on the last
frame while the window remains open. Frames are never skipped; if rendering
cannot sustain the requested rate, playback slows down instead.

Preparation defaults to `--vda-scale-mode auto`. It runs the same metric VDA
configuration on the original and handless selections, anchors original depth
to source-HDF5 hands, and transfers that scale through unchanged background.
Only a calibration that passes the temporal/support quality gates is used.
`scene_scale` remains a separate source-to-G1 conversion.
The handless video and HDF5 must have the same total frame count. If the
original container declares a different count, the HDF5 must name that exact
`source_video`. A genuinely longer decodable video also needs an explicit
`source_frame_start=0`; Nymeria c024 instead declares 460 but decodes exactly
the same 450 frames as its HDF5/handless sequence.

Replay defaults to `--scene-alignment initial-hands`. Before Isaac starts, it
uses the first 15 frames once, fits the robust world translation between the
source and G1 bilateral-hand midpoints, and applies that translation only to
the complete scene mesh. It does not move the robot, refit scale, or track the
hands during playback. Use `--scene-alignment none` only for legacy prepared
archives or an intentional unaligned comparison.

`easydict` is required by the upstream VDA temporal head. The checkpoint used
for this run is
`Video-Depth-Anything/checkpoints/metric_video_depth_anything_vitl.pth`
(SHA-256
`24eba25342e7ee0f054be25da6a852ebd7cfa40cd6bf41a353ecb6abfe24e620`).

## Result

All entries below summarize the same 62 adjacent frame pairs. Lower is better.

| Metric | DA3 baseline | Video Depth Anything |
|---|---:|---:|
| Frame-median depth jump, p50 | 1.95 cm | 0.30 cm |
| Frame-median depth jump, p95 | 6.69 cm | 0.99 cm |
| Frame-median depth jump, max | 11.50 cm | 1.30 cm |
| Median-depth symmetric ratio, p95 | 1.184 | 1.029 |
| Flow-warped AbsRel p95, sequence p95 | 22.14% | 5.53% |
| World symmetric-NN p50, sequence p95 | 7.15 cm | 0.71 cm |
| World symmetric-NN p95, sequence p95 | 12.96 cm | 1.74 cm |

The Isaac outputs contain exactly 63 frames at 30 FPS (2.1 seconds). The VDA
scene is substantially more temporally continuous than the DA3 baseline, so
this minimal check passes. It is not perfectly rigid: the worst transition is
frame 43→44, with a 1.3 cm frame-median depth jump and 0.98 cm world-NN median.
This coincides with an offline VDA window boundary and is visible as a mild
surface adjustment, not the large scale breathing present in the DA3 result.
The fixed-scale/unaligned versus automatic-scale/hand-aligned comparison is
`isaac/vda_fixed_vs_auto_hand_aligned_final_sameview.mp4`.

For this episode, the automatic raw-VDA-to-source scale is `0.455253`; the
separate source-to-G1 `scene_scale` is `0.8`. Initial-hand placement applies
`[-4.6, +26.3, -114.0]` mm once to the scene. Bilateral-midpoint p95 error in
the 15-frame fit window falls from 11.78 cm to 0.25 cm. The remaining
individual-hand median error is 7.39 cm and is reported only as a
human-to-G1 morphology/retarget diagnostic; scene placement does not pretend
to repair that IK error.

The same automatic scale estimator was also checked on the requested
cross-task data. These reports were regenerated with cache manifests binding
the source/handless videos, checkpoint, inference settings, and raw-depth
files by SHA-256:

| Task / episode | Automatic VDA scale | Quality result |
|---|---:|---|
| `insert_remove_usb` episode 0 | 0.40230 | validated |
| `insert_remove_usb` episode 1 | 0.57998 | validated |
| `pour` episode 21 | 0.40323 | validated |
| `pour` episode 22 | 0.45610 | validated |
| Nymeria `c024` | 0.48734 | validated |

These differing values are why `0.525` is retained only as an explicit
legacy comparison, not a default shared across tasks.

## Limits

- The legacy fixed `0.525` remains available only with
  `--vda-scale-mode fixed`; it is not a universal monocular metric-depth
  constant and is no longer the preparation default.
- The hand-removal video contains blur, shadows, and residual artifacts in the
  removed regions. Those artifacts create some holes and unstable object-edge
  geometry in the third-person mesh.
- Only episode 0 was run. Episode 1 must be tested before treating the result
  as task-wide validation.
