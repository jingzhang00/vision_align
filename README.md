# vision_align

## Human-to-G1 synthesis


The local models expected by the pipeline are:

```text
models/sam3/sam3.pt
ProPainter/weights/raft-things.pth
ProPainter/weights/recurrent_flow_completion.pth
ProPainter/weights/ProPainter.pth
```

Every episode HDF5 must provide 3D transforms and confidences for
`Hand`, `ThumbTip`, `IndexFingerTip`, and `MiddleFingerTip` on both sides,
with at least two valid frames per hand. Episodes without these annotations
are rejected before segmentation or inpainting starts.

### Folder pipeline

The pipeline accepts either one task directory or a dataset root. It
recursively discovers videos and only treats a file as an episode when a
same-directory, same-stem HDF5 file exists:

```text
basic_pick_place/0.mp4
basic_pick_place/0.hdf5
```

The directory containing the pair is the task and the shared profile is
generated from the ProPainter results of all discovered episodes in that
task. Unpaired MP4 files are ignored. Episodes are processed in natural
numeric order.

Run every task recursively under a dataset root:

```bash
.venv/bin/python -m human_to_robot.pipeline \
  --folder dataset/test/fry_bread \
  --output outputs/

.venv/bin/python -m human_to_robot.pipeline \
  --folder dataset/basic_pick_place \
  --output outputs/ \
  --isaacsim-root /home/ai01/isaacsim \
  --debug
```

```bash
.venv/bin/python -m human_to_robot.pipeline \
  --folder dataset/test/fry_bread \
  --output outputs/fry_bread_batch \
  --isaacsim-root /home/ai01/isaacsim \
  --debug
```

options:
`--debug`: to retain masks, inpainted video, trajectory, render
frames, logs, lighting profiles, and JSON reports.
`--no-reuse`: to force regeneration
`--no-hardware-encoding`: to force software encoding. Be default, use NVIDIA NVENC hardware acceleration, If NVENC is
unavailable at runtime, the pipeline automatically falls back to the source
codec (or software H.264).


ProPainter keeps the requested output canvas (1280×720 by default), but no
longer loads a whole long episode at that resolution in one invocation. The
pipeline groups non-empty arm-mask intervals into temporal chunks, adds
context, crops one fixed padded ROI per chunk, runs the unmodified ProPainter
inside that ROI, and pastes only the repaired mask region back into the full
canvas. `--propainter-subvideo-length` also acts as the 1280×720-equivalent
pixel-frame budget for each chunk. Context, ROI padding, boundary blending, and
the hard clip cap can be adjusted with `--propainter-context-frames`,
`--propainter-roi-margin`, `--propainter-transition-frames`, and
`--propainter-max-clip-frames`.


### Output layout

Without `--debug`, each task directory contains only the final composited MP4
and a byte-for-byte copy of the source HDF5 for every successfully processed
episode:

```text
outputs/
└── basic_pick_place/
    ├── 0.mp4
    ├── 0.hdf5
    ├── 1.mp4
    └── 1.hdf5
```

In this mode Isaac sends each rendered RGB, mask, and metric-depth frame
directly to the compositor in the same process. The final MP4 is encoded
incrementally and atomically published; the three numbered render-frame
directories are never created. `--debug` retains the PNG sequences and runs
the standalone compositor so every layer remains inspectable.

The processed MP4 is published beside a byte-for-byte copy of its source
HDF5, using the same basename. The source datasets are preserved and no robot
trajectory or hand-action fields are injected into the published HDF5.

With `--debug`, those deliverables remain at the same paths and the retained
workspace is added alongside them:

```text
outputs/
├── batch_report.json
└── basic_pick_place/
    ├── 0.mp4
    ├── 0.hdf5
    ├── lighting_profile.json
    ├── task_report.json
    └── 0/
        ├── pipeline.log
        ├── pipeline_state.json
        ├── episode_report.json
        ├── sam3/
        │   └── arm_masks/
        ├── propainter/
        │   └── 0/inpaint_out.mp4
        ├── g1/
        │   └── g1_trajectory.npz
        ├── g1_isaacsim/
        └── final/
```

The `final/` debug directory contains `synthetic_g1.mp4`, the
original/inpaint/G1 comparison video, and `composite_report.json`.

### Lighting profile and Isaac Sim

Lighting observations come from the ProPainter videos, after human-arm
removal. The original SAM arm region is still excluded as a low-reliability
area so generated inpainting texture cannot dominate the estimate. Episode
observations are robustly aggregated into one task-level
`lighting_profile.json`; the profile records confidence and warnings when
color or exposure varies substantially between episodes.

The action-alignment helper implements the paper's equations: it builds a
virtual finger from the index and middle fingertips, places the end effector
midway between it and the thumb, derives a right-handed grasp frame, smooths
position/width with Savitzky-Golay filtering, and smooths orientation on
SO(3). The production arm IK deliberately uses the stable HDF5 `leftHand` and
`rightHand` rigid-body tracks for position and rotation; only the paper's
fingertip-derived width drives hand articulation. This keeps noisy finger
detections from changing the arm trajectory.

Pinocchio solves the fixed-base G1 arm/body coordinates while freezing all 22
LinkerHand joints from arm IK. The paper's continuous smoothed width `w_t` is
scaled with the scene and mapped through a URDF-calibrated, monotonic
fingertip-aperture curve to continuous hand joint values. No per-episode binary
threshold is used. The full model has 51 actuated coordinates. Isaac's URDF
importer creates the arm-and-hand visuals, and the renderer writes precomputed
Pinocchio link poses to the USD stage. The project has no MuJoCo or Mink runtime
dependency. PhysX is intentionally not stepped, so its integration cannot
drift from the solved trajectory.
Isaac consumes the shared task profile for its fixed DomeLight and key light,
and the compositor uses the same profile for one fixed grade over the episode.
This is a bounded display match, not physical inverse rendering.

Current prototype limitations:

- LinkerHand articulation follows one calibrated scalar aperture per hand;
  independent finger shapes and contact forces are not reconstructed.
- Episodes without complete 3D wrist and fingertip annotations are unsupported.
- ProPainter removes the person well but can retain the original arm shadows.
- Optional held-object masks can lose the object under heavy hand occlusion;
  object segmentation is disabled unless `--object-prompt` is supplied.
- Only robot arms are rendered. This avoids the head/torso entering the
  egocentric camera while the waist bends.
