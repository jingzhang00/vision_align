# vision_align

## Human-to-G1 synthesis


The local models expected by the pipeline are:

```text
models/sam3/sam3.pt
models/DA3METRIC-LARGE/config.json
models/DA3METRIC-LARGE/model.safetensors
ProPainter/weights/raft-things.pth
ProPainter/weights/recurrent_flow_completion.pth
ProPainter/weights/ProPainter.pth
mano_v1_2/
```

Every standard episode HDF5 must provide 3D transforms and confidences for
`Hand`, `ThumbTip`, `IndexFingerTip`, and `MiddleFingerTip` on both sides,
with at least two valid frames per hand. Episodes without these annotations
are rejected before segmentation or inpainting starts. A dataset adapter may
instead declare `human_to_robot_hand_target_mode=rigid_palm` and provide
explicit `leftHand`/`rightHand` rigid poses; this maps those poses directly to
the same G1 palm IK targets and keeps both robot hands open. The default hand
controller evaluates the four annotated landmarks independently on every
frame and maps the resulting exact binary action to the six independent
LinkerHand O6 motors. `--no-binary-hands` instead uses the complete
25-landmark hand skeleton when available and retargets thumb yaw/flex plus
independent index, middle, ring, and little-finger flexion to the six O6
motors. Four-landmark inputs fall back to the continuous `(p, R, w)` action.

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
  --folder dataset/basic_pick_place \
  --output outputs/ \
  --no-binary-hands

.venv/bin/python -m human_to_robot.pipeline \
  --folder dataset/test/flip_coin \
  --output outputs/ \
  --isaacsim-root /home/ai01/isaacsim \
  --no-binary-hands \
  --debug
```

```bash
.venv/bin/python -m human_to_robot.pipeline \
  --folder dataset/test/fry_bread \
  --output outputs/fry_bread_batch \
  --isaacsim-root /home/ai01/isaacsim \
  --no-binary-hands \
  --debug \
  --visualize
```

options:
`--debug`: to retain masks, inpainted video, trajectory, render
frames, logs, lighting profiles, and JSON reports.
`--visualize`: to open the Isaac Sim window during rendering; this forces the
render stage to run instead of reusing cached output.
`--no-reuse`: to force regeneration.
`--no-binary-hands`: to retarget the complete articulated hand skeleton to
the six independent O6 motors instead of thresholding into the default binary
controller. Incomplete hand skeletons fall back to continuous `(p, R, w)`.
`--no-hardware-encoding`: to force software encoding. By default, NVIDIA
NVENC hardware acceleration is used when available, with automatic fallback
to the source codec or software H.264.
`visualize`: isaac render

Every processing stage preserves the source video's native resolution;
width and height are not command-line parameters. ProPainter no longer loads
a whole long episode at full resolution in one invocation. The
SAM arm-mask sequence first repairs one- or two-frame area collapses only when
the masks on both sides agree in area and location. This prevents a prompt
boundary dropout from leaking the original hand into ProPainter.
The pipeline groups non-empty arm-mask intervals into temporal chunks, adds
context, crops one fixed padded ROI per chunk, and pastes only the repaired
mask region back into the full canvas. The planner packs the largest
consecutive core range that fits each chunk budget. One persistent ProPainter
worker loads RAFT, flow completion, and inpainting weights once per episode
and returns only the PNG frames required by the compositor.

First-person arms that remain attached to an image edge can be reconstructed
as hand-shaped ProPainter blur even with an accurate SAM mask. The pipeline
therefore samples pixels outside the SAM masks, aligns those background
samples to each output frame with feature-matched homographies, color-matches
the aligned candidates, and takes a robust temporal median to remove residual
hands and connected neutral arm shadows. Failed or implausible alignments are
discarded; if no aligned candidate remains, the cleanup leaves that frame's
ProPainter result unchanged. A stationary-scene temporal reference is retained
as the fallback for feature-poor clips after its unmasked error passes the
reliability threshold.

`--propainter-subvideo-length` controls ProPainter's internal
memory-efficient inference window. The independent
`--propainter-max-chunk-pixel-frames` limit is the cropped ROI pixel count
multiplied by clip frames, so native source and ROI resolution are accounted
directly without a reference resolution. Its default is 60,000,000.
Temporal context defaults to 10 frames on each side. Context, ROI padding,
boundary blending, and the hard clip cap can be adjusted with
`--propainter-context-frames`, `--propainter-roi-margin`,
`--propainter-transition-frames`, and `--propainter-max-clip-frames`.


### Output layout

Without `--debug`, each task directory contains only the final composited MP4
and a byte-for-byte copy of the source HDF5 for every successfully processed
episode. Failed episodes retain their log and error report under
`.failures/<episode>/`:

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
        ├── depth_anything3/
        │   ├── scene_depth_mm/
        │   └── scene_depth_report.json
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

The action-alignment helper builds a
virtual finger from the index and middle fingertips, places the end effector
midway between it and the thumb, derives a right-handed grasp frame, smooths
position/width with Savitzky-Golay filtering, and smooths orientation on
SO(3). With full articulated tracks, production IK targets this grasp frame
directly. The robot TCP origin follows the current O6
thumb/virtual-finger midpoint, while its semantic gripper basis is fixed from
the open hand. Fixing that basis prevents curled fingertips from introducing
a false 80–170 degree orientation flip. The stable HDF5 `leftHand` and
`rightHand` tracks still supply visibility and tracking confidence.

The robot base follows Eq. (3) of Qwen-RobotManip: representative keyframes
are the per-hand spatial extrema, candidate upright SE(3) base poses are
grid-searched around the full-trajectory centroid within the G1 morphology
reach, and each candidate is ranked by independent full-6D IK feasibility.
Pinocchio then solves the selected fixed-base G1 arm/body coordinates while
freezing all 22 LinkerHand joints from arm IK. Binary grasp inference uses the
per-frame virtual-finger aperture divided by the wrist-to-virtual-finger
length. A visible operation hand is closed exactly when this ratio crosses
the configured threshold; the 0/1 result drives the O6 open/power-fist preset
directly. There is no event detector, temporal latch, interpolation,
hysteresis, or actuator blend in this binary path. `--no-binary-hands`
preserves the existing smoothed grasp TCP target `(p_t, R_t)` while fitting
the complete palm-local hand skeleton to six independent O6 motors. The four
long fingers each receive their own flexion command; the thumb receives yaw
and flexion commands. URDF mimic joints are expanded from those six motors.
When a complete skeleton is unavailable, the smoothed width `w_t` maps through
the calibrated O6 aperture curve as a compatibility fallback. The
state-dependent O6 TCP converts the resulting hand pose and `(p_t, R_t)` into
a palm IK target. The full model has 51 actuated coordinates. Isaac's URDF
importer creates the arm-and-hand visuals, and the renderer writes precomputed
Pinocchio link poses to the USD stage. The project
has no MuJoCo or Mink runtime dependency. PhysX is intentionally not stepped,
so its integration cannot drift from the solved trajectory.
Isaac consumes the shared task profile for its fixed DomeLight and key light,
and the compositor uses the same profile for one fixed grade over the episode.
This is a bounded display match, not physical inverse rendering.

Before Isaac rendering, the local `DA3METRIC-LARGE` checkpoint estimates
native metric depth from the ProPainter clean-background video. The official
`focal * network_depth / 300` conversion is brought into the same spatial
scale as the retargeted trajectory and cached as source-resolution uint16
millimeter PNGs. Compositing uses the hard per-pixel rule
`robot_mask & valid(robot_depth) & valid(scene_depth) &
(robot_depth <= scene_depth)`. Occluded or invalid pixels remain the
ProPainter background; there is no depth margin, alpha blur, or temporal
depth smoothing.

Current prototype limitations:

- Binary LinkerHand articulation selects one six-motor power-fist preset per
  hand; independent finger intent and contact forces are not reconstructed.
- The URDF full-close preset is valid for simulation. Real hardware still
  needs device-specific angle, current, and contact calibration.
- Episodes without complete 3D wrist and fingertip annotations are unsupported.
- Only robot arms are rendered. This avoids the head/torso entering the
  egocentric camera while the waist bends.
