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

Every standard episode HDF5 must provide 3D transforms for `Hand`, `ThumbTip`,
`IndexFingerTip`, and `MiddleFingerTip` on both sides, with at least two valid
frames per hand. Confidence tracks are used when present and otherwise
default to one. Episodes without the transforms are rejected before
segmentation or inpainting starts. A dataset adapter may
instead declare `human_to_robot_hand_target_mode=rigid_palm` and provide
explicit `leftHand`/`rightHand` rigid poses; this maps those poses directly to
the same G1 palm IK targets and keeps both robot hands open. When available,
the complete 25-landmark hand skeleton is used to retarget thumb yaw/flex plus
independent index, middle, ring, and little-finger flexion to the six
LinkerHand O6 motors. Four-landmark inputs fall back to the continuous
`(p, R, w)` action.

Spatial retargeting also requires the common upper-body transforms `hip`,
`leftArm`, `leftForearm`, `leftHand`, `rightArm`, `rightForearm`, and
`rightHand`. `neck4`, `neck3`, or `spine7` is used to identify the source
world-up axis; when none is present, the tracked camera-to-hip direction is
used. The source is not silently assigned a fixed human scale or camera
height.

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
  --propainter-mode chunked \
  --propainter-subvideo-length 20 \
  --debug


.venv/bin/python -m human_to_robot.pipeline \
  --folder dataset/test/flip_coin \
  --output outputs/ \
  --isaacsim-root /home/ai01/isaacsim \
  --debug

.venv/bin/python -m human_to_robot.pipeline \
  --folder dataset/pour \
  --output outputs/ \
  --propainter-mode chunked \
  --propainter-subvideo-length 20 \
  --debug
```

```bash
.venv/bin/python -m human_to_robot.pipeline \
  --folder dataset/test/fry_bread \
  --output outputs/ \
  --propainter-mode chunked \
  --propainter-subvideo-length 20 \
  --debug

```

options: 
`--debug`: to retain masks, inpainted video, trajectory, render
frames, logs, lighting profiles, and JSON reports.
`--visualize`: to open the Isaac Sim window during rendering; this forces the
render stage to run instead of reusing cached output.
`--no-reuse`: to force regeneration.
`--no-hardware-encoding`: to force software encoding. By default, NVIDIA
NVENC hardware acceleration is used when available, with automatic fallback
to the source codec or software H.264.
`--propainter-mode official`: to use the upstream whole-video, full-frame
ProPainter entry point. This is the default.
`--propainter-mode chunked`: to restore the memory-bounded temporal ROI
wrapper when the official path runs out of GPU memory.
`visualize`: isaac render

Every processing stage preserves the source video's native resolution;
width and height are not command-line parameters. The default
`--propainter-mode official` invokes `ProPainter/inference_propainter.py`
directly on the complete full-frame episode and SAM mask sequence. It uses
upstream ProPainter's internal RAFT batching, flow-completion/image-propagation
`--propainter-subvideo-length` windows, local neighbors, and global reference
frames without external ROI crops or independently inferred outer chunks.
FP16 is enabled by default and can be disabled with `--no-propainter-fp16`.

The optional `--propainter-mode chunked` mode groups non-empty arm-mask
intervals into temporal chunks, adds context, crops one fixed padded ROI per
chunk, and pastes only the repaired mask region back into the full canvas.
The planner packs the largest consecutive core range that fits each chunk
budget. One persistent ProPainter worker loads RAFT, flow completion, and
inpainting weights once per episode and returns only the PNG frames required
by the compositor.

The production pipeline does not build a background plate by averaging or
taking a pixelwise median across different video frames. Each output frame is
the ProPainter result for that timestamp, pasted only through the repaired SAM
mask. In chunked mode, overlapping outer chunk boundaries may cross-fade two
independently inferred versions of the same timestamp; they never aggregate
pixels from different timestamps. There is no temporal-background cleanup
stage or corresponding command-line option.

`--propainter-subvideo-length` controls upstream ProPainter's internal
memory-efficient inference window in both modes. In chunked mode only, the
independent `--propainter-max-chunk-pixel-frames` limit is the cropped ROI
pixel count multiplied by clip frames; its default is 60,000,000. Temporal
context defaults to 10 frames on each side. Context, ROI padding, boundary
blending, and the hard clip cap can be adjusted with
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
original/inpaint/G1 comparison video, and `composite_report.json`..
