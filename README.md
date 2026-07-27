# vision_align

## Human-to-G1 synthesis


The local models expected by the pipeline are:

```text
models/sam3/sam3.pt
ProPainter/weights/raft-things.pth
ProPainter/weights/recurrent_flow_completion.pth
ProPainter/weights/ProPainter.pth
```

replace
```bash
sam3/sam3/model_builder.py
```
with
```bash
utils/model_builder.py
```
to bypath hf model check each time



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
```

```bash
.venv/bin/python -m human_to_robot.pipeline \
  --folder dataset/test/fry_bread \
  --output outputs/fry_bread_batch \
  --backend isaacsim \
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
and a copy of the source HDF5 for every successfully processed episode:

```text
outputs/
└── basic_pick_place/
    ├── 0.mp4
    ├── 0.hdf5
    ├── 1.mp4
    └── 1.hdf5
```

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

Pinocchio solves the 29 fixed-base G1 joint coordinates and stores every link
pose from forward kinematics in the trajectory archive before Kit starts. The
renderer writes those NumPy poses to the USD stage. Isaac's MJCF importer is
used only to create the arm visual prims; the project has no MuJoCo or Mink
Python runtime dependency. PhysX is intentionally not stepped, so its
integration cannot drift from the solved trajectory.
Isaac consumes the shared task profile for its fixed DomeLight and key light,
and the compositor uses the same profile for one fixed grade over the episode.
This is a bounded display match, not physical inverse rendering.

Current prototype limitations:

- The fixed G1 hand mesh stays open; grasp state and finger motion are absent.
- Hand position uses the HDF5 palm/root point and therefore retains a fixed
  tool/contact offset. Wrist rotation uses the rigid hand frame, but finger
  articulation is still absent.
- ProPainter removes the person well but can retain the original arm shadows.
- Optional held-object masks can lose the object under heavy hand occlusion;
  object segmentation is disabled unless `--object-prompt` is supplied.
- Only robot arms are rendered. This avoids the head/torso entering the
  egocentric camera while the waist bends.
