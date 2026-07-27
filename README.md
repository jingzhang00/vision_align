# vision_align

## Human-to-G1 synthesis

`human_to_robot/` implements a first humanoid adaptation of §2.3 in the
Qwen-RobotManip technical report:

1. Read the aligned HDF5 hand and camera transforms.
2. Treat each `leftHand` / `rightHand` origin as one position point and use its
   rigid-transform rotation for wrist orientation. No MANO, fingertip
   retargeting, gripper width, or open/close action is used.
3. Smooth the hand points in the first-camera reference frame while preserving
   the measured per-frame head-camera motion for rendering. Confidence-valid
   HDF5 points remain active outside the image, avoiding an IK jump at the
   image boundary.
4. Search G1 pelvis placements by position IK feasibility. Solve full-pose
   `left_palm` / `right_palm` Mink tasks after independently calibrating each
   human hand frame to the G1 palm in frame 0. The floating base and legs are
   frozen; waist, shoulder/elbow, and wrist joints remain available.
5. Segment human arms with local SAM3 and remove them with ProPainter.
6. Estimate one fixed lighting profile for each task from its ProPainter
   videos. Every episode in the task shares that profile; there is no
   per-frame white balance.
7. Render only the two G1 shoulder subtrees with either MuJoCo or Isaac Sim,
   then composite their RGB, segmentation, and metric depth into the inpainted
   video.

The local models expected by the pipeline are:

```text
models/sam3/sam3.pt
ProPainter/weights/raft-things.pth
ProPainter/weights/recurrent_flow_completion.pth
ProPainter/weights/ProPainter.pth
```

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

Run the `fry_bread` task with Isaac Sim rendering:

```bash
.venv/bin/python -m human_to_robot.pipeline \
  --folder dataset/test/fry_bread \
  --output outputs/fry_bread_batch \
  --backend isaacsim
```

Here `--output` is the task output root, so episode `0` is written beneath
`outputs/fry_bread_batch/0/`.

Run every task recursively under a dataset root with MuJoCo rendering:

```bash
.venv/bin/python -m human_to_robot.pipeline \
  --folder dataset/test \
  --output outputs/test_batch \
  --backend mujoco
```

In the dataset-root form, the relative task directories are mirrored below
`--output`, for example
`outputs/test_batch/basic_pick_place/0/`. `--backend` selects `mujoco` or
`isaacsim`; MuJoCo/Mink remains the IK and forward-kinematics source in both
cases. Complete intermediates are reused by default. Reuse is guarded by an
episode state signature covering source files, model/code timestamps, relevant
CLI parameters, the selected backend, and the task lighting profile. An
interrupted phase is marked incomplete and cannot be mistaken for a reusable
result. Pass `--no-reuse` to force regeneration.

The terminal shows two episode-level progress bars: preparation, followed by
rendering/compositing. They report the current episode, elapsed time, ETA, and
the completed episode time. Verbose SAM3, ProPainter, IK, and renderer output
is written to each episode's `pipeline.log` instead of producing separate
stage progress bars. At completion, the pipeline reports total batch elapsed
time and writes machine-readable timing fields to the episode, task, and
batch reports.

### Output layout

For a dataset-root run, each task has one `lighting_profile.json`, while each
episode keeps its own masks, inpainted video, trajectory, render, log, and
timing report:

```text
outputs/test_batch/
├── batch_report.json
└── basic_pick_place/
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
        ├── g1_isaacsim/          # present when Isaac Sim is selected
        ├── final_mujoco/         # MuJoCo backend result
        └── final_isaacsim/       # Isaac Sim backend result
```

Only the selected backend's `final_<backend>/` directory is produced. Its main
result is `synthetic_g1.mp4`; it also contains the original/inpaint/G1
comparison video and `composite_report.json`.

### Lighting profile and Isaac Sim

Lighting observations come from the ProPainter videos, after human-arm
removal. The original SAM arm region is still excluded as a low-reliability
area so generated inpainting texture cannot dominate the estimate. Episode
observations are robustly aggregated into one task-level
`lighting_profile.json`; the profile records confidence and warnings when
color or exposure varies substantially between episodes.

The Isaac backend imports the G1 arms into Isaac Sim, writes every MuJoCo body
pose directly to the USD stage, and uses Isaac Sim only for camera rendering.
PhysX is intentionally not stepped, so its joint integration cannot drift
away from the solved MuJoCo trajectory. Isaac consumes the shared task profile
for its fixed DomeLight and key light, and the compositor uses the same
profile for one fixed grade over the episode. This is a bounded display match,
not physical inverse rendering.

Current prototype limitations:

- The fixed G1 hand mesh stays open; grasp state and finger motion are absent.
- Hand position uses the HDF5 palm/root point and therefore retains a fixed
  tool/contact offset. Wrist rotation uses the rigid hand frame, but finger
  articulation is still absent.
- ProPainter removes the person well but can retain the original arm shadows.
- The optional SAM3 stapler mask loses the object under heavy hand occlusion;
  the source object is nevertheless mostly retained because the arm mask does
  not include its black pixels.
- Only robot arms are rendered. This avoids the head/torso entering the
  egocentric camera while the waist bends.

## G1 viewer

```bash
.venv/bin/python -m mujoco.viewer \
  --mjcf=assets/unitree_description/mjcf/g1.xml

uv run examples/arm_ur5e.py
```
