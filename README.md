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
  --src-root dataset/insert_remove_usb \
  --output-root output_new/insert_remove_usb
```

```bash
uv run python repaint/traj_replay.py \
  --dataset output_new/pour
```

For the full-URDF replay in Isaac Sim, run it directly in the activated
`vision-align` environment:

```bash
uv run python repaint/isaac_replay.py \
  --dataset output_new/fry_bread
```

```bash
.venv/bin/python repaint/egodex_repaint.py \
  --input_dir output_new/basic_pick_place \
  --output_dir output_new/basic_pick_place/repaint
```


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

### Metric Video Depth Anything RGB-D replay

Prepare the arm-removed `basic_pick_place` episode with one temporally
contiguous Metric Video Depth Anything inference:

```bash
.venv/bin/python -m repaint.rgbd_prepare \
  --dataset output_new/basic_pick_place \
  --source-root dataset/basic_pick_place \
  --episode 0 \
  --scene-scale 0.8 \
  --arm-alignment trajectory-only \
  --camera-alignment source-extrinsic-only \
  --rgb-source processed \
  --mesh-width 480 \
  --vda-checkpoint \
    Video-Depth-Anything/checkpoints/metric_video_depth_anything_vitl.pth \
  --output \
    output_new/basic_pick_place/repaint/vda_rgbd_auto/episode_000000
```

The selected episode is decoded as one contiguous video and passed to VDA.
The default `--vda-scale-mode auto` estimates a per-episode raw-VDA-to-source
metric scale from source-HDF5 hand depth on the original video, then transfers
it to the handless prediction through unchanged background. It fails rather
than silently using a cross-task constant when its support/temporal quality
gates do not pass. The resolved scale is applied once before `scene_scale`;
the latter remains `0.8`, so camera motion and robot-world mapping do not
change. Depth is nearest-neighbor resized and stored as `uint16` millimeters.
For a controlled legacy comparison, use
`--vda-scale-mode fixed --vda-metric-scale 0.525`.

Run the complete G1, dynamic RGB-D scene, free scene viewport, and fixed
first-person robot-camera viewport continuously:

```bash
.venv/bin/python -m repaint.rgbd_isaac \
  --prepared \
    output_new/basic_pick_place/repaint/vda_rgbd_auto/episode_000000 \
  --interactive \
  --loop \
  --playback-fps 30 \
  --no-headless \
  --background-mode rgbd-mesh \
  --camera-pose-source source-extrinsic \
  --capture-view scene \
  --robot-camera-window \
  --scene-eye -0.8 -0.9 1.7 \
  --scene-target 0.2 0.05 1.05
```

Use normal Isaac viewport controls to orbit, pan, and zoom. In the terminal,
`s` pauses/resumes, `n` and `p` step, `g N` jumps to source frame `N`, and `q`
quits. `--scene-eye` and `--scene-target` only choose the initial free view.

For an offline scene-view MP4, omit the interactive flags and provide output:

```bash
.venv/bin/python -m repaint.rgbd_isaac \
  --prepared \
    output_new/basic_pick_place/repaint/vda_rgbd_fixed/episode_000000 \
  --output \
    output_new/basic_pick_place/repaint/vda_rgbd_fixed/episode_000000/isaac/g1_scene_fixed_scale.mp4 \
  --headless \
  --background-mode rgbd-mesh \
  --camera-pose-source source-extrinsic \
  --capture-view scene \
  --scene-eye -0.8 -0.9 1.7 \
  --scene-target 0.2 0.05 1.05
```

For the corresponding fixed robot-camera MP4:

```bash
.venv/bin/python -m repaint.rgbd_isaac \
  --prepared \
    output_new/basic_pick_place/repaint/vda_rgbd_fixed/episode_000000 \
  --output \
    output_new/basic_pick_place/repaint/vda_rgbd_fixed/episode_000000/isaac/g1_robot_camera_fixed_scale.mp4 \
  --headless \
  --background-mode rgbd-mesh \
  --camera-pose-source source-extrinsic \
  --capture-view robot-camera
```

The default robot-camera calibration is the nominal BeingBeyond
`head_cam` configuration:

```text
parent: torso_link
position XYZ (m): 0.0576235, 0.01753, 0.42987
rotation WXYZ:    0.91495966, 0, 0.4035453, 0
image size:       640 x 480
K:                [[607.212158203125, 0, 311.9999084472656],
                   [0, 607.0084838867188, 261.152099609375],
                   [0, 0, 1]]
distortion:       ideal pinhole (all zeros)
```

This nominal pose is the fixed `d435_joint` pose in the local G1 URDF. The
BeingBeyond reset event is intentionally not reproduced by deterministic
replay; its configured noise ranges are currently zero, while its occlusion
guard can still nudge the camera forward by 5 mm.

The source-video intrinsic remains separate and is used only to back-project
the reconstructed scene. The robot render camera always uses the fixed
BeingBeyond K and zero distortion; there is no live-device or CLI calibration
override. Its horizontal FoV is 55.6 degrees, versus 105 degrees for the
source video. FK projection places the right-hand base inside the 640×480
image in only 13 of 63 frames; at frame 0 its horizontal coordinate is about
792, beyond the image width of 640. Correcting reconstructed physical
dimensions does not change pinhole-image angles. Changing depth scale or
`--render-scale` therefore cannot widen this fixed FoV or recover the missing
right hand.
