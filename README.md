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
6. Render only the two G1 shoulder subtrees with MuJoCo RGB, segmentation, and
   metric depth, then composite them into the inpainted video.

The local models expected by the pipeline are:

```text
models/sam3/sam3.pt
ProPainter/weights/raft-things.pth
ProPainter/weights/recurrent_flow_completion.pth
ProPainter/weights/ProPainter.pth
```

Run the complete 1280×720 experiment:

```bash
.venv/bin/python -m human_to_robot.pipeline \
  --video dataset/test/basic_pick_place/0.mp4 \
  --hdf5 dataset/test/basic_pick_place/0.hdf5 \
  --output outputs/basic_pick_place_0
```

Add `--reuse` to reuse finished SAM3, G1, and ProPainter stages.

### Isaac Sim 5.1 renderer with MuJoCo/Mink IK

The Isaac version keeps MuJoCo/Mink as the source of truth for IK and forward
kinematics. It imports the G1 arms into Isaac Sim, writes every MuJoCo body pose
directly to the USD stage, and uses Isaac Sim only for camera rendering. PhysX
is intentionally not stepped, so its joint integration cannot drift away from
the solved MuJoCo trajectory. The default lighting is one fixed warm DomeLight
(4300 K) plus a soft overhead key (4500 K); there is no per-frame white balance.

After generating (or reusing) `g1/g1_trajectory.npz`, render all 126 frames:

```bash
.venv/bin/python -m human_to_robot.render_isaacsim \
  --trajectory outputs/basic_pick_place_0/g1/g1_trajectory.npz \
  --mjcf assets/unitree_description/mjcf/g1.xml \
  --video dataset/test/basic_pick_place/0.mp4 \
  --hdf5 dataset/test/basic_pick_place/0.hdf5 \
  --output outputs/basic_pick_place_0/g1_isaacsim_warm
```

Composite the Isaac RGB, mask, and metric depth into the ProPainter result:

```bash
.venv/bin/python -m human_to_robot.composite \
  --source-video dataset/test/basic_pick_place/0.mp4 \
  --inpaint-video outputs/basic_pick_place_0/propainter/0/inpaint_out.mp4 \
  --hdf5 dataset/test/basic_pick_place/0.hdf5 \
  --g1-root outputs/basic_pick_place_0/g1_isaacsim_warm \
  --object-masks outputs/basic_pick_place_0/sam3/object_masks \
  --output outputs/basic_pick_place_0/final_isaacsim_warm \
  --robot-exposure -0.10 \
  --robot-contrast 0.92 \
  --robot-saturation 0.82 \
  --robot-soften-sigma 0.55
```


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
