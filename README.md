### EgoDex upper body to Unitree G1 (LeRobot v3)

`stream/scripts/egodex_to_robot.py` consumes same-stem EgoDex `.hdf5` and `.mp4`
pairs, fits the local neutral SMPL-X model, retargets that fitted trajectory to
the 29-DoF Unitree G1, maps the moving camera into the same robot-world frame,
and writes a frame-aligned LeRobot v3.0 dataset:

```bash
python stream/scripts/egodex_to_robot.py \
  --src-root dataset/basic_pick_place \
  --output-root output/
```

The script loads SMPL-X code from `../smplx` and always uses
`../models/models/smplx/SMPLX_NEUTRAL.npz` with 16 zero shape coefficients.
The source sample has no lower-body tracks, so both the SMPL-X lower body and
G1 legs stay neutral. `observation.state` is the current G1 kinematic
joint-position reference; `action` is the next-frame reference by default.
Fitted SMPL-X parameters, the original RGB video, mapped OpenCV
`T_world_camera`, camera intrinsics, and raw EgoDex confidence values are
retained for later Isaac replay and repainting. Convert camera rotation with
`R_usd = R_cv @ diag(1, -1, -1)` before assigning it to an Isaac/USD camera.
Use `--inspect-only` to inspect inputs without IK, or
`--validate-existing PATH` to validate an existing output.