# Camera-relative calibration assessment

This directory is isolated from the current pipeline. No existing calibration
or rendering code is changed by this assessment.

## Clarified objective

The human and robot cameras are independent observers. They do not need equal
positions, optical axes, or images. A 30--45 degree angle between them is not a
calibration error by itself. The required invariant is instead:

```text
one source scene point -> one stable robot-world point
```

The scene must keep its position and orientation relative to the robot and the
other scene geometry while the physical G1 camera views it from its own pose.

## Required transform

First reconstruct source-camera depth in one stable source frame:

```text
p_camera(t) = depth(t) * inverse(K_source) * pixel
X_source(t) = T_source_from_camera(t) * p_camera(t)
```

Then map that source scene into robot world with one constant similarity:

```text
X_robot = anchor_robot + scale * Q * (X_source - anchor_source)
```

`anchor_source` can be the source hip or a stable bilateral hand anchor;
`anchor_robot` is its matching robot-world anchor. `Q` maps source body/world
axes to robot-world axes. It does not align either camera to the other.

Finally, the BeingBeyond camera independently observes `X_robot` through its
own forward-kinematic pose and intrinsic calibration.

## Meaning of source camera rotation

The source world's arbitrary absolute rotation is not an independent
calibration parameter. If every source transform is rotated by the same gauge
rotation, that rotation cancels against `Q`.

The per-frame source camera rotation cannot be removed from backprojection,
because it maps each local depth image into the same stable source scene. What
must be avoided is forcing that rotation onto the robot camera. The desired
separation is:

```text
source camera pose -> reconstruct source scene
source-to-robot Q  -> place the scene
robot camera pose  -> observe the placed scene
```

The current mapped-camera formula already has this general structure:

```text
c_mapped = root_robot + scale * B * (c_source - hip_source)
R_mapped = B * R_source
```

Here `B` is the source-body to robot-world basis, not a robot-camera alignment.
Therefore the mapped source camera and physical robot camera being different is
expected and should not be minimized as an error metric.

## What similar triangles can solve

Camera/anchor triangles are useful for estimating scale and relative
translation. They do not determine a full 3-D orientation from distances alone.
One camera-to-midpoint vector leaves rotation around that vector unobservable;
orientation still needs a stable body basis, gravity basis, or at least two
non-collinear matched directions.

The robot camera intrinsic does not need to match the source intrinsic under
this objective. Each intrinsic is used only for its own backprojection or
projection. Pixel-for-pixel source/robot reprojection equality is not a quality
criterion.

## Scene leveling

The current scene leveling is not camera calibration. It applies the non-rigid
world-space shear

```text
z' = z + a*x + b*y
```

while leaving camera quaternions unchanged. Because this changes relative
geometry and surface angles, it conflicts with the clarified priority when it
is used as an extra placement degree of freedom.

The isolated world-stable branch should therefore not apply scene leveling.
Keep the detected support-plane tilt as a read-only diagnostic. If a consistent
tilt remains, fix the upstream source pose or the constant rigid basis `Q`;
do not hide it with a per-scene non-rigid shear.

## Recommended isolated experiment

Keep the current baseline untouched and compare:

1. current mapped source scene with leveling,
2. the same mapping with leveling disabled,
3. one constant rigid `Q`, scale, and anchor translation fitted from stable
   body/scene correspondences.

Measure scene-to-robot anchor residual, preservation of pairwise scene
distances and directions, cross-frame static-scene drift, and unmodified
support-plane tilt. Do not score source-camera versus robot-camera angle.
