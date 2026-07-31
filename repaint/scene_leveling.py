"""Gravity-prior support-plane leveling for reconstructed RGB-D scenes.

The source camera pose must retain the human head motion: it is the transform
that takes each camera-space depth map back into a common world frame.  Metric
monocular depth can nevertheless leave a small, temporally stable slope on a
surface that should be horizontal.  This module estimates that residual slope
from the reconstructed geometry and returns one constant world-space affine
height correction.  The correction changes only world Z, preserving world X/Y
and the source camera orientation (including genuine human head pitch).

An affine height shear cannot be represented by a rigid camera quaternion.
Consequently the estimator deliberately leaves ``camera_pose_cv_xyzw``
unchanged and exposes ``linear_world`` plus ``pivot_world_xyz_m`` for applying
the correction directly to reconstructed world points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from repaint.rgbd_geometry import (
    backproject_z_depth,
    camera_points_to_world,
)


SUPPORT_PLANE_LEVELING_METHOD = (
    "gravity-prior dominant support plane, one constant Z-only affine correction"
)
SUPPORT_PLANE_LEVELING_STAGE = "support-plane-z-only-affine-leveling"


@dataclass(frozen=True)
class SceneLeveling:
    """One constant support-plane affine correction.

    ``camera_pose_cv_xyzw`` is the unmodified input trajectory.  Applying
    ``linear_world`` to a camera rotation would generally produce a non-rigid
    basis, so it must not be encoded as a pose quaternion.
    """

    camera_pose_cv_xyzw: np.ndarray
    transformed_camera_origins_world_xyz_m: np.ndarray
    linear_world: np.ndarray
    pivot_world_xyz_m: np.ndarray
    report: dict[str, object]


@dataclass(frozen=True)
class _PlaneFit:
    normal_world: np.ndarray
    centroid_world_xyz_m: np.ndarray
    support_fraction: float
    inlier_count: int
    point_count: int
    residual_p50_m: float
    residual_p95_m: float


def apply_world_affine_to_points(
    points_world_xyz_m: np.ndarray,
    *,
    linear_world: np.ndarray,
    pivot_world_xyz_m: np.ndarray,
    maximum_condition_number: float = 10.0,
) -> np.ndarray:
    """Apply a pivoted world-space affine transform to one or more 3D points.

    This helper intentionally operates on points, not camera poses.  A
    non-rigid affine transform cannot be faithfully stored in a position plus
    quaternion pose.
    """

    points = np.asarray(points_world_xyz_m, dtype=np.float64)
    linear = np.asarray(linear_world, dtype=np.float64)
    pivot = np.asarray(pivot_world_xyz_m, dtype=np.float64)
    if (
        points.ndim < 1
        or points.shape[-1] != 3
        or not np.isfinite(points).all()
    ):
        raise ValueError("points_world_xyz_m must be finite [...,3]")
    if linear.shape != (3, 3) or not np.isfinite(linear).all():
        raise ValueError("linear_world must be a finite 3x3 matrix")
    if pivot.shape != (3,) or not np.isfinite(pivot).all():
        raise ValueError("pivot_world_xyz_m must be a finite 3-vector")
    if (
        not np.isfinite(maximum_condition_number)
        or maximum_condition_number < 1.0
    ):
        raise ValueError("maximum_condition_number must be finite and >= 1")
    determinant = float(np.linalg.det(linear))
    condition_number = float(np.linalg.cond(linear))
    if determinant <= 1e-8:
        raise ValueError(
            "linear_world must preserve orientation with positive determinant"
        )
    if (
        not np.isfinite(condition_number)
        or condition_number > maximum_condition_number
    ):
        raise ValueError(
            "linear_world is ill-conditioned: "
            f"{condition_number:.6g} > {maximum_condition_number:.6g}"
        )
    return (
        (points - pivot) @ linear.T
        + pivot
    )


def _normal_to_z_only_affine(normal_world: np.ndarray) -> np.ndarray:
    """Return a unit-determinant affine map that changes only world Z.

    For a plane ``n dot (x - c) = 0``, the returned matrix uses

    ``z' = z + (n_x / n_z) dx + (n_y / n_z) dy``.

    Therefore all points on the plane receive one constant output Z while
    their world X/Y coordinates remain exactly unchanged.
    """

    normal = np.asarray(normal_world, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("normal_world must be a finite 3-vector")
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= 1e-8:
        raise ValueError("normal_world must be non-zero")
    normal = normal / magnitude
    if normal[2] < 0.0:
        normal = -normal
    if normal[2] <= 1e-8:
        raise ValueError("support normal must have a positive world-Z component")
    linear = np.eye(3, dtype=np.float64)
    linear[2, 0] = normal[0] / normal[2]
    linear[2, 1] = normal[1] / normal[2]
    return linear


def _transform_plane_normal(
    normal_world: np.ndarray,
    *,
    linear_world: np.ndarray,
) -> np.ndarray:
    """Transform a plane normal by the inverse transpose of an affine map."""

    transformed = np.linalg.solve(
        np.asarray(linear_world, dtype=np.float64).T,
        np.asarray(normal_world, dtype=np.float64),
    )
    transformed /= np.linalg.norm(transformed)
    if transformed[2] < 0.0:
        transformed = -transformed
    return transformed


def _sample_world_points(
    *,
    depth_path: Path,
    mesh_size: tuple[int, int],
    mesh_intrinsic: np.ndarray,
    camera_pose_cv_xyzw: np.ndarray,
    maximum_points: int,
) -> np.ndarray:
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise RuntimeError(f"Failed to read depth image: {depth_path}")
    width, height = (int(value) for value in mesh_size)
    if depth_mm.shape != (height, width):
        raise ValueError(
            f"Depth image shape {depth_mm.shape} does not match "
            f"mesh_size {(width, height)}"
        )
    depth_m = np.asarray(depth_mm, dtype=np.float64) / 1000.0
    points_camera = backproject_z_depth(depth_m, mesh_intrinsic)
    points_world = camera_points_to_world(
        points_camera,
        camera_pose_cv_xyzw,
    )
    pixel_count = width * height
    grid_stride = max(
        1,
        int(np.ceil(np.sqrt(pixel_count / maximum_points))),
    )
    sampled = points_world[::grid_stride, ::grid_stride].reshape(-1, 3)
    return sampled[np.isfinite(sampled).all(axis=1)]


def _fit_gravity_prior_plane(
    points_world: np.ndarray,
    *,
    residual_threshold_m: float,
    maximum_tilt_degrees: float,
    minimum_support_fraction: float,
    trials: int,
    random_seed: int,
) -> _PlaneFit:
    """Fit the dominant approximately horizontal plane with deterministic RANSAC."""

    points = np.asarray(points_world, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 100
        or not np.isfinite(points).all()
    ):
        raise ValueError("support-plane fitting needs at least 100 finite points")
    cosine_limit = float(np.cos(np.deg2rad(maximum_tilt_degrees)))
    generator = np.random.default_rng(random_seed)
    best_inlier: np.ndarray | None = None
    best_count = 0
    best_residual = float("inf")
    for _ in range(trials):
        sample = points[generator.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 1e-10:
            continue
        normal /= magnitude
        if normal[2] < 0.0:
            normal = -normal
        if normal[2] < cosine_limit:
            continue
        residual = np.abs((points - sample[0]) @ normal)
        inlier = residual <= residual_threshold_m
        count = int(np.sum(inlier))
        median_residual = (
            float(np.median(residual[inlier]))
            if count
            else float("inf")
        )
        if count > best_count or (
            count == best_count and median_residual < best_residual
        ):
            best_count = count
            best_residual = median_residual
            best_inlier = inlier
    if best_inlier is None:
        raise ValueError("no approximately horizontal RANSAC plane was found")

    inlier = best_inlier
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    centroid = np.zeros(3, dtype=np.float64)
    for _ in range(3):
        inlier_points = points[inlier]
        if len(inlier_points) < 3:
            raise ValueError("support-plane refinement has too few inliers")
        centroid = np.mean(inlier_points, axis=0)
        _, _, right_vectors = np.linalg.svd(
            inlier_points - centroid,
            full_matrices=False,
        )
        normal = right_vectors[-1]
        if normal[2] < 0.0:
            normal = -normal
        normal /= np.linalg.norm(normal)
        if normal[2] < cosine_limit:
            raise ValueError(
                "refined support plane violates the gravity prior"
            )
        residual = np.abs((points - centroid) @ normal)
        inlier = residual <= residual_threshold_m

    inlier_points = points[inlier]
    centroid = np.mean(inlier_points, axis=0)
    residual = np.abs((inlier_points - centroid) @ normal)
    support_fraction = float(len(inlier_points) / len(points))
    if support_fraction < minimum_support_fraction:
        raise ValueError(
            "dominant support plane is too small: "
            f"{support_fraction:.1%} < {minimum_support_fraction:.1%}"
        )
    return _PlaneFit(
        normal_world=normal,
        centroid_world_xyz_m=centroid,
        support_fraction=support_fraction,
        inlier_count=int(len(inlier_points)),
        point_count=int(len(points)),
        residual_p50_m=float(np.percentile(residual, 50.0)),
        residual_p95_m=float(np.percentile(residual, 95.0)),
    )


def level_scene_camera_poses_from_support_plane(
    *,
    depth_paths: tuple[Path, ...],
    mesh_size: tuple[int, int],
    mesh_intrinsic: np.ndarray,
    camera_pose_cv_xyzw: np.ndarray,
    window_frames: int,
    residual_threshold_m: float = 0.01,
    maximum_tilt_degrees: float = 30.0,
    minimum_support_fraction: float = 0.12,
    maximum_points_per_frame: int = 20_000,
    ransac_trials: int = 256,
    maximum_temporal_p95_degrees: float = 5.0,
    maximum_affine_condition_number: float = 10.0,
    pivot_world_xyz_m: np.ndarray | None = None,
    pivot_source: str | None = None,
) -> SceneLeveling:
    """Level a scene from a stable initial support plane.

    One plane is fitted independently in each initial frame.  A robust
    spherical consensus normal is then mapped to +Z by changing world Z only.
    The returned input camera poses remain unchanged because this affine
    correction cannot be encoded in rigid pose quaternions.  Callers must apply
    ``linear_world`` around ``pivot_world_xyz_m`` to reconstructed world points.
    """

    poses = np.asarray(camera_pose_cv_xyzw, dtype=np.float64)
    intrinsic = np.asarray(mesh_intrinsic, dtype=np.float64)
    if (
        poses.ndim != 2
        or poses.shape[1] != 7
        or len(poses) != len(depth_paths)
        or not np.isfinite(poses).all()
    ):
        raise ValueError("depth paths and finite camera poses must be frame aligned")
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
    ):
        raise ValueError("mesh_intrinsic must be a valid pinhole matrix")
    if not 3 <= window_frames <= len(poses):
        raise ValueError("window_frames must be between 3 and the frame count")
    if (
        not np.isfinite(residual_threshold_m)
        or residual_threshold_m <= 0.0
        or not np.isfinite(maximum_tilt_degrees)
        or not 0.0 < maximum_tilt_degrees < 90.0
        or not np.isfinite(minimum_support_fraction)
        or not 0.0 < minimum_support_fraction < 1.0
        or maximum_points_per_frame < 100
        or ransac_trials <= 0
        or not np.isfinite(maximum_temporal_p95_degrees)
        or maximum_temporal_p95_degrees <= 0.0
        or not np.isfinite(maximum_affine_condition_number)
        or maximum_affine_condition_number < 1.0
    ):
        raise ValueError("support-plane leveling thresholds are invalid")

    fits: list[_PlaneFit] = []
    failed_frames: list[dict[str, object]] = []
    for frame_id in range(window_frames):
        try:
            points = _sample_world_points(
                depth_path=depth_paths[frame_id],
                mesh_size=mesh_size,
                mesh_intrinsic=intrinsic,
                camera_pose_cv_xyzw=poses[frame_id],
                maximum_points=maximum_points_per_frame,
            )
            fits.append(
                _fit_gravity_prior_plane(
                    points,
                    residual_threshold_m=residual_threshold_m,
                    maximum_tilt_degrees=maximum_tilt_degrees,
                    minimum_support_fraction=minimum_support_fraction,
                    trials=ransac_trials,
                    random_seed=17_171 + frame_id,
                )
            )
        except (RuntimeError, ValueError) as exc:
            failed_frames.append(
                {
                    "local_frame": frame_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    minimum_fit_count = max(3, int(np.ceil(0.8 * window_frames)))
    if len(fits) < minimum_fit_count:
        raise ValueError(
            "support plane was not stable enough across the initial window: "
            f"{len(fits)} successful frames < {minimum_fit_count}"
        )

    normals = np.stack([fit.normal_world for fit in fits])
    consensus = np.median(normals, axis=0)
    consensus /= np.linalg.norm(consensus)
    for _ in range(3):
        angular_residual = np.arccos(
            np.clip(normals @ consensus, -1.0, 1.0)
        )
        median = float(np.median(angular_residual))
        mad = float(
            np.median(np.abs(angular_residual - median))
        )
        threshold = max(
            np.deg2rad(0.5),
            median + 3.0 * 1.4826 * mad,
        )
        robust = angular_residual <= threshold
        if int(np.sum(robust)) < minimum_fit_count:
            raise ValueError(
                "support-plane normals have too few temporal inliers"
            )
        consensus = np.median(normals[robust], axis=0)
        consensus /= np.linalg.norm(consensus)

    angular_residual_degrees = np.degrees(
        np.arccos(np.clip(normals @ consensus, -1.0, 1.0))
    )
    temporal_p95 = float(
        np.percentile(angular_residual_degrees, 95.0)
    )
    if temporal_p95 > maximum_temporal_p95_degrees:
        raise ValueError(
            "support-plane normal is not temporally stable: "
            f"p95={temporal_p95:.3f} deg > "
            f"{maximum_temporal_p95_degrees:.3f} deg"
        )
    correction_degrees = float(
        np.degrees(np.arccos(np.clip(consensus[2], -1.0, 1.0)))
    )
    if correction_degrees > maximum_tilt_degrees:
        raise ValueError(
            "support-plane correction exceeds the gravity prior: "
            f"{correction_degrees:.3f} deg"
        )

    support_plane_centroid = np.median(
        np.stack(
            [
                fit.centroid_world_xyz_m
                for fit, keep in zip(fits, robust, strict=True)
                if keep
            ]
        ),
        axis=0,
    )
    if pivot_world_xyz_m is None:
        pivot = support_plane_centroid
        resolved_pivot_source = "robust initial support-plane centroid"
    else:
        pivot = np.asarray(pivot_world_xyz_m, dtype=np.float64)
        if pivot.shape != (3,) or not np.isfinite(pivot).all():
            raise ValueError("pivot_world_xyz_m must be a finite 3-vector")
        if not pivot_source:
            raise ValueError(
                "pivot_source is required with pivot_world_xyz_m"
            )
        resolved_pivot_source = str(pivot_source)
    leveling_linear = _normal_to_z_only_affine(consensus)
    determinant = float(np.linalg.det(leveling_linear))
    condition_number = float(np.linalg.cond(leveling_linear))
    if determinant <= 1e-8:
        raise ValueError(
            "support-plane affine correction has non-positive determinant"
        )
    if (
        not np.isfinite(condition_number)
        or condition_number > maximum_affine_condition_number
    ):
        raise ValueError(
            "support-plane affine correction is ill-conditioned: "
            f"{condition_number:.6g} > "
            f"{maximum_affine_condition_number:.6g}"
        )
    transformed_camera_origins = apply_world_affine_to_points(
        poses[:, :3],
        linear_world=leveling_linear,
        pivot_world_xyz_m=pivot,
        maximum_condition_number=maximum_affine_condition_number,
    )
    leveled_normal = _transform_plane_normal(
        consensus,
        linear_world=leveling_linear,
    )
    transformed_support_centroid = apply_world_affine_to_points(
        support_plane_centroid,
        linear_world=leveling_linear,
        pivot_world_xyz_m=pivot,
        maximum_condition_number=maximum_affine_condition_number,
    )
    support_height_before_m = float(support_plane_centroid[2])
    support_height_after_m = float(transformed_support_centroid[2])
    report: dict[str, object] = {
        "schema_version": 2,
        "method": SUPPORT_PLANE_LEVELING_METHOD,
        "status": "applied",
        "estimator": (
            "per-frame deterministic RANSAC over the full reconstructed "
            "point cloud with a +Z gravity prior, followed by robust "
            "temporal spherical consensus and one constant Z-only affine "
            "height shear"
        ),
        "window_frame_count": int(window_frames),
        "successful_plane_frame_count": int(len(fits)),
        "minimum_successful_plane_frame_count": int(minimum_fit_count),
        "failed_frames": failed_frames,
        "support_normal_world_before": consensus.tolist(),
        "support_normal_world_after": leveled_normal.tolist(),
        "correction_degrees": correction_degrees,
        "linear_world": leveling_linear.tolist(),
        "linear_world_determinant": determinant,
        "linear_world_condition_number": condition_number,
        "pivot_world_xyz_m": pivot.tolist(),
        "pivot_source": resolved_pivot_source,
        "support_plane_centroid_world_xyz_m": (
            support_plane_centroid.tolist()
        ),
        "support_height_before_m": support_height_before_m,
        "support_height_after_m": support_height_after_m,
        "support_height_delta_m": (
            support_height_after_m - support_height_before_m
        ),
        "temporal_normal_residual_degrees": {
            "p50": float(np.percentile(angular_residual_degrees, 50.0)),
            "p95": temporal_p95,
            "max": float(np.max(angular_residual_degrees)),
        },
        "plane_support_fraction": {
            "p50": float(
                np.percentile(
                    [fit.support_fraction for fit in fits],
                    50.0,
                )
            ),
            "p95": float(
                np.percentile(
                    [fit.support_fraction for fit in fits],
                    95.0,
                )
            ),
            "min": float(
                np.min([fit.support_fraction for fit in fits])
            ),
        },
        "plane_residual_p95_m": {
            "p50": float(
                np.percentile(
                    [fit.residual_p95_m for fit in fits],
                    50.0,
                )
            ),
            "p95": float(
                np.percentile(
                    [fit.residual_p95_m for fit in fits],
                    95.0,
                )
            ),
            "max": float(
                np.max([fit.residual_p95_m for fit in fits])
            ),
        },
        "quality_gates": {
            "residual_threshold_m": float(residual_threshold_m),
            "maximum_input_tilt_degrees": float(maximum_tilt_degrees),
            "minimum_support_fraction": float(minimum_support_fraction),
            "maximum_temporal_p95_degrees": float(
                maximum_temporal_p95_degrees
            ),
            "minimum_affine_determinant_exclusive": 1e-8,
            "maximum_affine_condition_number": float(
                maximum_affine_condition_number
            ),
            "passed": True,
        },
        "fits_yaw": False,
        "fits_rotation": False,
        "fits_height_shear": True,
        "fits_translation": False,
        "fits_scale": False,
        "preserves_per_frame_camera_motion": True,
        "preserves_input_camera_quaternions": True,
        "preserves_frame_alignment": True,
        "preserves_world_xy": True,
        "preserves_camera_orientation": True,
        "application_count": 1,
        "applied_to": (
            "reconstructed world points before exact fingertip-TCP XYZ "
            "translation; never encoded into camera quaternions"
        ),
    }
    return SceneLeveling(
        camera_pose_cv_xyzw=poses.copy(),
        transformed_camera_origins_world_xyz_m=(
            transformed_camera_origins
        ),
        linear_world=leveling_linear,
        pivot_world_xyz_m=pivot,
        report=report,
    )
