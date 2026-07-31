#!/usr/bin/env python3
"""Deterministic, image-space G1 arm appearance transfer from the real meshes.

The G1 URDF is evaluated once at its zero pose only to build a canonical
forearm/hand appearance atlas. Per-video-frame work is strictly two-dimensional:
the same atlas is warped into each SAM arm mask and stabilized with optical
flow. No per-frame robot FK, IK, joint fitting, or mesh pose optimization is
performed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
from typing import Iterable
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
import trimesh
from tqdm.auto import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = (
    WORKSPACE_ROOT
    / "assets"
    / "unitree_description"
    / "urdf"
    / "g1"
    / "main_with_hand_new.urdf"
)
DEFAULT_DATASET = WORKSPACE_ROOT / "dataset" / "basic_pick_place"
DEFAULT_MASK_ROOT = (
    WORKSPACE_ROOT
    / "output_new"
    / "basic_pick_place"
    / "videos_handless"
    / "_masks"
    / "observation.images.ego"
    / "chunk-000"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "mesh_output"


@dataclass(frozen=True)
class Episode:
    name: str
    video_path: Path
    mask_dir: Path


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int
    fps: float


@dataclass(frozen=True)
class VisualPart:
    link: str
    mesh_path: Path
    transform: np.ndarray
    color: np.ndarray


@dataclass(frozen=True)
class Atlas:
    rendered_rgba: np.ndarray
    texture_rgb: np.ndarray


def parse_vector(value: str | None, length: int) -> np.ndarray:
    if not value:
        return np.zeros(length, dtype=np.float64)
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"Expected {length} values, got {value!r}")
    return result


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def origin_matrix(element: ET.Element | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if element is None:
        return result
    result[:3, :3] = rpy_matrix(parse_vector(element.get("rpy"), 3))
    result[:3, 3] = parse_vector(element.get("xyz"), 3)
    return result


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def load_materials(root: ET.Element) -> dict[str, np.ndarray]:
    materials: dict[str, np.ndarray] = {}
    for element in root.findall("material"):
        name = element.get("name")
        color = element.find("color")
        if name and color is not None:
            rgba = parse_vector(color.get("rgba"), 4)
            materials[name] = np.clip(rgba[:3] * 255.0, 0, 255)
    return materials


def mesh_path_from_urdf(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        raise ValueError(
            f"package:// paths are not supported by this local asset: {filename}"
        )
    return (urdf_path.parent / filename).resolve()


def collect_visual_parts(urdf_path: Path, side: str) -> list[VisualPart]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    materials = load_materials(root)
    links = {
        element.get("name"): element
        for element in root.findall("link")
        if element.get("name")
    }
    children: dict[str, list[tuple[str, np.ndarray]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        if not parent_name or not child_name:
            continue
        children.setdefault(parent_name, []).append(
            (child_name, origin_matrix(joint.find("origin")))
        )

    root_link = f"{side}_elbow_link"
    if root_link not in links:
        raise ValueError(f"Missing {root_link} in {urdf_path}")
    link_transforms = {root_link: np.eye(4, dtype=np.float64)}
    queue = [root_link]
    while queue:
        parent = queue.pop(0)
        for child, parent_from_child in children.get(parent, []):
            if child in link_transforms:
                continue
            link_transforms[child] = (
                link_transforms[parent] @ parent_from_child
            )
            queue.append(child)

    parts: list[VisualPart] = []
    for link_name, transform in link_transforms.items():
        link = links.get(link_name)
        if link is None:
            continue
        for visual in link.findall("visual"):
            mesh_element = visual.find("geometry/mesh")
            if mesh_element is None or not mesh_element.get("filename"):
                continue
            mesh_path = mesh_path_from_urdf(
                urdf_path, mesh_element.get("filename", "")
            )
            if not mesh_path.is_file():
                raise FileNotFoundError(
                    f"{link_name} references missing mesh {mesh_path}"
                )
            scale = parse_vector(mesh_element.get("scale"), 3)
            if not np.any(scale):
                scale = np.ones(3, dtype=np.float64)
            visual_transform = transform @ origin_matrix(
                visual.find("origin")
            )
            visual_transform = visual_transform.copy()
            visual_transform[:3, :3] *= scale[None, :]
            material = visual.find("material")
            material_name = material.get("name") if material is not None else ""
            color = materials.get(
                material_name or "",
                np.array([150.0, 150.0, 155.0]),
            )
            parts.append(
                VisualPart(
                    link_name,
                    mesh_path,
                    visual_transform,
                    color.astype(np.float64),
                )
            )
    if not parts:
        raise ValueError(f"No visual mesh parts found below {root_link}")
    return parts


def load_part_geometry(
    parts: list[VisualPart],
) -> list[tuple[VisualPart, np.ndarray]]:
    result: list[tuple[VisualPart, np.ndarray]] = []
    for part in parts:
        mesh = trimesh.load_mesh(part.mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Unsupported mesh type for {part.mesh_path}")
        vertices = transform_points(
            np.asarray(mesh.vertices, dtype=np.float64), part.transform
        )
        result.append((part, vertices))
    return result


def projection_basis(all_vertices: np.ndarray) -> np.ndarray:
    centered = all_vertices - np.mean(all_vertices, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered))
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    longitudinal = vectors[:, order[0]]
    # The elbow-to-wrist chain advances primarily along +X in the selected
    # elbow-link frame. Keep the atlas direction deterministic.
    if longitudinal[0] < 0:
        longitudinal = -longitudinal
    lateral = vectors[:, order[1]]
    depth = np.cross(longitudinal, lateral)
    depth /= np.linalg.norm(depth)
    lateral = np.cross(depth, longitudinal)
    lateral /= np.linalg.norm(lateral)
    return np.stack([lateral, longitudinal, depth], axis=1)


def render_mesh_atlas(
    urdf_path: Path,
    side: str,
    *,
    width: int = 384,
    height: int = 1024,
    include_hand: bool = True,
) -> Atlas:
    parts = collect_visual_parts(urdf_path, side)
    if not include_hand:
        parts = [
            part
            for part in parts
            if "_elbow_" in part.link or "_wrist_" in part.link
        ]
        if not parts:
            raise ValueError(
                f"No elbow/wrist visual mesh parts found for {side}"
            )
    geometry = load_part_geometry(parts)
    sampled = [
        vertices[:: max(1, len(vertices) // 20000)]
        for _, vertices in geometry
    ]
    all_vertices = np.concatenate(sampled, axis=0)
    basis = projection_basis(all_vertices)
    projected_all = all_vertices @ basis
    x_min, y_min = np.percentile(projected_all[:, :2], 0.2, axis=0)
    x_max, y_max = np.percentile(projected_all[:, :2], 99.8, axis=0)
    x_padding = 0.12 * (x_max - x_min)
    y_padding = 0.04 * (y_max - y_min)
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding

    projected_parts: list[
        tuple[float, VisualPart, np.ndarray]
    ] = []
    depth_values: list[float] = []
    for part, vertices in geometry:
        projected = vertices @ basis
        depth = float(np.mean(projected[:, 2]))
        projected_parts.append((depth, part, projected))
        depth_values.append(depth)
    depth_low, depth_high = min(depth_values), max(depth_values)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.uint8)
    for depth, part, projected in sorted(
        projected_parts, key=lambda item: item[0]
    ):
        x = (projected[:, 0] - x_min) / max(1e-9, x_max - x_min)
        y = (projected[:, 1] - y_min) / max(1e-9, y_max - y_min)
        pixels = np.stack(
            [
                np.clip(np.rint(x * (width - 1)), 0, width - 1),
                np.clip(
                    np.rint((1.0 - y) * (height - 1)),
                    0,
                    height - 1,
                ),
            ],
            axis=1,
        ).astype(np.int32)
        hull = cv2.convexHull(pixels)
        depth_fraction = (depth - depth_low) / max(
            1e-9, depth_high - depth_low
        )
        shade = 0.82 + 0.24 * depth_fraction
        color = np.clip(part.color * shade, 0, 255).astype(np.uint8)
        cv2.fillConvexPoly(rgb, hull, color.tolist(), lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(alpha, hull, 255, lineType=cv2.LINE_AA)
        outline = np.clip(color.astype(np.int16) - 42, 0, 255).astype(
            np.uint8
        )
        cv2.polylines(
            rgb,
            [hull],
            True,
            outline.tolist(),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # Add fixed, subtle cylindrical lighting so the atlas reads as a material
    # rather than flat segmentation colors.
    x_coordinate = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    highlight = 0.90 + 0.13 * np.sqrt(
        np.maximum(0.0, 1.0 - x_coordinate**2)
    )
    rgb = np.clip(
        rgb.astype(np.float32) * highlight[None, :, None], 0, 255
    ).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])

    known = alpha > 0
    if not np.any(known):
        raise RuntimeError(f"Empty {side} mesh atlas")

    # Turn the irregular robot silhouette into a rectangular material field.
    # Each row is a real cross-section of the rendered meshes, stretched only
    # laterally. This preserves the longitudinal hand/wrist/forearm ordering
    # without the radiating artifacts produced by 2-D nearest-pixel fill.
    texture = np.zeros_like(rgb)
    valid_rows: list[int] = []
    for row in range(height):
        columns = np.flatnonzero(known[row])
        if len(columns) == 0:
            continue
        left_column = int(columns[0])
        right_column = int(columns[-1])
        cross_section = rgb[
            row : row + 1, left_column : right_column + 1
        ]
        texture[row] = cv2.resize(
            cross_section,
            (width, 1),
            interpolation=cv2.INTER_LINEAR,
        )[0]
        valid_rows.append(row)
    valid_array = np.asarray(valid_rows, dtype=np.int32)
    for row in range(height):
        if row in valid_rows:
            continue
        nearest_row = int(valid_array[np.argmin(np.abs(valid_array - row))])
        texture[row] = texture[nearest_row]
    return Atlas(rgba, texture)


def numeric_video_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 2**31 - 1, path.name


def discover_episodes(
    dataset_root: Path, mask_root: Path, requested: set[str] | None
) -> list[Episode]:
    videos = sorted(dataset_root.glob("*.mp4"), key=numeric_video_sort_key)
    masks = sorted(
        (path for path in mask_root.glob("file-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if not videos or len(videos) != len(masks):
        raise ValueError(
            f"Found {len(videos)} videos and {len(masks)} mask directories"
        )
    result = [
        Episode(video.stem, video, mask)
        for video, mask in zip(videos, masks, strict=True)
        if requested is None or video.stem in requested
    ]
    if not result:
        raise ValueError("No requested episode was found")
    return result


def video_info(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open {path}")
        result = VideoInfo(
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        capture.release()
    if min(result.width, result.height, result.frame_count) <= 0:
        raise ValueError(f"Invalid video metadata for {path}: {result}")
    return result


def parse_frame_spec(
    value: str | None, frame_count: int
) -> set[int] | None:
    if value is None:
        return None
    result: set[int] = set()
    for token in value.split(","):
        index = int(token.strip())
        if index < 0:
            index += frame_count
        if not 0 <= index < frame_count:
            raise ValueError(f"Frame {token!r} is outside this episode")
        result.add(index)
    return result


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read {path}")
    if mask.shape != shape:
        raise ValueError(
            f"Mask {path} has shape {mask.shape}; expected {shape}"
        )
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def significant_components(mask: np.ndarray) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    minimum = max(128, int(mask.size * 0.0002))
    components: list[np.ndarray] = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < minimum:
            continue
        components.append(labels == label)
    return components


def component_coordinates(
    component: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = component.shape
    y, x = np.nonzero(component)
    points = np.stack([x, y], axis=1).astype(np.float32)
    border_distance = np.minimum.reduce(
        [x, width - 1 - x, y, height - 1 - y]
    )
    border = points[border_distance <= max(5, int(min(height, width) * 0.008))]
    if len(border) >= 8:
        root = np.median(border, axis=0)
    else:
        root = np.median(
            points[points[:, 1] >= np.percentile(points[:, 1], 95)],
            axis=0,
        )

    centered = points - np.mean(points, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered))
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    centroid_direction = np.mean(points, axis=0) - root
    if float(np.dot(axis, centroid_direction)) < 0:
        axis = -axis
    perpendicular = np.array([-axis[1], axis[0]], dtype=np.float32)

    relative = points - root
    longitudinal = relative @ axis
    lateral = relative @ perpendicular
    s_low, s_high = np.percentile(longitudinal, [1.0, 99.0])
    t_low, t_high = np.percentile(lateral, [1.0, 99.0])
    s = np.clip(
        (longitudinal - s_low) / max(1e-6, s_high - s_low), 0.0, 1.0
    )
    t = np.clip(
        (lateral - t_low) / max(1e-6, t_high - t_low), 0.0, 1.0
    )
    return points, s, t


def fresh_mesh_texture(
    mask: np.ndarray, left_atlas: Atlas, right_atlas: Atlas
) -> np.ndarray:
    height, width = mask.shape
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    for component in significant_components(mask):
        points, longitudinal, lateral = component_coordinates(component)
        # In an egocentric image the arm entering from the left half is the
        # left arm. This label only selects the mirrored G1 atlas.
        x_root_side = float(np.median(points[:, 0]))
        atlas = left_atlas if x_root_side < width / 2 else right_atlas
        atlas_height, atlas_width = atlas.texture_rgb.shape[:2]
        map_x = np.clip(
            np.rint(lateral * (atlas_width - 1)),
            0,
            atlas_width - 1,
        ).astype(np.int32)
        map_y = np.clip(
            np.rint((1.0 - longitudinal) * (atlas_height - 1)),
            0,
            atlas_height - 1,
        ).astype(np.int32)
        colors = atlas.texture_rgb[map_y, map_x]
        x = points[:, 0].astype(np.int32)
        y = points[:, 1].astype(np.int32)
        texture[y, x] = colors
    return texture


def optical_flow_warp(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_texture: np.ndarray,
    previous_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = current_gray.shape
    small_size = (max(32, width // 2), max(32, height // 2))
    current_small = cv2.resize(
        current_gray, small_size, interpolation=cv2.INTER_AREA
    )
    previous_small = cv2.resize(
        previous_gray, small_size, interpolation=cv2.INTER_AREA
    )
    # Current -> previous is the inverse sampling direction required by remap.
    flow = cv2.calcOpticalFlowFarneback(
        current_small,
        previous_small,
        None,
        0.5,
        3,
        19,
        3,
        5,
        1.2,
        0,
    )
    flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= width / small_size[0]
    flow[..., 1] *= height / small_size[1]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped_texture = cv2.remap(
        previous_texture,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_mask = cv2.remap(
        previous_mask,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return warped_texture, warped_mask


def apply_scene_shading(
    texture: np.ndarray, original_rgb: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    illumination = cv2.GaussianBlur(gray, (0, 0), sigmaX=18.0)
    selected = illumination[mask > 0]
    median = float(np.median(selected)) if len(selected) else 150.0
    local = np.clip(illumination / max(1.0, median), 0.82, 1.18)
    global_gain = np.clip(median / 155.0, 0.78, 1.12)
    shaded = texture.astype(np.float32) * (
        local * global_gain
    )[..., None]
    return np.clip(shaded, 0, 255).astype(np.uint8)


def strict_composite(
    original_rgb: np.ndarray,
    texture_rgb: np.ndarray,
    mask: np.ndarray,
    feather: float,
) -> np.ndarray:
    binary = mask > 0
    if feather <= 0:
        alpha = binary.astype(np.float32)
    else:
        distance = cv2.distanceTransform(
            binary.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        alpha = np.clip(distance / feather, 0.0, 1.0)
        alpha[~binary] = 0.0
    result = (
        texture_rgb.astype(np.float32) * alpha[..., None]
        + original_rgb.astype(np.float32) * (1.0 - alpha[..., None])
    )
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def make_comparison(
    original: np.ndarray, mask: np.ndarray, result: np.ndarray
) -> np.ndarray:
    overlay = original.astype(np.float32)
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    alpha = (mask.astype(np.float32) / 255.0 * 0.45)[..., None]
    overlay = overlay * (1.0 - alpha) + red * alpha
    return np.concatenate(
        [original, np.clip(overlay, 0, 255).astype(np.uint8), result],
        axis=1,
    )


def start_video_writer(
    path: Path, info: VideoInfo
) -> subprocess.Popen[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{info.width}x{info.height}",
        "-r",
        f"{info.fps:.12g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def process_episode(
    episode: Episode,
    left_atlas: Atlas,
    right_atlas: Atlas,
    args: argparse.Namespace,
) -> dict[str, object]:
    info = video_info(episode.video_path)
    paths = sorted(episode.mask_dir.glob("*.png"))
    if len(paths) != info.frame_count:
        raise ValueError(
            f"{episode.name}: {info.frame_count} frames, {len(paths)} masks"
        )
    selected = parse_frame_spec(args.frames, info.frame_count)
    complete_video = selected is None
    selected = selected or set(range(info.frame_count))

    output_root = args.output_dir / episode.name
    frames_dir = output_root / "frames"
    comparisons_dir = output_root / "comparisons"
    for directory in (frames_dir, comparisons_dir):
        directory.mkdir(parents=True, exist_ok=True)
    temporary_video = output_root / f".{episode.name}_g1_mesh.tmp.mp4"
    final_video = output_root / f"{episode.name}_g1_mesh.mp4"
    writer = (
        start_video_writer(temporary_video, info) if complete_video else None
    )

    capture = cv2.VideoCapture(str(episode.video_path))
    previous_gray: np.ndarray | None = None
    previous_texture: np.ndarray | None = None
    previous_mask: np.ndarray | None = None
    try:
        for index in tqdm(
            range(info.frame_count),
            desc=f"mesh episode {episode.name}",
            unit="frame",
            dynamic_ncols=True,
        ):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index}")
            original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            mask = read_mask(paths[index], (info.height, info.width))
            fresh = fresh_mesh_texture(mask, left_atlas, right_atlas)
            stabilized = fresh
            if (
                previous_gray is not None
                and previous_texture is not None
                and previous_mask is not None
                and args.temporal_blend > 0
            ):
                warped, warped_mask = optical_flow_warp(
                    previous_gray,
                    gray,
                    previous_texture,
                    previous_mask,
                )
                overlap = (mask > 0) & (warped_mask > 0)
                stabilized = fresh.copy()
                blend = args.temporal_blend
                stabilized[overlap] = np.clip(
                    warped[overlap].astype(np.float32) * blend
                    + fresh[overlap].astype(np.float32) * (1.0 - blend),
                    0,
                    255,
                ).astype(np.uint8)
            previous_gray = gray
            previous_texture = stabilized
            previous_mask = mask

            shaded = apply_scene_shading(stabilized, original, mask)
            result = strict_composite(
                original, shaded, mask, args.mask_feather
            )
            if index in selected:
                Image.fromarray(result).save(frames_dir / f"{index:06d}.png")
                comparison = make_comparison(original, mask, result)
                Image.fromarray(comparison).save(
                    comparisons_dir / f"{index:06d}.jpg",
                    quality=92,
                    subsampling=0,
                )
            if writer is not None:
                assert writer.stdin is not None
                writer.stdin.write(np.ascontiguousarray(result).tobytes())
    finally:
        capture.release()
        if writer is not None:
            assert writer.stdin is not None
            writer.stdin.close()
            status = writer.wait()
            if status:
                raise RuntimeError(f"ffmpeg exited with status {status}")
    if writer is not None:
        temporary_video.replace(final_video)
    return {
        "episode": episode.name,
        "source": str(episode.video_path),
        "mask_dir": str(episode.mask_dir),
        "frame_count": info.frame_count,
        "video": str(final_video) if complete_video else None,
        "frames_dir": str(frames_dir),
        "comparisons_dir": str(comparisons_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map a canonical atlas rendered from the actual G1 meshes into "
            "the existing 2-D human-arm masks."
        )
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", nargs="+")
    parser.add_argument(
        "--frames",
        help="Comma-separated preview frames; omit for complete videos",
    )
    parser.add_argument(
        "--temporal-blend",
        type=float,
        default=0.0,
        help=(
            "Optional optical-flow-warped previous texture weight. Disabled "
            "by default because high-contrast G1 part boundaries should stay "
            "sharp."
        ),
    )
    parser.add_argument("--mask-feather", type=float, default=3.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.temporal_blend <= 1.0:
        raise ValueError("--temporal-blend must be in [0, 1]")
    if args.mask_feather < 0:
        raise ValueError("--mask-feather must be non-negative")
    args.urdf = args.urdf.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.mask_root = args.mask_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    left = render_mesh_atlas(args.urdf, "left")
    right = render_mesh_atlas(args.urdf, "right")
    atlas_dir = args.output_dir / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(left.rendered_rgba).save(atlas_dir / "left_g1.png")
    Image.fromarray(right.rendered_rgba).save(atlas_dir / "right_g1.png")

    requested = set(args.episodes) if args.episodes else None
    episodes = discover_episodes(
        args.dataset_root, args.mask_root, requested
    )
    results = [
        process_episode(episode, left, right, args) for episode in episodes
    ]
    manifest = {
        "method": "deterministic G1 mesh atlas mapped inside 2-D arm masks",
        "urdf": str(args.urdf),
        "per_frame_robot_kinematics": False,
        "temporal_stabilization": (
            "disabled"
            if args.temporal_blend == 0
            else "Farneback optical flow"
        ),
        "temporal_blend": args.temporal_blend,
        "mask_feather": args.mask_feather,
        "atlas": {
            "left": str(atlas_dir / "left_g1.png"),
            "right": str(atlas_dir / "right_g1.png"),
        },
        "episodes": results,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
