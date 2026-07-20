#!/usr/bin/env python3
"""DirectPart: zero-shot, training-free 3D part segmentation on ShapeNetPart.

Each mesh is rendered from viewpoints sampled on an icosphere, every view is
segmented with text-prompted SAM3, and the 2D masks are back-projected onto
the annotated point cloud through the depth buffer. Per-point labels are
assigned by majority vote across views.

Example:
    python directpart.py \
        --category 03001627 \
        --core_root /path/to/ShapeNetCore \
        --part_root /path/to/PartAnnotation \
        --test_split_file shuffled_test_file_list.json \
        --sam3_checkpoint /path/to/sam3.pt
"""
import gc
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import torch
from PIL import Image
import trimesh
from scipy.spatial import cKDTree

from refine import refine_labels, get_compat
from texture_loader import load_mesh_with_texture

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CATEGORY_NAMES = {
    '02691156': 'Airplane', '02773838': 'Bag', '02954340': 'Cap',
    '02958343': 'Car', '03001627': 'Chair', '03261776': 'Earphone',
    '03467517': 'Guitar', '03624134': 'Knife', '03636649': 'Lamp',
    '03642806': 'Laptop', '03790512': 'Motorbike', '03797390': 'Mug',
    '03948459': 'Pistol', '04099429': 'Rocket', '04225987': 'Skateboard',
    '04379243': 'Table',
}

CATEGORY_LABELS = {
    '02691156': ['airplane fuselage', 'airplane wing', 'airplane tail', 'airplane engine'],
    '02773838': ['bag handle', 'bag body'],
    '02954340': ['cap body', 'cap visor'],
    '02958343': ['car roof', 'car hood', 'car wheel', 'car body'],
    '03001627': ['chair back', 'chair seat', 'chair leg', 'chair arm'],
    '03261776': ['earphone', 'earphone headband'],
    '03467517': ['guitar head', 'guitar neck', 'guitar body'],
    '03624134': ['knife blade', 'knife handle'],
    '03636649': ['lamp canopy', 'lamp shade', 'lamp base'],
    '03642806': ['laptop keyboard', 'laptop screen'],
    '03790512': ['motorbike gas tank', 'motorbike seat', 'motorbike wheel', 'motorbike handle', 'motorbike light'],
    '03797390': ['mug handle', 'mug body'],
    '03948459': ['pistol barrel', 'pistol grip', 'pistol trigger'],
    '04099429': ['rocket body', 'rocket fin', 'rocket nose'],
    '04225987': ['skateboard wheel', 'skateboard deck'],
    '04379243': ['table top', 'table leg', 'table shelf'],
}


def build_prompts(category_id: str, variant: str) -> List[str]:
    """Build the SAM3 text prompts for a category.

    Variants (see the paper's prompt sensitivity analysis):
      A: bare part name ("wing")
      B: category + part ("airplane wing") -- default
      C: full phrase ("the wing of an airplane")
    """
    base = CATEGORY_LABELS.get(category_id, ['part'])
    cat = CATEGORY_NAMES.get(category_id, '').lower()
    parts = [lbl.split(' ', 1)[1] if ' ' in lbl else lbl for lbl in base]
    if variant == 'A':
        return parts
    if variant == 'C':
        return [f"the {p} of a {cat}" for p in parts]
    return base


class ShapeNetDataset:
    def __init__(self, core_root: str, part_root: str, test_split_file: Optional[str] = None):
        self.core_root = Path(core_root)
        self.part_root = Path(part_root)
        self.test_split_file = test_split_file
        self.test_models_by_category = {}
        if test_split_file:
            self._load_test_split(test_split_file)

    def _load_test_split(self, split_file: str):
        print(f"\n[i] Test split: {split_file}")
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        for entry in split_data:
            parts = entry.split('/')
            if len(parts) >= 3:
                cat_id, model_id = parts[1], parts[2]
                self.test_models_by_category.setdefault(cat_id, []).append(model_id)

    def get_labels_for_category(self, category_id: str) -> List[str]:
        return CATEGORY_LABELS.get(category_id, ['part'])

    def get_mesh_path(self, category_id, model_id) -> Path:
        return self.core_root / category_id / model_id / "models" / "model_normalized.obj"

    def get_points_path(self, category_id, model_id) -> Path:
        return self.part_root / category_id / "points" / f"{model_id}.pts"

    def get_labels_path(self, category_id, model_id) -> Path:
        return self.part_root / category_id / "expert_verified" / "points_label" / f"{model_id}.seg"

    def list_models(self, category_id: str, use_test_split: bool = True) -> List[str]:
        mesh_dir = self.core_root / category_id
        if not mesh_dir.exists():
            print(f"  [!] Mesh directory not found: {mesh_dir}")
            return []
        mesh_models = set(d.name for d in mesh_dir.iterdir() if d.is_dir())
        labels_dir = self.part_root / category_id / "expert_verified" / "points_label"
        label_models = (set(f.stem for f in labels_dir.glob("*.seg") if ":Zone" not in f.name)
                        if labels_dir.exists() else set())
        available = mesh_models & label_models
        if use_test_split and self.test_split_file and category_id in self.test_models_by_category:
            test_models = set(self.test_models_by_category[category_id])
            common = sorted(available & test_models)
            print(f"  Category {category_id} ({CATEGORY_NAMES.get(category_id, '?')}): "
                  f"meshes={len(mesh_models)} labels={len(label_models)} "
                  f"test={len(test_models)} usable={len(common)}")
            return common
        return sorted(available)

    def load_points_and_labels(self, category_id, model_id) -> Tuple[np.ndarray, np.ndarray]:
        points = np.loadtxt(str(self.get_points_path(category_id, model_id)))
        labels = np.loadtxt(str(self.get_labels_path(category_id, model_id)), dtype=int) - 1
        return points, labels


# ---------------------------------------------------------------------------
# Icosphere view sampling
# ---------------------------------------------------------------------------

class IcosphereViewGenerator:
    """Uniformly distributed viewpoints from a subdivided icosahedron.

    Subdivision level 0 gives 12 views, level 1 gives 42, level 2 gives 162.
    """

    def __init__(self, subdivision_level: int = 1):
        self.vertices, self.faces = [], []
        self._create_icosahedron()
        self._subdivide(subdivision_level)

    def _create_icosahedron(self):
        phi = (1 + np.sqrt(5)) / 2
        verts = [[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                 [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                 [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]]
        self.vertices = [np.array(v) / np.linalg.norm(v) for v in verts]
        self.faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]

    def _get_middle_point(self, v1, v2):
        m = (v1 + v2) / 2
        return m / np.linalg.norm(m)

    def _subdivide(self, levels: int):
        for _ in range(levels):
            new_faces, cache = [], {}
            for face in self.faces:
                v0, v1, v2 = [self.vertices[i] for i in face]

                def get_mid(va, vb):
                    key = tuple(sorted([tuple(va), tuple(vb)]))
                    if key not in cache:
                        self.vertices.append(self._get_middle_point(va, vb))
                        cache[key] = len(self.vertices) - 1
                    return cache[key]

                a, b, c = get_mid(v0, v1), get_mid(v1, v2), get_mid(v2, v0)
                i0, i1, i2 = face
                new_faces.extend([[i0, a, c], [i1, b, a], [i2, c, b], [a, b, c]])
            self.faces = new_faces


# ---------------------------------------------------------------------------
# Multi-view rendering (RGB + depth buffer)
# ---------------------------------------------------------------------------

@dataclass
class ViewData:
    view_id: int
    image: np.ndarray
    camera_position: np.ndarray
    focal_point: np.ndarray
    view_up: np.ndarray
    fov: float = 50.0
    depth: Optional[np.ndarray] = None


class MultiViewRenderer:
    def __init__(self, subdivision_level=1, image_size=512, fov=50.0, zoom=1.3):
        self.image_size, self.fov, self.zoom = image_size, fov, zoom
        self.icosphere = IcosphereViewGenerator(subdivision_level)

    def render_views(self, mesh: pv.PolyData, bounding_radius: float,
                     vertex_colors: Optional[np.ndarray] = None) -> List[ViewData]:
        views = []
        fov_rad = np.radians(self.fov)
        # zoom > 1 moves the camera closer to the object (zoom-in), see Eq. 1
        camera_distance = (bounding_radius / np.tan(fov_rad / 2)) * 1.5 / self.zoom
        focal_point = np.array([0, 0, 0])
        use_colors = vertex_colors is not None
        if use_colors:
            mesh['RGB'] = vertex_colors

        for i, vertex in enumerate(self.icosphere.vertices):
            cam_pos = vertex * camera_distance
            view_up = np.array([0, 1, 0]) if abs(vertex[1]) < 0.9 else np.array([0, 0, 1])

            plotter = pv.Plotter(off_screen=True, window_size=[self.image_size, self.image_size])
            plotter.remove_all_lights()
            plotter.set_background(color='#ffffff', top='#b0b0b0')
            if use_colors:
                plotter.add_mesh(mesh, scalars='RGB', rgb=True, smooth_shading=True,
                                 backface_culling=False, ambient=0.3, diffuse=0.7, specular=0.0)
            else:
                plotter.add_mesh(mesh, color='#b0b0b0', smooth_shading=True,
                                 backface_culling=False, ambient=0.3, diffuse=0.7, specular=0.0)
            plotter.add_light(pv.Light(position=(cam_pos[0] + 1, cam_pos[1] + 1, cam_pos[2] + 1),
                                       focal_point=(0, 0, 0), intensity=0.7, color='white'))
            plotter.add_light(pv.Light(position=(-cam_pos[0], -cam_pos[1] + 0.5, cam_pos[2]),
                                       focal_point=(0, 0, 0), intensity=0.3, color='white'))
            plotter.add_light(pv.Light(light_type='headlight', intensity=0.25))
            plotter.camera_position = [cam_pos.tolist(), focal_point.tolist(), view_up.tolist()]
            plotter.camera.view_angle = self.fov

            image = plotter.screenshot(return_img=True)
            depth = plotter.get_image_depth()  # negative values; NaN on background
            plotter.close()
            del plotter

            views.append(ViewData(view_id=i, image=image, camera_position=cam_pos,
                                  focal_point=focal_point, view_up=view_up, fov=self.fov,
                                  depth=np.asarray(depth, dtype=np.float64)))
        pv.close_all()
        gc.collect()
        return views


# ---------------------------------------------------------------------------
# SAM3 segmenter (loaded once per run)
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    view_id: int
    masks: Dict[str, np.ndarray]


class SAM3Segmenter:
    def __init__(self, prompts: List[str], checkpoint_path: str,
                 instance_policy: str = "score", score_thresh: float = 0.5):
        self.prompts = prompts
        self.checkpoint_path = checkpoint_path
        self.instance_policy = instance_policy  # "top1" | "all" | "score"
        self.score_thresh = score_thresh
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is None:
            print("  [i] Loading SAM3...")
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            self.model = build_sam3_image_model(checkpoint_path=self.checkpoint_path)
            self.processor = Sam3Processor(self.model)
            print("  [i] SAM3 ready")

    def _select_instances(self, n: int, scores_np: Optional[np.ndarray]) -> List[int]:
        if scores_np is None or len(scores_np) != n:
            scores_np = None
        if self.instance_policy == "top1":
            return [int(np.argmax(scores_np))] if scores_np is not None else [0]
        if self.instance_policy == "score" and scores_np is not None:
            keep = [i for i in range(n) if scores_np[i] >= self.score_thresh]
            return keep if keep else [int(np.argmax(scores_np))]  # fallback: best instance
        return list(range(n))  # "all"

    def segment_view(self, image: np.ndarray, view_id: int) -> SegmentationResult:
        pil_image = Image.fromarray(image)
        inference_state = self.processor.set_image(pil_image)
        masks_dict = {}
        h, w = image.shape[:2]

        for prompt in self.prompts:
            try:
                output = self.processor.set_text_prompt(state=inference_state,
                                                        prompt=prompt.replace('_', ' '))
                masks = output.get("masks")
                scores = output.get("scores")
                if masks is None:
                    continue
                masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks)
                if masks_np.size == 0:
                    continue
                if masks_np.ndim == 4:
                    masks_np = masks_np[:, 0]  # (N, H, W)
                elif masks_np.ndim == 2:
                    masks_np = masks_np[None, :, :]
                if scores is not None:
                    scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
                    scores_np = scores_np.ravel()
                else:
                    scores_np = None

                keep_idx = self._select_instances(masks_np.shape[0], scores_np)
                merged = np.zeros((h, w), dtype=bool)
                for i in keep_idx:
                    if masks_np[i].max() > 0:
                        merged |= masks_np[i].astype(bool)
                if merged.any():
                    masks_dict[prompt] = merged
            except Exception as e:
                # Real errors (e.g. CUDA failures) must stay visible and not be
                # silently treated as "no detection".
                print(f"      [!] prompt '{prompt}' failed (view {view_id}): "
                      f"{type(e).__name__}: {e}")
        return SegmentationResult(view_id=view_id, masks=masks_dict)

    def segment_all_views(self, views: List[ViewData]) -> List[SegmentationResult]:
        return [self.segment_view(v.image, v.view_id) for v in views]


# ---------------------------------------------------------------------------
# Point cloud / mesh alignment
# ---------------------------------------------------------------------------

def align_pointcloud_to_mesh(points: np.ndarray, mesh_points: np.ndarray) -> np.ndarray:
    """Align the annotated point cloud to the mesh frame.

    ShapeNetPart point clouds and ShapeNetCore meshes use inconsistent axis
    conventions, so the best of a fixed set of axis-aligned rotations is
    selected by mean nearest-neighbour distance.
    """
    mesh_center = (mesh_points.max(axis=0) + mesh_points.min(axis=0)) / 2
    mesh_centered = mesh_points - mesh_center
    mesh_scale = np.abs(mesh_centered).max()

    pc_center = (points.max(axis=0) + points.min(axis=0)) / 2
    pts_centered = points - pc_center
    pts_scaled = pts_centered / np.abs(pts_centered).max() * mesh_scale

    rotations = {
        'identity': lambda p: p,
        'swap_xy': lambda p: p[:, [1, 0, 2]],
        'swap_xz': lambda p: p[:, [2, 1, 0]],
        'swap_yz': lambda p: p[:, [0, 2, 1]],
        'rot_x_90': lambda p: np.column_stack([p[:, 0], -p[:, 2], p[:, 1]]),
        'rot_x_-90': lambda p: np.column_stack([p[:, 0], p[:, 2], -p[:, 1]]),
        'rot_y_90': lambda p: np.column_stack([p[:, 2], p[:, 1], -p[:, 0]]),
        'rot_y_-90': lambda p: np.column_stack([-p[:, 2], p[:, 1], p[:, 0]]),
        'rot_z_90': lambda p: np.column_stack([-p[:, 1], p[:, 0], p[:, 2]]),
        'rot_z_-90': lambda p: np.column_stack([p[:, 1], -p[:, 0], p[:, 2]]),
    }
    tree = cKDTree(mesh_centered)
    best_score, best_pts = float('inf'), pts_scaled
    for rot_fn in rotations.values():
        rotated = rot_fn(pts_scaled.copy())
        rot_center = (rotated.max(axis=0) + rotated.min(axis=0)) / 2
        rotated_centered = rotated - rot_center
        distances, _ = tree.query(rotated_centered, k=1)
        if distances.mean() < best_score:
            best_score, best_pts = distances.mean(), rotated_centered
    return best_pts


# ---------------------------------------------------------------------------
# Back-projection via depth buffer
# ---------------------------------------------------------------------------

class PointCloudBackProjector:
    def __init__(self, points: np.ndarray, prompts: List[str]):
        self.points = points
        self.prompts = prompts
        self.n_points = len(points)
        self.point_votes = {i: {} for i in range(self.n_points)}
        self.point_tree = cKDTree(points)

    def backproject_view(self, view: ViewData, seg: SegmentationResult,
                         image_size: int, max_rays: int = 3000):
        cam, focal, up, fov, depth = (view.camera_position, view.focal_point,
                                      view.view_up, view.fov, view.depth)
        fwd = focal - cam
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(fwd, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        up = up / np.linalg.norm(up)
        tan_fov = np.tan(np.radians(fov) / 2)

        for label, mask in seg.masks.items():
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            z = depth[ys, xs]
            valid = np.isfinite(z)
            ys, xs, z = ys[valid], xs[valid], np.abs(z[valid])
            if len(ys) == 0:
                continue
            if len(ys) > max_rays:
                idx = np.random.choice(len(ys), max_rays, replace=False)
                ys, xs, z = ys[idx], xs[idx], z[idx]
            nx = (2 * xs / image_size) - 1
            ny = 1 - (2 * ys / image_size)
            dirs = (fwd[None, :]
                    + nx[:, None] * tan_fov * right[None, :]
                    + ny[:, None] * tan_fov * up[None, :])
            world = cam[None, :] + z[:, None] * dirs
            _, pids = self.point_tree.query(world, k=1)
            for pid in pids:
                self.point_votes[pid][label] = self.point_votes[pid].get(label, 0) + 1

    def backproject_all(self, views, segmentations, image_size, max_rays=3000):
        for view, seg in zip(views, segmentations):
            self.backproject_view(view, seg, image_size, max_rays)

    def get_final_labels(self) -> np.ndarray:
        pred = np.full(self.n_points, -1, dtype=int)
        for pid in range(self.n_points):
            votes = self.point_votes[pid]
            if votes:
                best = max(votes, key=votes.get)
                if best in self.prompts:
                    pred[pid] = self.prompts.index(best)
        return pred


# ---------------------------------------------------------------------------
# Metrics and output
# ---------------------------------------------------------------------------

def compute_metrics(pred, gt, label_names) -> Dict:
    valid = pred >= 0
    pred_v, gt_v = pred[valid], gt[valid]
    accuracy = (pred_v == gt_v).mean() if len(pred_v) > 0 else 0.0
    iou_per_class = {}
    for c, name in enumerate(label_names):
        pred_c, gt_c = pred_v == c, gt_v == c
        union = (pred_c | gt_c).sum()
        if union > 0:
            iou_per_class[name] = (pred_c & gt_c).sum() / union
    mean_iou = np.mean(list(iou_per_class.values())) if iou_per_class else 0.0
    return {'accuracy': accuracy, 'mean_iou': mean_iou,
            'iou_per_class': iou_per_class, 'coverage': valid.mean()}


def save_pointcloud_ply(points, labels, label_names, output_path):
    cmap = plt.cm.tab10
    colors = np.ones((len(points), 4), dtype=np.uint8) * 128
    colors[:, 3] = 255
    for i, lbl in enumerate(labels):
        if 0 <= lbl < 10:
            colors[i, :3] = [int(c * 255) for c in cmap(lbl)[:3]]
    trimesh.PointCloud(vertices=points, colors=colors).export(str(output_path))


def save_results(results, labels, output_dir, start_time, policy_str, final=False, category=""):
    ok = [r for r in results if r['status'] == 'success']
    err = [r for r in results if r['status'] == 'error']
    if not ok:
        return
    accs = [r['accuracy'] for r in ok]
    mious = [r['mean_iou'] for r in ok]
    covs = [r['coverage'] for r in ok]
    class_ious = {l: [] for l in labels}
    for r in ok:
        for l, v in r['iou_per_class'].items():
            class_ious[l].append(v)
    elapsed = datetime.now() - start_time

    fname = "results_final.txt" if final else "results_partial.txt"
    with open(output_dir / fname, 'w') as f:
        f.write("ShapeNetPart segmentation - test set\n\n")
        f.write(f"Category: {category} ({CATEGORY_NAMES.get(category, '?')})\n")
        f.write(f"Instance policy: {policy_str}\n")
        f.write(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}  Elapsed: {elapsed}\n")
        f.write(f"Models: {len(results)}  Success: {len(ok)}  Errors: {len(err)}\n\n")
        f.write(f"Accuracy:  {np.mean(accs) * 100:.2f}% +/- {np.std(accs) * 100:.2f}%\n")
        f.write(f"Mean IoU:  {np.mean(mious) * 100:.2f}% +/- {np.std(mious) * 100:.2f}%\n")
        f.write(f"Coverage:  {np.mean(covs) * 100:.2f}% +/- {np.std(covs) * 100:.2f}%\n\n")
        f.write("Per-class IoU\n")
        for l in labels:
            if class_ious[l]:
                f.write(f"  {l:22s}: {np.mean(class_ious[l]) * 100:.2f}% +/- {np.std(class_ious[l]) * 100:.2f}%\n")
        f.write("\nPer-model results\n")
        f.write(f"  {'Model ID':<40}{'Acc':>8}{'mIoU':>8}{'Cov':>8}\n")
        for r in ok:
            f.write(f"  {r['model_id']:<40}{r['accuracy'] * 100:>7.1f}%"
                    f"{r['mean_iou'] * 100:>7.1f}%{r['coverage'] * 100:>7.1f}%\n")
        if err:
            f.write("\nErrors\n")
            for r in err:
                f.write(f"  {r['model_id']}: {r.get('error')}\n")

    jname = "results_final.json" if final else "results_partial.json"
    with open(output_dir / jname, 'w') as f:
        json.dump({
            'category': category, 'split': 'test', 'instance_policy': policy_str,
            'summary': {
                'n_models': len(results), 'n_success': len(ok), 'n_errors': len(err),
                'mean_accuracy': float(np.mean(accs)), 'mean_iou': float(np.mean(mious)),
                'std_iou': float(np.std(mious)), 'mean_coverage': float(np.mean(covs)),
                'class_iou': {l: float(np.mean(v)) for l, v in class_ious.items() if v},
            },
            'results': results,
        }, f, indent=2)
    print(f"  [i] Saved: {output_dir / fname}")


# ---------------------------------------------------------------------------
# Per-model processing (shared SAM3 instance)
# ---------------------------------------------------------------------------

def process_model(model_id, dataset, segmenter, labels, args, output_dir) -> Dict:
    np.random.seed(42)
    model_output = output_dir / "pointclouds" / model_id
    model_output.mkdir(parents=True, exist_ok=True)

    if getattr(args, "use_texture", False):
        mesh, vertex_colors, _ = load_mesh_with_texture(dataset.get_mesh_path(args.category, model_id))
    else:
        mesh = pv.read(str(dataset.get_mesh_path(args.category, model_id)))
        vertex_colors = None
    mesh.translate(-np.array(mesh.center), inplace=True)
    mesh_radius = np.linalg.norm(np.array(mesh.points), axis=1).max()

    points, gt_labels = dataset.load_points_and_labels(args.category, model_id)
    points_aligned = align_pointcloud_to_mesh(points, np.array(mesh.points))

    renderer = MultiViewRenderer(args.subdivision, args.image_size, zoom=args.zoom)
    views = renderer.render_views(mesh, mesh_radius, vertex_colors)

    segmentations = segmenter.segment_all_views(views)

    backprojector = PointCloudBackProjector(points_aligned, labels)
    backprojector.backproject_all(views, segmentations, args.image_size, args.max_rays)
    pred_labels = refine_labels(
        points_aligned, backprojector.point_votes, len(points_aligned), labels,
        refine=args.refine, tree=backprojector.point_tree,
        k=args.refine_k, sigma=args.refine_sigma, alpha=args.refine_alpha,
        n_iter=args.refine_iter, lam=args.refine_lambda,
        compat=get_compat(args.category))

    metrics = compute_metrics(pred_labels, gt_labels, labels)
    save_pointcloud_ply(points_aligned, pred_labels, labels, model_output / "prediction.ply")
    save_pointcloud_ply(points_aligned, gt_labels, labels, model_output / "ground_truth.ply")

    del mesh, views, segmentations, backprojector
    pv.close_all()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {'model_id': model_id, 'status': 'success',
            'accuracy': metrics['accuracy'], 'mean_iou': metrics['mean_iou'],
            'coverage': metrics['coverage'], 'iou_per_class': metrics['iou_per_class']}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--core_root", required=True,
                   help="Path to ShapeNetCore (mesh directories per category)")
    p.add_argument("--part_root", required=True,
                   help="Path to ShapeNetPart annotations (PartAnnotation)")
    p.add_argument("--test_split_file", required=True,
                   help="Official test split JSON (shuffled_test_file_list.json)")
    p.add_argument("--sam3_checkpoint", required=True,
                   help="Path to the SAM3 checkpoint (sam3.pt)")
    p.add_argument("--category", default="02691156",
                   help="ShapeNet synset ID (default: 02691156, Airplane)")
    p.add_argument("--max_models", type=int, default=None)
    p.add_argument("--subdivision", type=int, default=1,
                   help="Icosphere subdivision level: 0=12, 1=42, 2=162 views")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--max_rays", type=int, default=3000,
                   help="Max sampled pixels per mask during back-projection")
    p.add_argument("--zoom", type=float, default=1.3,
                   help="Zoom-in factor (> 1 moves the camera closer)")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--instance_policy", choices=["top1", "all", "score"], default="top1",
                   help="SAM3 instance selection (paper results use top1)")
    p.add_argument("--score_thresh", type=float, default=0.7,
                   help="Confidence threshold tau for --instance_policy score")
    p.add_argument("--refine", choices=["none", "smooth", "crf"], default="none",
                   help="Optional post-hoc label refinement (paper results use none)")
    p.add_argument("--refine_k", type=int, default=8)
    p.add_argument("--refine_sigma", type=float, default=None)
    p.add_argument("--refine_alpha", type=float, default=0.5)
    p.add_argument("--refine_iter", type=int, default=10)
    p.add_argument("--refine_lambda", type=float, default=1.0)
    p.add_argument("--use_texture", action="store_true",
                   help="Render with mesh texture baked into vertex colors")
    p.add_argument("--prompt_variant", choices=["A", "B", "C"], default="B",
                   help="Prompt formulation (see build_prompts)")
    args = p.parse_args()

    policy_str = (f"{args.instance_policy}(tau={args.score_thresh})"
                  if args.instance_policy == "score" else args.instance_policy)

    dataset = ShapeNetDataset(args.core_root, args.part_root, args.test_split_file)
    labels = build_prompts(args.category, args.prompt_variant)

    models = dataset.list_models(args.category, use_test_split=True)
    if not models:
        print(f"\n[!] No models for category {args.category} in the test set.")
        return
    if args.max_models:
        models = models[:args.max_models]

    print(f"\nTest set - {len(models)} models")
    print(f"Category: {args.category} ({CATEGORY_NAMES.get(args.category, '?')})")
    print(f"Instance policy: {policy_str}")
    print(f"Texture: {'yes' if args.use_texture else 'no'}")
    print(f"Prompts: {labels}\n")

    output_dir = Path(args.output_dir) / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint file for resuming interrupted runs
    ckpt = output_dir / "checkpoint.json"
    all_results, done = [], set()
    if ckpt.exists():
        with open(ckpt) as f:
            all_results = json.load(f).get('results', [])
        done = set(r['model_id'] for r in all_results)
        print(f"[i] Resumed from checkpoint: {len(done)} models already processed\n")

    # SAM3 is loaded once and shared across all models
    segmenter = SAM3Segmenter(labels, checkpoint_path=args.sam3_checkpoint,
                              instance_policy=args.instance_policy,
                              score_thresh=args.score_thresh)
    segmenter.load_model()

    start = datetime.now()
    todo = [m for m in models if m not in done]
    for i, model_id in enumerate(todo):
        print(f"[{i + 1}/{len(todo)}] {model_id}")
        try:
            try:
                r = process_model(model_id, dataset, segmenter, labels, args, output_dir)
            except Exception as e:
                print(f"   [skip] {model_id}: {type(e).__name__}: {e}")
                r = {"model_id": model_id, "status": "error", "error": str(e)}
            if r.get("status") == "success":
                print(f"   [ok] Acc {r['accuracy'] * 100:.1f}%  mIoU {r['mean_iou'] * 100:.1f}%  "
                      f"Cov {r['coverage'] * 100:.1f}%")
        except Exception as e:
            traceback.print_exc()
            r = {'model_id': model_id, 'status': 'error', 'error': str(e)}
        all_results.append(r)
        if (i + 1) % 5 == 0:
            with open(ckpt, 'w') as f:
                json.dump({'results': all_results}, f)
            save_results(all_results, labels, output_dir, start, policy_str, category=args.category)

    save_results(all_results, labels, output_dir, start, policy_str, final=True, category=args.category)
    if ckpt.exists():
        ckpt.unlink()
    print(f"\n[ok] Done ({policy_str})")


if __name__ == "__main__":
    main()
