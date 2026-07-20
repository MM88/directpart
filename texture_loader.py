"""Load ShapeNet meshes with textures baked into per-vertex colors.

Used by directpart.py when --use_texture is set: the texture is baked into
vertex colors and passed to the renderer, so only the image seen by SAM3
changes; depth capture and back-projection are unaffected.
"""
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyvista as pv
import trimesh


def _parse_mtl_for_texture(mtl_path: Path, obj_dir: Path) -> Optional[Path]:
    """Look for texture map references in the .mtl file."""
    try:
        with open(mtl_path, 'r') as f:
            content = f.read()
        patterns = [r'map_Kd\s+(.+)', r'map_Ka\s+(.+)', r'map_d\s+(.+)']
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                tex_ref = match.group(1).strip()
                # ShapeNet models store textures in a handful of layouts
                candidates = [
                    obj_dir / tex_ref,
                    obj_dir / Path(tex_ref).name,
                    obj_dir / "untitled" / Path(tex_ref).name,
                    obj_dir / "image" / Path(tex_ref).name,
                    obj_dir / "images" / Path(tex_ref).name,
                ]
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
        print("  [i] MTL found but no usable texture reference")
    except Exception as e:
        print(f"  [!] MTL parsing failed: {e}")
    return None


def load_mesh_with_texture(obj_path: Path) -> Tuple[pv.PolyData, Optional[np.ndarray], None]:
    """Load a ShapeNet mesh with its texture baked into vertex colors.

    Returns (mesh, vertex_colors, None). If no usable texture or material
    color is found, vertex_colors is None and the renderer falls back to
    the default uniform gray.
    """
    obj_path = Path(obj_path)
    obj_dir = obj_path.parent
    texture_file = None

    # 1. Texture referenced by the .mtl file
    mtl_path = obj_path.with_suffix('.mtl')
    if not mtl_path.exists():
        mtl_path = obj_dir / "model_normalized.mtl"
    if mtl_path.exists():
        texture_file = _parse_mtl_for_texture(mtl_path, obj_dir)

    # 2. Fallback: scan common texture folders, skipping non-albedo maps
    if texture_file is None:
        texture_folders = ['untitled', 'image', 'images', 'textures', '.']
        texture_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tga']
        for folder_name in texture_folders:
            folder = obj_dir / folder_name if folder_name != '.' else obj_dir
            if not folder.exists():
                continue
            for ext in texture_extensions:
                candidates = list(folder.glob(f"*{ext}"))
                if candidates:
                    for c in candidates:
                        nl = c.stem.lower()
                        if not any(x in nl for x in ['normal', 'spec', 'bump', 'ao', 'rough', 'metal']):
                            texture_file = c
                            break
                    if texture_file is None:
                        texture_file = candidates[0]
                    break
            if texture_file:
                break

    # 3. Load with trimesh and bake texture/material into vertex colors
    vertex_colors = None
    try:
        tm_mesh = trimesh.load(str(obj_path), force='mesh')
        if hasattr(tm_mesh, 'visual'):
            if texture_file and texture_file.exists() and hasattr(tm_mesh.visual, 'uv'):
                try:
                    cv = tm_mesh.visual.to_color()
                    if getattr(cv, 'vertex_colors', None) is not None:
                        vertex_colors = cv.vertex_colors[:, :3]
                        print(f"  [i] Texture baked: {len(vertex_colors)} vertex colors")
                except Exception as e:
                    print(f"  [!] Texture baking failed: {e}")
            if vertex_colors is None:
                try:
                    cv = tm_mesh.visual.to_color()
                    if getattr(cv, 'vertex_colors', None) is not None:
                        vc = cv.vertex_colors
                        if not np.allclose(vc[:, :3], vc[0, :3]):
                            vertex_colors = vc[:, :3]
                            print(f"  [i] Material colors: {len(vertex_colors)} vertex colors")
                        else:
                            print("  [i] Uniform material color, using default gray")
                except Exception as e:
                    print(f"  [!] Color extraction failed: {e}")

        faces = tm_mesh.faces
        pv_faces = np.column_stack([np.full(len(faces), 3), faces]).flatten()
        mesh_pv = pv.PolyData(tm_mesh.vertices, pv_faces)
    except Exception as e:
        print(f"  [!] trimesh load failed, falling back to pv.read: {e}")
        mesh_pv = pv.read(str(obj_path))
        vertex_colors = None

    mesh_pv.translate(-np.array(mesh_pv.center), inplace=True)

    # Sanity check: color array must match the vertex count
    if vertex_colors is not None and len(vertex_colors) != mesh_pv.n_points:
        print(f"  [!] Color/vertex mismatch ({len(vertex_colors)} vs {mesh_pv.n_points}), ignoring colors")
        vertex_colors = None

    return mesh_pv, vertex_colors, None
