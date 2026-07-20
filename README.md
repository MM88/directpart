# DirectPart

Zero-shot, training-free 3D part segmentation on ShapeNetPart.

DirectPart segments 3D shapes into semantic parts without any training or fine-tuning. Each mesh is rendered from viewpoints uniformly sampled on an icosphere, every rendered view is segmented with text-prompted [SAM3](https://github.com/facebookresearch/sam3), and the resulting 2D masks are back-projected onto the annotated point cloud through the depth buffer. Per-point labels are assigned by majority vote across views.


## Repository contents

- `directpart.py` — full pipeline: rendering, SAM3 segmentation, back-projection, evaluation
- `refine.py` — optional post-hoc label refinement (kNN smoothing or CRF)
- `texture_loader.py` — optional textured rendering (`--use_texture`), baking mesh textures into vertex colors

## Installation

Tested with Python 3.12, PyTorch 2.7+, CUDA 12.6+ on Linux.

**1. Create an environment and install SAM3** (follow the [official instructions](https://github.com/facebookresearch/sam3)):

```bash
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e . && cd ..
```

**2. Download the SAM3 checkpoint.** The checkpoints are gated: request access on the [SAM3 Hugging Face repository](https://huggingface.co/facebook/sam3), then authenticate with your Hugging Face token (`hf auth login`) and download `sam3.pt`.

**3. Install the remaining dependencies:**

```bash
pip install -r requirements.txt
```

For the optional CRF refinement (`--refine crf`), also install `pygco`.

## Data

The evaluation uses two components of the ShapeNet ecosystem:

- **ShapeNetCore** meshes (`model_normalized.obj` per model), obtained from [shapenet.org](https://shapenet.org) upon registration
- **ShapeNetPart annotations** (PartAnnotation: `.pts` point clouds and expert-verified `.seg` labels) with the official test split file (`shuffled_test_file_list.json`)

Expected layout:

```
<core_root>/<synset_id>/<model_id>/models/model_normalized.obj
<part_root>/<synset_id>/points/<model_id>.pts
<part_root>/<synset_id>/expert_verified/points_label/<model_id>.seg
```

### Main options

| Option | Default | Description |
|---|---|---|
| `--subdivision` | 1 | Icosphere subdivision: 0 = 12 views, 1 = 42, 2 = 162 |
| `--zoom` | 1.3 | Zoom-in factor (> 1 moves the camera closer) |
| `--instance_policy` | `score` | SAM3 instance selection: `top1`, `all`, or `score` (threshold `--score_thresh`) |
| `--prompt_variant` | `B` | Prompt formulation: `A` bare part, `B` category + part, `C` full phrase |
| `--use_texture` | off | Render with mesh textures baked into vertex colors |
| `--refine` | `none` | Post-hoc refinement: `none`, `smooth`, or `crf` |

## License

[MIT / to be confirmed]


