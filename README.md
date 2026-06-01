# SASA: Semantic Anchors and Spatial Arbitration

Official implementation of the ICME 2026 paper:

**Weakly Supervised Incremental Segmentation via Semantic Anchors and Spatial Arbitration**

Zhonggai Wang, Kai Fang, and Guangyu Gao  
Beijing Institute of Technology

[![Paper](https://img.shields.io/badge/Paper-ICME%202026-blue)](#citation)
[![Task](https://img.shields.io/badge/Task-WILSS-green)](#overview)
[![Code](https://img.shields.io/badge/Code-PyTorch-red)](#getting-started)

## Overview

SASA is a drift-resilient framework for **Weakly Supervised Incremental Semantic Segmentation (WILSS)**. WILSS incrementally learns new semantic classes from image-level labels while preserving previously learned classes. The main challenge is that CAM-based pseudo labels and old-model predictions can inject contradictory supervision, which gradually causes feature drift and class overwriting.

SASA addresses this problem at two levels:

- **Drift-Resilient Semantic Anchors (DSA):** learnable class-level anchors provide stable semantic references, while Elastic Residual Tokens (ERTs) capture instance-specific variations without corrupting class identity.
- **Spatial Label Arbitration (SLA):** geometry-aware object masks enforce a "One Object, One Class" constraint and denoise conflicting pseudo labels before they supervise the model.

Together, DSA and SLA stabilize representations and improve pseudo-label reliability under weak incremental supervision.

## Highlights

- Stabilizes class identity with rigid learnable semantic anchors.
- Uses elastic residual adaptation for controlled instance-level refinement.
- Filters noisy pseudo labels through geometry-aware spatial arbitration.
- Achieves strong results on Pascal VOC and COCO-to-VOC incremental segmentation benchmarks.

## Results

Main overlap-setting results reported in the paper. `P` denotes pixel-level supervision and `I` denotes image-level supervision.

| Method | Sup. | 10-10 VOC All | 15-5 VOC All | COCO-to-VOC All | 10-2 VOC All | 10-5 VOC All |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| WILSON (ViT) | I | 68.8 | 69.1 | 41.8 | 46.2 | 66.9 |
| ToCo (ViT) | I | 67.4 | 67.9 | 41.1 | 43.4 | 65.7 |
| **SASA (ViT)** | **I** | **74.3** | **73.0** | **47.5** | **51.5** | **71.7** |
| Joint (ViT) | P | 78.2 | 78.2 | 50.3 | 78.2 | 78.2 |

Disjoint-setting results from the supplementary material:

| Method | Sup. | 10-10 VOC All | 15-5 VOC All |
| --- | --- | ---: | ---: |
| WILSON (ViT) | I | 64.5 | 68.2 |
| ToCo (ViT) | I | 63.2 | 67.4 |
| **SASA (ViT)** | **I** | **67.3** | **71.9** |
| Joint (ViT) | P | 78.2 | 78.2 |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ZhonggaiWang/SASA.git
cd SASA
```

### 2. Create the environment

The code is implemented in PyTorch. A typical environment contains:

```bash
conda create -n sasa python=3.9 -y
conda activate sasa

pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn pillow imageio opencv-python tqdm matplotlib texttable joblib timm
```

Optional CRF post-processing requires `pydensecrf`.

### 3. Prepare datasets

#### Pascal VOC 2012

Download Pascal VOC 2012 and the augmented SBD annotations. The expected structure is:

```text
VOCdevkit/
`-- VOC2012/
    |-- Annotations/
    |-- ImageSets/
    |-- JPEGImages/
    |-- SegmentationClass/
    |-- SegmentationClassAug/
    `-- SegmentationObject/
```

The repository already includes incremental split files under:

```text
datasets/voc/incremental_split/
```

#### MS COCO

For COCO experiments, prepare COCO 2014 images and VOC-style segmentation masks:

```text
MSCOCO/
|-- coco2014/
|   |-- train2014/
|   `-- val2014/
`-- SegmentationClass/
    |-- train2014/
    `-- val2014/
```

### 4. Prepare geometry priors for SLA

SLA uses object masks and geometry cues. The current code expects `.npy` files for:

- SAM/object masks
- depth maps
- normal maps

Before training, update the dataset paths in `datasets/voc.py` and the refinement paths in `process_sam.py` to match your local machine.

To refine raw SAM masks with depth and normal cues:

```bash
python process_sam.py
```

## Training

Training scripts for common Pascal VOC settings are provided:

```bash
bash sem_20-0.sh
bash sem_10-10.sh
bash sem_15-5.sh
bash sem_10-5.sh
bash sem_10-2.sh
```

You can also launch a specific step manually:

```bash
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29106 \
  scripts/dist_train_voc_seg_neg.py \
  --step 0 \
  --task 10-5 \
  --max_iters 20000 \
  --lr 6e-5 \
  --work_dir output_voc
```

For incremental steps:

```bash
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29106 \
  scripts/dist_train_voc_seg_neg.py \
  --step 1 \
  --task 10-5 \
  --max_iters 8000 \
  --lr 2e-5 \
  --work_dir output_voc
```

## Evaluation

Evaluate a trained Pascal VOC checkpoint:

```bash
python tools/infer_seg_voc.py \
  --model_path path/to/checkpoint.pth \
  --backbone vit_base_patch16_224 \
  --infer_set val
```

For COCO:

```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=29501 \
  tools/infer_seg_coco_ddp.py \
  --model_path path/to/checkpoint.pth
```

## Repository Structure

```text
SASA/
|-- continual/          # Incremental training logic
|-- datasets/           # VOC/COCO datasets and incremental splits
|-- model/              # Backbone, decoder, losses, and SASA modules
|-- scripts/            # Distributed training entry points
|-- tools/              # Inference and evaluation utilities
|-- utils/              # Optimization, CAM, CRF, and metric utilities
|-- process_sam.py      # Geometry-aware SAM mask refinement
`-- sem_*.sh            # VOC incremental training scripts
```

## Citation

If this project is useful for your research, please cite:

```bibtex
@inproceedings{wang2026sasa,
  title     = {Weakly Supervised Incremental Segmentation via Semantic Anchors and Spatial Arbitration},
  author    = {Wang, Zhonggai and Fang, Kai and Gao, Guangyu},
  booktitle = {IEEE International Conference on Multimedia and Expo (ICME)},
  year      = {2026}
}
```

## Acknowledgement

This codebase builds on common weakly supervised semantic segmentation and incremental segmentation tooling, including ViT/DeiT backbones from `timm`, CAM-based pseudo-label generation, and CRF refinement. We thank the authors of these open-source projects and related WILSS methods.
