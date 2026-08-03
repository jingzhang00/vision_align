# Installation

```bash
git submodule update --init --recursive
```
Download [Releases V0.1.0](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0) to the `ProPainter/weights` folder.

```bash
uv sync
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

- follow `Inpaint-Anything/README.md -> Download the weights` to put the weights
```
pretrained_models/
├── sam3.pt                 # SAM 3, images + video tracking   (gated, see above)
├── config.json             # SAM 3 config, downloaded alongside sam3.pt
├── big-lama/               # LaMa, default image removal      (auto)
│   ├── config.yaml
│   └── models/best.ckpt
└── propainter/             # ProPainter, default video inpainting (auto)
    ├── ProPainter.pth
    ├── recurrent_flow_completion.pth
    └── raft-things.pth
pytracking/pretrain/
└── vitb_384_mae_ce_32x4_ep300.pth   # legacy --tracker ostrack only  (manual)
weights/
└── mobile_sam.pt           # MobileSAM, ships with the repo
```
- replace `sam3/sam3/model_builder.py` with `utils/model_builder.py` to bypass Hugginface check

## Run tasks
To process episode 0 of every task under `dataset/egodex_test`:

```bash
bash batch_egodex.sh
```

```bash
bash batch_nymeria.sh
```

## Run one task and retain intermediates

```bash
.venv/bin/python repaint/egodex/run_single_task.py \
  --task flip_coin \
  --dataset-root dataset \
  --output-root egodex_single_task

.venv/bin/python repaint/nymeria/run_single_task.py \
  --task c000 \
  --session 20230814_s1_david_hall_act0_n9k3wo
```

Use `--overwrite` only when the selected output should be rebuilt from
scratch.

## Fast trajectory-only check

```bash
.venv/bin/python -m repaint.traj_replay \
  --dataset egodex_single_task/pour/lerobot \
  --episode 0

.venv/bin/python -m repaint.traj_replay \
  --dataset egodex
```

## Adjustable Isaac view

To inspect a prepared episode in isaacsim:

```bash
.venv/bin/python -m repaint.vda_scene_isaac \
  --prepared egodex_single_task/pour/prepared/episode_000000 \
  --interactive \
  --loop \
  --no-headless

.venv/bin/python -m repaint.vda_scene_isaac \
  --prepared nymeria/single_task/prepared/episode_000000 \
  --interactive \
  --loop \
  --no-headless \
  --scene-leveling none \
  --scene-alignment none
```

## Nymeria

```bash
.venv/bin/python -m repaint.nymeria.nymeria_vda_pipeline \
  --dataset-root dataset/nymeria \
  --output-root nymeria/vda \
  --session 20230814_s1_david_hall_act0_n9k3wo \
  --clip c024
```