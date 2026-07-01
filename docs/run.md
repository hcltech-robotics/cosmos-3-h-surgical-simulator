# Run instructions

## Overview

`Cosmos-3-ac-Surgical` is a LoRA fine-tune of `nvidia/Cosmos3-Super`
for action-conditioned surgical robotics video generation. It was trained on
Open-H-Embodiment surgical robotics trajectories using 8x GB200 GPUs.

The FlashDreams adapter in this repository provides:

- a batch runner: `cosmos-3-ac-surgical-run`
- a WebRTC runner: `cosmos-3-ac-surgical-webrtc`
- a FlashDreams runner slug: `cosmos-3-ac-surgical`
- JSON action loading and JSON-to-NumPy preconversion
- Hugging Face checkpoint resolution for
  `hcltech-robotics/cosmos-3-ac-surgical-alpha`

## Runtime contract

- Base model: `nvidia/Cosmos3-Super`
- LoRA: `Cosmos-3-ac-Surgical`
- Config: `configs/cosmos3-super-webrtc.toml`
- Cosmos 3 experiment: `cosmos3_super_openh_surgical_lora`
- Cosmos config file: `cosmos3_h_surgical_simulator/experiment.py`
- Checkpoint: `checkpoints/iter_000000060`
- Frame staging: `512 x 288` RGB
- Cosmos image size: `256`
- Action tensor: `12 x 44`
- Sampling: 16 steps, guidance 3.0, shift 5.0
- Weights: regular weights, EMA disabled

## Install

```bash
python3 -m pip install -e ".[inference]"
export COSMOS3_ROOT=/opt/cosmos/packages/cosmos3
export WAN_VAE_PATH=/opt/cosmos/checkpoints/wan22_vae/Wan2.2_VAE.pth
export HF_TOKEN=<token>
```

`HF_TOKEN` is read by `huggingface_hub` when `--checkpoint` is omitted.

## Prepare actions

The runtime accepts `.npy` action chunks or JSON files with an `actions` array.
The JSON file must contain 12 rows and 44 numeric values per row.

```bash
python3 - <<'PY'
import json
from pathlib import Path

row = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    *([0.0] * 24),
]
Path("runs/input").mkdir(parents=True, exist_ok=True)
Path("runs/input/actions.json").write_text(json.dumps({"actions": [row] * 12}, indent=2) + "\n")
PY
```

## Batch inference

```bash
cosmos-3-ac-surgical-run \
  --config configs/cosmos3-super-webrtc.toml \
  --cosmos3-root "$COSMOS3_ROOT" \
  --project-root "$PWD" \
  --image-path runs/input/first_frame.png \
  --actions-path runs/input/actions.json \
  --preconvert-actions-to runs/input/actions.npy \
  --output-root runs/cosmos-3-ac-surgical
```

When `--checkpoint` is not supplied, the runner resolves the checkpoint from
`hcltech-robotics/cosmos-3-ac-surgical-alpha`.

## WebRTC serving

```bash
cosmos-3-ac-surgical-webrtc \
  --config configs/cosmos3-super-webrtc.toml \
  --cosmos3-root "$COSMOS3_ROOT" \
  --project-root "$PWD" \
  --first-frame runs/input/first_frame.png
```

Open `http://127.0.0.1:8080`.

The tracked config binds to `127.0.0.1` and does not include TURN credentials.
For remote serving, keep deployment-specific TURN settings outside the tracked
config.

## Project page

The project page is bundled with the Python package:

```bash
python3 -m http.server 8088 \
  --bind 127.0.0.1 \
  --directory src/flashdreams_cosmos3_h_surgical/web
```

Open `http://127.0.0.1:8088`.

The page includes `video.mp4`, the source split image, and comparison slides for
visual fidelity, motion, physics, and hallucination control.
