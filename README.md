# Cosmos 3 H Surgical Simulator for FlashDreams

FlashDreams runner and WebRTC serving adapter for
[`hcltech-robotics/cosmos3-h-surgical-simulator-alpha`](https://huggingface.co/hcltech-robotics/cosmos3-h-surgical-simulator-alpha),
an action-conditioned LoRA checkpoint on `nvidia/Cosmos3-Super`.

This package provides the runtime layer around the Cosmos 3 Super checkpoint:
browser control, 44D surgical action conditioning, Hugging Face checkpoint
resolution, Cosmos 3 sample construction, and WebRTC video delivery.

## Runtime contract

- FlashDreams runner slug: `cosmos3-h-surgical-simulator`
- Model repo: `hcltech-robotics/cosmos3-h-surgical-simulator-alpha`
- Source repo: `https://github.com/hcltech-robotics/cosmos-3-h-surgical-simulator`
- Base model: `nvidia/Cosmos3-Super`
- Checkpoint format: PyTorch Distributed Checkpoint under `checkpoints/iter_000000060`
- Cosmos 3 experiment: `cosmos3_super_openh_surgical_lora`
- Cosmos config file: `cosmos3_h_surgical_simulator/experiment.py`
- Training data: Open-H-Embodiment surgical robotics trajectories
- Training hardware: 8x GB200 GPUs
- Action tensor: `12 x 44`, with a 20D dual-tool dVRK command prefix
- Frame staging: `512 x 288` RGB, Cosmos `image_size = 256`, 30 FPS metadata
- Sampling defaults: 16 steps, guidance 3.0, shift 5.0, regular weights only

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## Project page

The bundled website presents the video result, comparison carousel, model
contract, training setup, and run commands:

```bash
python3 -m http.server 8088 \
  --bind 127.0.0.1 \
  --directory src/flashdreams_cosmos3_h_surgical/web
```

Open `http://127.0.0.1:8088`.

For a machine that runs the Cosmos 3 sampler, install the inference extras and
make a matching Cosmos 3 checkout available:

```bash
python3 -m pip install -e ".[inference]"
export COSMOS3_ROOT=/opt/cosmos/packages/cosmos3
export WAN_VAE_PATH=/opt/cosmos/checkpoints/wan22_vae/Wan2.2_VAE.pth
```

Authentication for gated Hub assets should come from the standard Hugging Face
environment:

```bash
export HF_TOKEN=...
```

## WebRTC runtime

```bash
cosmos3-h-surgical-webrtc \
  --config configs/cosmos3-super-webrtc.toml \
  --cosmos3-root "$COSMOS3_ROOT" \
  --project-root "$PWD" \
  --first-frame runs/local/first_frame.png
```

Open `http://127.0.0.1:8080`. The browser sends keyboard state over a WebRTC
data channel, the server converts that state into a validated `12 x 44` action
chunk, and Cosmos 3 generates a 13-frame action-conditioned rollout.

`--checkpoint` is optional. When omitted, the runtime resolves the latest
checkpoint from `hcltech-robotics/cosmos3-h-surgical-simulator-alpha` through
`huggingface_hub.snapshot_download`.

## Batch runner

```bash
cosmos3-h-surgical-run \
  --config configs/cosmos3-super-webrtc.toml \
  --cosmos3-root "$COSMOS3_ROOT" \
  --project-root "$PWD" \
  --image-path runs/local/first_frame.png \
  --actions-path runs/local/actions.json \
  --output-root runs/cosmos3-h-surgical
```

The action loader accepts either `.npy` files or a JSON object with an
`actions` array. This command writes a valid neutral `12 x 44` JSON action
file:

```bash
python3 - <<'PY'
import json
from pathlib import Path

row = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    *([0.0] * 24),
]
Path("runs/local").mkdir(parents=True, exist_ok=True)
Path("runs/local/actions.json").write_text(json.dumps({"actions": [row] * 12}, indent=2) + "\n")
PY
```

The runtime validates the full `12 x 44` shape and writes both `actions.npy` and
`actions.json` into the sample directory used by Cosmos 3.

To preconvert a JSON action file before inference:

```bash
cosmos3-h-surgical-run \
  --config configs/cosmos3-super-webrtc.toml \
  --cosmos3-root "$COSMOS3_ROOT" \
  --project-root "$PWD" \
  --image-path runs/local/first_frame.png \
  --actions-path runs/local/actions.json \
  --preconvert-actions-to runs/local/actions.npy \
  --output-root runs/cosmos3-h-surgical
```

## Action contract

Each row has 44 floating point values:

| Index range | Field |
| --- | --- |
| `0:3` | left tool relative translation `(dx, dy, dz)` |
| `3:9` | left tool relative rotation, 6D rotation representation |
| `9` | left jaw/gripper target |
| `10:13` | right tool relative translation `(dx, dy, dz)` |
| `13:19` | right tool relative rotation, 6D rotation representation |
| `19` | right jaw/gripper target |
| `20:44` | reserved bridge channels, zero padded |

Keyboard control integrates held keys into a physical action trajectory at the
runtime FPS, then validates the same tensor shape used by batch inference.

## FlashDreams discovery

The package exposes the standard runner entry point:

```toml
[project.entry-points."flashdreams.runner_configs"]
"cosmos3-h-surgical-simulator" = "flashdreams_cosmos3_h_surgical.config:RUNNER_COSMOS3_H_SURGICAL_SIMULATOR"
```

After `pip install -e .`, FlashDreams can discover the runner by slug:

```bash
flashdreams-run cosmos3-h-surgical-simulator \
  --cosmos3-root "$COSMOS3_ROOT" \
  --image-path runs/local/first_frame.png \
  --actions-path runs/local/actions.npy \
  --output-dir runs/flashdreams
```

## Network and credential policy

The tracked WebRTC config binds to `127.0.0.1` and contains no TURN credential.
For a remote deployment, keep TURN settings in a deployment-specific TOML file
that is not committed. `/status` reports whether TURN is configured, but it does
not return the credential.

## Verification

```bash
python3 -m pytest -q
python3 -m ruff check .
```

The CPU tests cover:

- FlashDreams static runner registration
- `12 x 44` action-vector validation
- browser keyboard to action-chunk conversion
- JSON action loading and `.npy` preconversion
- Cosmos 3 input JSON construction
- `cosmos_framework.scripts.inference` command construction with
  `--no-use-ema-weights`
- WebRTC status payload without credential disclosure

A GPU rollout uses the same batch or WebRTC command with a valid Cosmos 3
checkout, VAE path, Hugging Face token, and the Hub checkpoint.
