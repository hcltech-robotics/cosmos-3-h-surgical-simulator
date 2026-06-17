"""Runtime input, action conversion, and media helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .action import validate_action_chunk


class InputError(ValueError):
    """Raised when a runtime input cannot be loaded."""


@dataclass(frozen=True)
class FirstFrame:
    image: Image.Image
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    path: Path | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "source_size": list(self.source_size),
            "target_size": list(self.target_size),
            "path": None if self.path is None else str(self.path),
        }


def load_first_frame(path: Path, *, width: int, height: int) -> FirstFrame:
    if not path.is_file():
        raise InputError(f"first frame not found: {path}")
    image = Image.open(path).convert("RGB")
    source_size = image.size
    target_size = (int(width), int(height))
    if source_size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    return FirstFrame(image=image, source_size=source_size, target_size=target_size, path=path)


def _coerce_action_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in ("actions", "action", "rows", "action_chunk"):
            if key in payload:
                return payload[key]
        raise InputError("action JSON object must contain actions, action, rows, or action_chunk")
    return payload


def load_action_chunk(path: Path) -> list[list[float]]:
    if not path.is_file():
        raise InputError(f"action chunk not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        arr = np.asarray(_coerce_action_payload(json.loads(path.read_text(encoding="utf-8"))), dtype=np.float32)
    else:
        arr = np.load(path).astype(np.float32)
    chunk = arr.tolist()
    validate_action_chunk(chunk)
    return chunk


def preconvert_action_chunk(path: Path, output_path: Path | None = None) -> Path:
    chunk = load_action_chunk(path)
    output = output_path or path.with_suffix(".npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(chunk, dtype=np.float32))
    return output


def write_action_files(sample_dir: Path, chunk: list[list[float]]) -> tuple[Path, Path]:
    validate_action_chunk(chunk)
    sample_dir.mkdir(parents=True, exist_ok=True)
    npy_path = sample_dir / "actions.npy"
    json_path = sample_dir / "actions.json"
    arr = np.asarray(chunk, dtype=np.float32)
    np.save(npy_path, arr)
    json_path.write_text(json.dumps(arr.astype(float).tolist()) + "\n", encoding="utf-8")
    return npy_path, json_path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_video_frames(path: Path) -> list[np.ndarray]:
    if not path.is_file():
        raise InputError(f"video not found: {path}")
    try:
        import av
    except ImportError as exc:
        raise InputError("PyAV is required for video frame extraction") from exc
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def write_single_frame_video(path: Path, image: Image.Image, *, fps: int) -> None:
    try:
        import av
    except ImportError as exc:
        raise InputError("PyAV is required for video writing") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    width, height = rgb.size
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=max(1, int(fps)))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.asarray(rgb), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
