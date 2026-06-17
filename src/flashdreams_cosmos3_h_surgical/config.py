"""Static FlashDreams and Cosmos 3 runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class RuntimeConfig:
    name: str = "cosmos3-h-surgical-simulator"
    profile: str = "cosmos3-super-webrtc"
    host: str = "127.0.0.1"
    port: int = 8080
    width: int = 512
    height: int = 288
    fps: int = 30
    min_inference_fps: float = 1.0
    action_chunk_steps: int = 12
    num_steps: int = 16
    guidance: float = 3.0
    shift: float = 5.0
    seed: int = 3407
    num_gpus: int = 8
    master_port: int = 29517
    torchrun: str = "torchrun"
    compile: bool = False
    cuda_graphs: bool = False
    anchor_blend: float = 0.0
    hf_repo_id: str = "hcltech-robotics/cosmos3-h-surgical-simulator-alpha"
    hf_revision: str = "main"
    image_size: int = 256
    domain_name: str = "open_h_surgical_sim"
    experiment: str = "cosmos3_super_openh_surgical_lora"
    config_file: str = "cosmos3_h_surgical_simulator/experiment.py"
    model_mode: str = "forward_dynamics"
    view_point: str = "ego_view"
    use_ema_weights: bool = False
    guardrails: bool = False
    turn_server: str = ""
    turn_username: str = ""
    turn_credential: str = ""
    browser_turn_port: int = 3478
    browser_ice_transport_policy: str = "all"
    prompt: str = "Endoscopic robotic surgery with dual robotic instruments manipulating tissue."

    @classmethod
    def from_toml(cls, path: Path) -> "RuntimeConfig":
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        runtime = payload.get("runtime", {})
        values = {field: runtime.get(field, getattr(cls, field)) for field in cls.__dataclass_fields__}
        return cls(**values)


DEFAULT_RUNTIME = RuntimeConfig()

COSMOS3_H_SURGICAL_CONFIGS = {
    DEFAULT_RUNTIME.name: DEFAULT_RUNTIME,
}


def __getattr__(name: str):
    if name not in {
        "PIPELINE_COSMOS3_H_SURGICAL_SIMULATOR",
        "RUNNER_COSMOS3_H_SURGICAL_SIMULATOR",
    }:
        raise AttributeError(name)
    from .flashdreams_runner import (
        PIPELINE_COSMOS3_H_SURGICAL_SIMULATOR,
        RUNNER_COSMOS3_H_SURGICAL_SIMULATOR,
    )

    values = {
        "PIPELINE_COSMOS3_H_SURGICAL_SIMULATOR": PIPELINE_COSMOS3_H_SURGICAL_SIMULATOR,
        "RUNNER_COSMOS3_H_SURGICAL_SIMULATOR": RUNNER_COSMOS3_H_SURGICAL_SIMULATOR,
    }
    globals().update(values)
    return values[name]
