"""FlashDreams runner registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch.nn as nn

from .config import DEFAULT_RUNTIME, RuntimeConfig
from .endpoint import SurgicalSimulatorEndpoint
from .fd_compat import InstantiateConfig, Runner, RunnerConfig
from .io import load_action_chunk, load_first_frame


@dataclass(kw_only=True)
class Cosmos3HSurgicalPipelineConfig(InstantiateConfig):
    _target: type["Cosmos3HSurgicalPipeline"] = field(
        default_factory=lambda: Cosmos3HSurgicalPipeline
    )
    name: str = DEFAULT_RUNTIME.name
    runtime: RuntimeConfig = DEFAULT_RUNTIME


class Cosmos3HSurgicalPipeline(nn.Module):
    def __init__(self, config: Cosmos3HSurgicalPipelineConfig) -> None:
        super().__init__()
        self.config = config

    def generate(
        self,
        *,
        checkpoint: Path | None,
        cosmos3_root: Path,
        project_root: Path,
        output_root: Path,
        first_frame_path: Path,
        action_chunk_path: Path,
        prompt: str,
    ) -> Path:
        frame = load_first_frame(
            first_frame_path,
            width=self.config.runtime.width,
            height=self.config.runtime.height,
        )
        chunk = load_action_chunk(action_chunk_path)
        endpoint = SurgicalSimulatorEndpoint(
            runtime=self.config.runtime,
            checkpoint=checkpoint,
            cosmos3_root=cosmos3_root,
            project_root=project_root,
            output_root=output_root,
        )
        result = endpoint.generate(
            first_frame=frame.image,
            action_chunk=chunk,
            prompt=prompt,
            sample_name=self.config.name,
        )
        return result.video_path


@dataclass(kw_only=True)
class Cosmos3HSurgicalRunnerConfig(RunnerConfig):
    _target: type["Cosmos3HSurgicalRunner"] = field(default_factory=lambda: Cosmos3HSurgicalRunner)
    pipeline: Cosmos3HSurgicalPipelineConfig = field(default_factory=Cosmos3HSurgicalPipelineConfig)
    checkpoint: Path | None = None
    cosmos3_root: Path = Path("/opt/cosmos/packages/cosmos3")
    project_root: Path = Path(".")
    image_path: Path = Path("first_frame.png")
    actions_path: Path = Path("actions.npy")
    prompt: str = DEFAULT_RUNTIME.prompt


class Cosmos3HSurgicalRunner(Runner[Cosmos3HSurgicalRunnerConfig, Cosmos3HSurgicalPipeline]):
    config: Cosmos3HSurgicalRunnerConfig
    pipeline: Cosmos3HSurgicalPipeline

    def __init__(self, config: Cosmos3HSurgicalRunnerConfig) -> None:
        self.config = config
        config.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = config.pipeline.setup().to(device=config.device).eval()

    def run(self) -> None:
        self.pipeline.generate(
            checkpoint=self.config.checkpoint,
            cosmos3_root=self.config.cosmos3_root,
            project_root=self.config.project_root,
            output_root=self.config.output_dir,
            first_frame_path=self.config.image_path,
            action_chunk_path=self.config.actions_path,
            prompt=self.config.prompt,
        )


PIPELINE_COSMOS3_H_SURGICAL_SIMULATOR = Cosmos3HSurgicalPipelineConfig(
    name=DEFAULT_RUNTIME.name,
    runtime=DEFAULT_RUNTIME,
)
RUNNER_COSMOS3_H_SURGICAL_SIMULATOR = Cosmos3HSurgicalRunnerConfig(
    runner_name=PIPELINE_COSMOS3_H_SURGICAL_SIMULATOR.name,
    description="Cosmos 3 H Surgical Simulator WebRTC inference runtime.",
    pipeline=PIPELINE_COSMOS3_H_SURGICAL_SIMULATOR,
)

__all__ = [
    "Cosmos3HSurgicalPipeline",
    "Cosmos3HSurgicalPipelineConfig",
    "Cosmos3HSurgicalRunner",
    "Cosmos3HSurgicalRunnerConfig",
    "PIPELINE_COSMOS3_H_SURGICAL_SIMULATOR",
    "RUNNER_COSMOS3_H_SURGICAL_SIMULATOR",
]
