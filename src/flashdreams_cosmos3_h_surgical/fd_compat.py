"""FlashDreams interface compatibility for local validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

try:
    from flashdreams.infra.config import InstantiateConfig as InstantiateConfig
    from flashdreams.infra.runner import Runner as Runner
    from flashdreams.infra.runner import RunnerConfig as RunnerConfig
except ModuleNotFoundError:

    @dataclass(kw_only=True)
    class InstantiateConfig:
        _target: type[Any] | None = None

        def setup(self) -> Any:
            if self._target is None:
                raise TypeError("_target is required")
            return self._target(self)

    ConfigT = TypeVar("ConfigT")
    PipelineT = TypeVar("PipelineT")

    @dataclass(kw_only=True)
    class RunnerConfig(InstantiateConfig):
        runner_name: str
        description: str = ""
        output_dir: Path = Path("runs")
        device: str = "cuda"

    class Runner(Generic[ConfigT, PipelineT]):
        config: ConfigT
        pipeline: PipelineT

        def __init__(self, config: ConfigT) -> None:
            self.config = config

        def run(self) -> None:
            raise NotImplementedError
