"""Cosmos 3 H Surgical Simulator endpoint client."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

from huggingface_hub import snapshot_download
from PIL import Image

from .action import CHUNK_STEPS, validate_action_chunk
from .config import RuntimeConfig
from .io import write_action_files, write_json, write_single_frame_video


class EndpointError(RuntimeError):
    """Raised when endpoint inference does not complete."""


@dataclass(frozen=True)
class GenerationResult:
    video_path: Path
    sample_dir: Path
    duration_s: float
    frame_count: int | None = None

    @property
    def fps(self) -> float | None:
        if self.frame_count is None or self.duration_s <= 0:
            return None
        return self.frame_count / self.duration_s


@dataclass
class SurgicalSimulatorEndpoint:
    runtime: RuntimeConfig
    checkpoint: Path | None
    cosmos3_root: Path
    project_root: Path
    output_root: Path = Path("runs/webrtc")
    torchrun: str | None = None

    def generate(
        self,
        *,
        first_frame: Image.Image,
        action_chunk: list[list[float]],
        prompt: str,
        sample_name: str,
    ) -> GenerationResult:
        if not self.cosmos3_root.exists():
            raise EndpointError(f"Cosmos 3 root not found: {self.cosmos3_root}")
        validate_action_chunk(action_chunk)
        sample_dir = self.output_root / "samples" / sample_name
        raw_dir = self.output_root / "cosmos3_raw"
        sample_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        first_frame_path = sample_dir / "first_frame.png"
        first_frame.save(first_frame_path)
        _actions_npy, actions_json = write_action_files(sample_dir, action_chunk)
        cosmos3_input = self._write_cosmos3_input(
            sample_dir=sample_dir,
            sample_name=sample_name,
            first_frame=first_frame,
            first_frame_path=first_frame_path,
            actions_json=actions_json,
            prompt=prompt,
        )
        payload_path = sample_dir / "runner_payload.json"
        write_json(
            payload_path,
            self._runner_payload(
                sample_name=sample_name,
                cosmos3_input=cosmos3_input,
                raw_dir=raw_dir,
            ),
        )

        command = self._command(payload_path=payload_path)
        env = self._env()
        started = time.perf_counter()
        subprocess.run(command, cwd=self.cosmos3_root, env=env, check=True)
        duration_s = time.perf_counter() - started
        video_path = self._find_video(raw_dir, sample_name)
        return GenerationResult(video_path=video_path, sample_dir=sample_dir, duration_s=duration_s)

    def _write_cosmos3_input(
        self,
        *,
        sample_dir: Path,
        sample_name: str,
        first_frame: Image.Image,
        first_frame_path: Path,
        actions_json: Path,
        prompt: str,
    ) -> Path:
        conditioning_video = sample_dir / "conditioning.mp4"
        write_single_frame_video(conditioning_video, first_frame, fps=self.runtime.fps)
        cosmos3_input = sample_dir / "cosmos3_input.json"
        write_json(
            cosmos3_input,
            {
                "action_chunk_size": CHUNK_STEPS,
                "action_path": str(actions_json.resolve()),
                "domain_name": self.runtime.domain_name,
                "fps": self.runtime.fps,
                "guidance": self.runtime.guidance,
                "image_size": self.runtime.image_size,
                "model_mode": self.runtime.model_mode,
                "name": sample_name,
                "num_frames": CHUNK_STEPS + 1,
                "num_outputs": 1,
                "num_steps": self.runtime.num_steps,
                "prompt": prompt,
                "seed": self.runtime.seed,
                "shift": self.runtime.shift,
                "video_save_quality": 8,
                "view_point": self.runtime.view_point,
                "vision_path": str(first_frame_path.resolve()),
            },
        )
        return cosmos3_input

    def _runner_payload(
        self,
        *,
        sample_name: str,
        cosmos3_input: Path,
        raw_dir: Path,
    ) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self._checkpoint_path()),
            "config_file": self.runtime.config_file,
            "experiment": self.runtime.experiment,
            "cosmos3_input": str(cosmos3_input.resolve()),
            "guardrails": self.runtime.guardrails,
            "master_port": self.runtime.master_port,
            "num_gpus": self.runtime.num_gpus,
            "output_dir": str((raw_dir / sample_name / "setup").resolve()),
            "raw_dir": str(raw_dir.resolve()),
            "sample_name": sample_name,
            "seed": self.runtime.seed,
            "use_cuda_graphs": self.runtime.cuda_graphs,
            "use_ema_weights": self.runtime.use_ema_weights,
            "use_torch_compile": self.runtime.compile,
        }

    def _command(self, *, payload_path: Path) -> list[str]:
        payload = self._read_payload(payload_path)
        command = [
            self.torchrun or self.runtime.torchrun,
            f"--nproc-per-node={self.runtime.num_gpus}",
            f"--master-port={self.runtime.master_port}",
            "-m",
            "cosmos_framework.scripts.inference",
            "--parallelism-preset=throughput",
            f"--dp-shard-size={self.runtime.num_gpus}",
            "--dp-replicate-size=1",
            "--cp-size=1",
            "--cfgp-size=1",
            "--max-num-seqs=1",
            "-i",
            str(payload["cosmos3_input"]),
            "-o",
            str(payload["raw_dir"]),
            "--checkpoint-path",
            str(payload["checkpoint_path"]),
            "--config-file",
            str(payload["config_file"]),
            "--experiment",
            str(payload["experiment"]),
            "--seed",
            str(payload["seed"]),
        ]
        command.append("--use-ema-weights" if self.runtime.use_ema_weights else "--no-use-ema-weights")
        if not self.runtime.guardrails:
            command.append("--no-guardrails")
        if not self.runtime.compile:
            command.append("--no-use-torch-compile")
        if not self.runtime.cuda_graphs:
            command.append("--no-use-cuda-graphs")
        vae_path = self._tokenizer_path()
        if vae_path is not None:
            command.extend(["--experiment-overrides", f"model.config.tokenizer.vae_path={vae_path}"])
        return command

    @staticmethod
    def _read_payload(payload_path: Path) -> dict[str, Any]:
        import json

        return json.loads(payload_path.read_text(encoding="utf-8"))

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        pythonpath = [
            str(self.project_root / "src"),
            str(self.cosmos3_root),
            str(self.cosmos3_root / "packages" / "cosmos-oss"),
            str(self.cosmos3_root.parent.parent),
        ]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(pythonpath)
        cuda_library_paths = self._cuda_library_paths()
        if env.get("LD_LIBRARY_PATH"):
            cuda_library_paths.append(env["LD_LIBRARY_PATH"])
        if cuda_library_paths:
            env["LD_LIBRARY_PATH"] = ":".join(cuda_library_paths)
        tokenizer_path = self._tokenizer_path()
        if tokenizer_path is not None:
            env.setdefault("COSMOS_WAN2PT1_VAE_PATH", str(tokenizer_path))
        env["BASE_CHECKPOINT_PATH"] = str(self._checkpoint_path())
        env["COSMOS3_H_SURGICAL_INFERENCE_COMPAT"] = "1"
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("WANDB_SILENT", "true")
        return env

    def _checkpoint_path(self) -> Path:
        checkpoint = self._local_checkpoint_root()
        if checkpoint.is_file():
            return checkpoint.resolve()
        if not checkpoint.is_dir():
            raise EndpointError(f"checkpoint not found: {checkpoint}")
        if (checkpoint / ".metadata").is_file() or (checkpoint / "model" / ".metadata").is_file():
            return checkpoint.resolve()

        checkpoint_roots = [
            checkpoint / "checkpoints",
        ]
        for checkpoint_root in checkpoint_roots:
            latest = checkpoint_root / "latest_checkpoint.txt"
            if not latest.is_file():
                continue
            iteration = latest.read_text(encoding="utf-8").strip()
            candidate = checkpoint_root / iteration
            if candidate.is_dir():
                return candidate.resolve()

        raise EndpointError(f"could not resolve a Cosmos 3 checkpoint under {checkpoint}")

    def _local_checkpoint_root(self) -> Path:
        if self.checkpoint is not None:
            return self.checkpoint
        return Path(
            snapshot_download(
                repo_id=self.runtime.hf_repo_id,
                revision=self.runtime.hf_revision,
                allow_patterns=[
                    "README.md",
                    "checkpoint_manifest.json",
                    "config.yaml",
                    "checkpoints/latest_checkpoint.txt",
                    "checkpoints/iter_*/model/*",
                ],
            )
        )

    def _tokenizer_path(self) -> Path | None:
        candidates = [
            Path(os.environ["WAN_VAE_PATH"]) if os.environ.get("WAN_VAE_PATH") else None,
            self.project_root / "artifacts" / "checkpoints" / "wan22_vae" / "Wan2.2_VAE.pth",
            self.cosmos3_root.parent / "checkpoints" / "wan22_vae" / "Wan2.2_VAE.pth",
            Path("/weights/Wan2.2_VAE.pth"),
        ]
        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                return candidate.resolve()
        return None

    @staticmethod
    def _cuda_library_paths() -> list[str]:
        nvidia_root = (
            Path(sys.prefix)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "nvidia"
        )
        if not nvidia_root.is_dir():
            return []
        paths = [path for path in sorted(nvidia_root.glob("*/lib")) if path.is_dir()]
        nvvm_lib64 = nvidia_root / "cuda_nvcc" / "nvvm" / "lib64"
        if nvvm_lib64.is_dir():
            paths.append(nvvm_lib64)
        transformer_engine_wheel_lib = nvidia_root.parent / "transformer_engine" / "wheel_lib"
        if transformer_engine_wheel_lib.is_dir():
            paths.append(transformer_engine_wheel_lib)
        return [str(path) for path in paths]

    @staticmethod
    def _find_video(raw_dir: Path, sample_name: str) -> Path:
        for suffix in ("_single_chunk.mp4", "_chunk.mp4", ".mp4"):
            direct_output = raw_dir / f"{sample_name}{suffix}"
            if direct_output.is_file():
                return direct_output
        direct = raw_dir / sample_name / "vision.mp4"
        if direct.is_file():
            return direct
        matches = list(raw_dir.glob(f"**/{sample_name}/vision.*"))
        for match in matches:
            if match.suffix.lower() == ".mp4":
                return match
        raise EndpointError(f"generated video not found under {raw_dir}")

    def command_text(self, *, payload_path: Path) -> str:
        return " ".join(shlex.quote(part) for part in self._command(payload_path=payload_path))


def nvidia_gpu_visible() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    result = subprocess.run(
        [nvidia_smi, "-L"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.returncode == 0 and "GPU" in result.stdout
