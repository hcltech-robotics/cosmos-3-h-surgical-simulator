"""Batch inference command."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import RuntimeConfig
from .endpoint import EndpointError, SurgicalSimulatorEndpoint
from .io import InputError, load_action_chunk, load_first_frame, preconvert_action_chunk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cosmos3-super-webrtc.toml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cosmos3-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--actions-path", type=Path, required=True)
    parser.add_argument("--preconvert-actions-to", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--output-root", type=Path, default=Path("runs/batch"))
    parser.add_argument("--sample-name", default="cosmos-3-ac-surgical")
    return parser


def run(args: argparse.Namespace) -> int:
    runtime = RuntimeConfig.from_toml(args.config)
    frame = load_first_frame(args.image_path, width=runtime.width, height=runtime.height)
    if args.preconvert_actions_to is not None:
        preconvert_action_chunk(args.actions_path, args.preconvert_actions_to)
    chunk = load_action_chunk(args.actions_path)
    endpoint = SurgicalSimulatorEndpoint(
        runtime=runtime,
        checkpoint=args.checkpoint,
        cosmos3_root=args.cosmos3_root,
        project_root=args.project_root,
        output_root=args.output_root,
    )
    endpoint.generate(
        first_frame=frame.image,
        action_chunk=chunk,
        prompt=args.prompt or runtime.prompt,
        sample_name=args.sample_name,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (EndpointError, InputError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
