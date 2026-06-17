from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import numpy as np
import pytest

from flashdreams_cosmos3_h_surgical.action import (
    ACTION_DIM,
    CHUNK_STEPS,
    DVRK_NATIVE_DIM,
    PADDING_SLICE,
    ToolCommand,
    build_action_vector,
    repeat_action,
    validate_action_chunk,
)
from flashdreams_cosmos3_h_surgical.config import RuntimeConfig
from flashdreams_cosmos3_h_surgical.endpoint import SurgicalSimulatorEndpoint
from flashdreams_cosmos3_h_surgical.io import load_action_chunk, preconvert_action_chunk
from flashdreams_cosmos3_h_surgical.keyboard import KeyboardConfig
from flashdreams_cosmos3_h_surgical.server import ServerState, VideoFrameSource


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cosmos3-super-webrtc.toml"


def test_action_vector_contract() -> None:
    vector = build_action_vector(
        ToolCommand(dx=0.01, dy=0.02, dz=0.03, gripper=0.4),
        ToolCommand(dx=-0.01, dy=-0.02, dz=-0.03, gripper=0.8),
    )
    assert len(vector) == ACTION_DIM
    assert vector[0:3] == pytest.approx([0.01, 0.02, 0.03])
    assert vector[10:13] == pytest.approx([-0.01, -0.02, -0.03])
    assert vector[PADDING_SLICE] == pytest.approx([0.0] * (ACTION_DIM - DVRK_NATIVE_DIM))
    chunk = repeat_action(vector)
    validate_action_chunk(chunk)
    assert len(chunk) == CHUNK_STEPS


def test_keyboard_config_maps_browser_codes_to_12x44_chunk() -> None:
    keyboard = KeyboardConfig.from_toml(CONFIG)
    chunk = keyboard.chunk_from_keys({"KeyW", "KeyD", "ArrowUp", "PageUp", "ShiftLeft"})
    validate_action_chunk(chunk)
    assert len(chunk) == CHUNK_STEPS
    assert len(chunk[-1]) == ACTION_DIM
    assert chunk[-1][0] > 0
    assert chunk[-1][1] > 0
    assert chunk[-1][10] == pytest.approx(0.0)
    assert chunk[-1][11] > 0
    assert chunk[-1][12] > 0


def test_keyboard_trajectory_integrates_held_keys_across_chunks() -> None:
    keyboard = KeyboardConfig.from_toml(CONFIG)
    first, left, right = keyboard.trajectory_chunk_from_keys(
        {"KeyD", "ArrowRight"},
        left_origin=ToolCommand(),
        right_origin=ToolCommand(),
        fps=18,
    )
    second, next_left, next_right = keyboard.trajectory_chunk_from_keys(
        {"KeyD", "ArrowRight"},
        left_origin=left,
        right_origin=right,
        fps=18,
    )
    validate_action_chunk(first)
    validate_action_chunk(second)
    assert first[-1][0] > first[0][0]
    assert second[-1][0] > first[-1][0]
    assert second[-1][10] > first[-1][10]
    assert next_left.dx > left.dx
    assert next_right.dx > right.dx


def test_json_actions_load_and_preconvert_to_npy(tmp_path: Path) -> None:
    chunk = repeat_action(build_action_vector(ToolCommand(dx=0.01), ToolCommand()))
    json_path = tmp_path / "actions.json"
    json_path.write_text(json.dumps({"actions": chunk}) + "\n")
    npy_path = tmp_path / "actions.npy"

    np.testing.assert_allclose(load_action_chunk(json_path), chunk)
    written = preconvert_action_chunk(json_path, npy_path)

    assert written == npy_path
    assert np.asarray(load_action_chunk(npy_path)).shape == (CHUNK_STEPS, ACTION_DIM)


def test_endpoint_writes_cosmos3_input_and_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RuntimeConfig.from_toml(CONFIG)
    checkpoint = tmp_path / "checkpoint"
    checkpoint_model = checkpoint / "checkpoints" / "iter_000000060" / "model"
    checkpoint_model.mkdir(parents=True)
    (checkpoint / "checkpoints" / "latest_checkpoint.txt").write_text("iter_000000060\n")
    (checkpoint_model / ".metadata").write_text("{}\n")
    cosmos3_root = tmp_path / "cosmos3"
    cosmos3_root.mkdir()
    project_root = tmp_path / "project"
    project_root.mkdir()
    endpoint = SurgicalSimulatorEndpoint(
        runtime=runtime,
        checkpoint=checkpoint,
        cosmos3_root=cosmos3_root,
        project_root=project_root,
        output_root=tmp_path / "runs",
        torchrun="torchrun",
    )
    chunk = repeat_action(build_action_vector(ToolCommand(dx=0.01), ToolCommand()))

    def captured_run(command, cwd, env, check):
        assert command[0] == "torchrun"
        assert command[3:5] == ["-m", "cosmos_framework.scripts.inference"]
        assert "--checkpoint-path" in command
        assert "--no-use-ema-weights" in command
        assert "--no-guardrails" in command
        assert "--no-use-torch-compile" in command
        assert "--no-use-cuda-graphs" in command
        assert cwd == cosmos3_root
        assert env["COSMOS3_H_SURGICAL_INFERENCE_COMPAT"] == "1"
        assert str(cosmos3_root / "packages" / "cosmos-oss") in env["PYTHONPATH"]
        assert env["BASE_CHECKPOINT_PATH"].endswith("checkpoints/iter_000000060")
        video_dir = tmp_path / "runs" / "cosmos3_raw" / "sample_001"
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "vision.mp4").write_bytes(b"mp4")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("flashdreams_cosmos3_h_surgical.endpoint.subprocess.run", captured_run)
    result = endpoint.generate(
        first_frame=Image.new("RGB", (runtime.width, runtime.height)),
        action_chunk=chunk,
        prompt=runtime.prompt,
        sample_name="sample_001",
    )
    sample_dir = tmp_path / "runs" / "samples" / "sample_001"
    payload = json.loads((sample_dir / "runner_payload.json").read_text())
    cosmos3_input = json.loads((sample_dir / "cosmos3_input.json").read_text())
    assert payload["sample_name"] == "sample_001"
    assert payload["experiment"] == "cosmos3_super_openh_surgical_lora"
    assert payload["checkpoint_path"].endswith("checkpoints/iter_000000060")
    assert cosmos3_input["model_mode"] == "forward_dynamics"
    assert cosmos3_input["domain_name"] == "open_h_surgical_sim"
    assert cosmos3_input["action_chunk_size"] == CHUNK_STEPS
    assert Path(cosmos3_input["action_path"]).name == "actions.json"
    assert Path(cosmos3_input["vision_path"]).name == "first_frame.png"
    assert result.video_path.name == "vision.mp4"


def test_server_ingests_keyboard_payload(tmp_path: Path) -> None:
    runtime = RuntimeConfig.from_toml(CONFIG)
    frame_path = tmp_path / "first.png"
    Image.new("RGB", (runtime.width, runtime.height)).save(frame_path)
    state = ServerState(
        runtime=runtime,
        keyboard=KeyboardConfig.from_toml(CONFIG),
        endpoint=SurgicalSimulatorEndpoint(
            runtime=runtime,
            checkpoint=tmp_path,
            cosmos3_root=tmp_path,
            project_root=tmp_path,
            output_root=tmp_path / "runs",
        ),
        first_frame_path=frame_path,
        prompt=runtime.prompt,
    )
    result = state.ingest_packet_text(
        json.dumps(
            {
                "type": "keyboard_state",
                "seq": 1,
                "timestamp": 1.0,
                "keys": ["KeyW", "KeyD"],
            }
        )
    )
    assert result["ok"] is True
    assert result["action_shape"] == [CHUNK_STEPS, ACTION_DIM]
    assert len(result["dvrk_prefix"]) == DVRK_NATIVE_DIM
    status = state.status_payload()
    assert status["keyboard"]["repeat_hz"] == pytest.approx(30.0)
    assert status["conditioning_seed"] == str(frame_path)
    assert status["webrtc"]["turn_configured"] is False
    assert "turn_credential" not in status["webrtc"]


def test_server_pauses_generation_when_keyboard_controls_release(tmp_path: Path) -> None:
    runtime = RuntimeConfig.from_toml(CONFIG)
    frame_path = tmp_path / "first.png"
    Image.new("RGB", (runtime.width, runtime.height)).save(frame_path)
    state = ServerState(
        runtime=runtime,
        keyboard=KeyboardConfig.from_toml(CONFIG),
        endpoint=SurgicalSimulatorEndpoint(
            runtime=runtime,
            checkpoint=tmp_path,
            cosmos3_root=tmp_path,
            project_root=tmp_path,
            output_root=tmp_path / "runs",
        ),
        first_frame_path=frame_path,
        prompt=runtime.prompt,
    )
    down = state.ingest_packet_text(
        json.dumps(
            {
                "type": "action",
                "seq": 1,
                "timestamp": 1.0,
                "action": {"event": "keydown", "key": "KeyD"},
            }
        )
    )
    assert down["active_keys"] == ["KeyD"]
    assert state.last_action_chunk is not None
    assert state.first_action_received.is_set()
    assert state.status_payload()["generation_paused"] is False

    up = state.ingest_packet_text(
        json.dumps(
            {
                "type": "action",
                "seq": 2,
                "timestamp": 2.0,
                "action": {"event": "keyup", "key": "KeyD"},
            }
        )
    )
    assert up["active_keys"] == []
    assert state.last_action_chunk is None
    assert not state.first_action_received.is_set()
    assert state.status_payload()["generation_paused"] is True


def test_server_reset_publishes_conditioning_seed(tmp_path: Path) -> None:
    runtime = RuntimeConfig.from_toml(CONFIG)
    frame_path = tmp_path / "first.png"
    Image.new("RGB", (runtime.width, runtime.height), color=(10, 20, 30)).save(frame_path)
    state = ServerState(
        runtime=runtime,
        keyboard=KeyboardConfig.from_toml(CONFIG),
        endpoint=SurgicalSimulatorEndpoint(
            runtime=runtime,
            checkpoint=tmp_path,
            cosmos3_root=tmp_path,
            project_root=tmp_path,
            output_root=tmp_path / "runs",
        ),
        first_frame_path=frame_path,
        prompt=runtime.prompt,
    )
    state.ingest_packet_text(
        json.dumps(
            {
                "type": "action",
                "seq": 1,
                "timestamp": 1.0,
                "action": {"event": "keydown", "key": "KeyD"},
            }
        )
    )
    assert state.last_action_chunk is not None
    result = state.ingest_packet_text(json.dumps({"type": "reset_original"}))
    assert result["ok"] is True
    assert result["submit_generation"] is False
    assert result["conditioning_seed"] == str(frame_path)
    assert state.latest_keys == set()
    assert state.last_action_chunk is None
    assert state.status_payload()["generation_paused"] is True
    assert state.video_source.ready is True


def test_video_source_replays_cached_frames() -> None:
    source = VideoFrameSource()
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    source.publish([frame])
    queue = source.subscribe()
    assert queue.get_nowait().shape == (4, 6, 3)


def test_video_source_keeps_latest_frame_for_live_clients() -> None:
    source = VideoFrameSource()
    queue = source.subscribe()
    old_frame = np.zeros((4, 6, 3), dtype=np.uint8)
    new_frame = np.full((4, 6, 3), 255, dtype=np.uint8)
    source.publish_stream([old_frame])
    source.publish_stream([new_frame])
    assert queue.qsize() == 1
    assert queue.get_nowait().mean() == pytest.approx(255.0)


def test_video_source_preserves_stream_frame_order() -> None:
    source = VideoFrameSource()
    queue = source.subscribe()
    frames = [
        np.full((4, 6, 3), value, dtype=np.uint8)
        for value in (16, 64, 128)
    ]
    source.publish_stream(frames)
    assert queue.qsize() == 3
    assert [queue.get_nowait().mean() for _ in frames] == pytest.approx([16.0, 64.0, 128.0])
