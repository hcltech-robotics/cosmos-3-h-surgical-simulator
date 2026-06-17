"""WebRTC server for browser-controlled inference."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from .action import ToolCommand, chunk_from_rows, validate_action_chunk
from .config import RuntimeConfig
from .endpoint import SurgicalSimulatorEndpoint, nvidia_gpu_visible
from .io import InputError, load_first_frame, read_video_frames
from .keyboard import KeyboardConfig, KeyboardPacket


class VideoFrameSource:
    def __init__(self, *, buffer_frames: int = 48) -> None:
        self._frames: list[np.ndarray] = []
        self._subscribers: set[asyncio.Queue[np.ndarray]] = set()
        self._buffer_frames = max(1, buffer_frames)

    def subscribe(self) -> asyncio.Queue[np.ndarray]:
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=self._buffer_frames)
        for frame in self._frames:
            self._put_ordered(queue, frame)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[np.ndarray]) -> None:
        self._subscribers.discard(queue)

    @property
    def ready(self) -> bool:
        return bool(self._frames)

    def publish(self, frames: list[np.ndarray]) -> None:
        if not frames:
            raise InputError("generated video contains no frames")
        self._frames = [np.asarray(frame, dtype=np.uint8) for frame in frames]
        for queue in list(self._subscribers):
            for frame in self._frames:
                self._put_latest(queue, frame)

    def publish_stream(self, frames: list[np.ndarray]) -> None:
        if not frames:
            raise InputError("generated stream contains no frames")
        stream_frames = [np.asarray(frame, dtype=np.uint8) for frame in frames]
        self._frames = [stream_frames[-1]]
        for queue in list(self._subscribers):
            self._clear_queue(queue)
            for frame in stream_frames:
                self._put_ordered(queue, frame)

    def publish_image(self, frame: np.ndarray) -> None:
        self.publish([frame])

    @staticmethod
    def _put_latest(queue: asyncio.Queue[np.ndarray], frame: np.ndarray) -> None:
        VideoFrameSource._clear_queue(queue)
        queue.put_nowait(frame)

    @staticmethod
    def _put_ordered(queue: asyncio.Queue[np.ndarray], frame: np.ndarray) -> None:
        while queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        queue.put_nowait(frame)

    @staticmethod
    def _clear_queue(queue: asyncio.Queue[np.ndarray]) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break


@dataclass
class ServerState:
    runtime: RuntimeConfig
    keyboard: KeyboardConfig
    endpoint: SurgicalSimulatorEndpoint
    first_frame_path: Path
    prompt: str
    video_source: VideoFrameSource = field(default_factory=VideoFrameSource)
    started_at: float = field(default_factory=time.time)
    packets_received: int = 0
    chunks_submitted: int = 0
    last_error: str | None = None
    last_action_chunk: list[list[float]] | None = None
    latest_keys: set[str] = field(default_factory=set)
    latest_seq: int | None = None
    left_control_state: ToolCommand = field(default_factory=ToolCommand)
    right_control_state: ToolCommand = field(default_factory=ToolCommand)
    first_action_received: asyncio.Event = field(default_factory=asyncio.Event)
    generation_task: asyncio.Task[None] | None = None
    generation_epoch: int = 0
    stop_requested: bool = False
    generation_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cosmos3-surgical-generation",
        )
    )
    peer_connections: set[Any] = field(default_factory=set)
    data_channels: set[Any] = field(default_factory=set)
    last_inference_fps: float | None = None
    last_inference_duration_s: float | None = None
    last_model_duration_s: float | None = None
    last_decode_duration_s: float | None = None
    last_generated_frames: int | None = None

    def status_payload(self) -> dict[str, Any]:
        return {
            "ok": self.last_error is None,
            "name": self.runtime.name,
            "profile": self.runtime.profile,
            "uptime_s": round(time.time() - self.started_at, 3),
            "gpu_visible": nvidia_gpu_visible(),
            "video_ready": self.video_source.ready,
            "packets_received": self.packets_received,
            "chunks_submitted": self.chunks_submitted,
            "generation_running": self.generation_task is not None and not self.generation_task.done(),
            "generation_paused": not self._has_active_generation_controls(),
            "active_keys": sorted(self.latest_keys),
            "last_inference_fps": self.last_inference_fps,
            "last_inference_duration_s": self.last_inference_duration_s,
            "last_model_duration_s": self.last_model_duration_s,
            "last_decode_duration_s": self.last_decode_duration_s,
            "last_generated_frames": self.last_generated_frames,
            "performance": {
                "min_inference_fps": self.runtime.min_inference_fps,
                "fps": self.runtime.fps,
                "width": self.runtime.width,
                "height": self.runtime.height,
            },
            "webrtc": {
                "turn_port": self.runtime.browser_turn_port,
                "turn_username": self.runtime.turn_username,
                "turn_configured": bool(self.runtime.turn_server),
                "ice_transport_policy": self.runtime.browser_ice_transport_policy,
            },
            "conditioning_seed": str(self.first_frame_path),
            "keyboard": self.keyboard.browser_payload(),
            "last_error": self.last_error,
        }

    def ingest_packet_text(self, payload_text: str | bytes) -> dict[str, Any]:
        payload = json.loads(payload_text)
        return self.ingest_payload(payload)

    def ingest_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("packet must be a JSON object")
        packet_type = payload.get("type")
        if packet_type == "keyboard_state":
            return self._ingest_keyboard(payload)
        if packet_type == "action":
            return self._ingest_action_event(payload)
        if packet_type == "heartbeat":
            return {"ok": True, "type": "heartbeat", "submit_generation": False}
        if packet_type == "action_chunk":
            return self._ingest_action_chunk(payload)
        if packet_type == "reset_original":
            return self.reset_to_original()
        raise ValueError(f"unknown packet type {packet_type!r}")

    def start_generation(self) -> bool:
        if self.last_action_chunk is None:
            return False
        if self.generation_task is not None and not self.generation_task.done():
            return False
        self.chunks_submitted += 1
        chunk = [row[:] for row in self.last_action_chunk]
        sample_name = f"chunk_{self.chunks_submitted:06d}"
        self.generation_task = asyncio.create_task(
            self._generate_and_publish(action_chunk=chunk, sample_name=sample_name)
        )
        return True

    async def stop_generation_loop(self, *, cancel: bool = True) -> None:
        self.stop_requested = True
        self.first_action_received.set()
        task = self.generation_task
        if task is None or task.done():
            return
        if cancel:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def close(self) -> None:
        await self.stop_generation_loop()
        self.generation_executor.shutdown(wait=False, cancel_futures=True)

    def broadcast(self, payload: dict[str, Any]) -> None:
        message = json.dumps(payload, sort_keys=True)
        for channel in list(self.data_channels):
            try:
                if getattr(channel, "readyState", None) == "open":
                    channel.send(message)
            except Exception:
                self.data_channels.discard(channel)

    def _ingest_keyboard(self, payload: dict[str, Any]) -> dict[str, Any]:
        packet = KeyboardPacket.from_payload(payload)
        self.packets_received += 1
        self.latest_keys = set(packet.keys)
        self.latest_seq = packet.seq
        if self.keyboard.recenter_key in self.latest_keys:
            self._reset_control_state()
        chunk = self._preview_keyboard_chunk()
        self._set_latest_action_chunk(chunk if self._has_active_generation_controls() else None)
        return {
            "ok": True,
            "type": "keyboard_state",
            "seq": packet.seq,
            "active_keys": sorted(packet.keys),
            "action_shape": [len(chunk), len(chunk[0])],
            "dvrk_prefix": chunk[-1][:20],
            "submit_generation": self._has_active_generation_controls(),
        }

    def _ingest_action_chunk(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("action_chunk.rows must be a list")
        chunk = chunk_from_rows(rows)
        self.packets_received += 1
        self._set_latest_action_chunk(chunk if self._chunk_has_generation_controls(chunk) else None)
        return {
            "ok": True,
            "type": "action_chunk",
            "action_shape": [len(chunk), len(chunk[0])],
            "dvrk_prefix": chunk[-1][:20],
            "submit_generation": self.last_action_chunk is not None,
        }

    def _ingest_action_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action", payload)
        if not isinstance(action, dict):
            raise ValueError("action must be an object")
        event = str(action.get("event", "")).strip().lower()
        if event == "step":
            return {"ok": True, "type": "action", "submit_generation": False}
        if event not in {"keydown", "keyup"}:
            raise ValueError("action.event must be keydown or keyup")
        key = str(action.get("key", "")).strip()
        if not key:
            raise ValueError("action.key must be a non-empty key code")
        if event == "keydown":
            self.latest_keys.add(key)
        else:
            self.latest_keys.discard(key)
        if self.keyboard.recenter_key in self.latest_keys:
            self._reset_control_state()

        self.packets_received += 1
        seq = payload.get("seq")
        self.latest_seq = seq if isinstance(seq, int) else self.latest_seq
        chunk = self._preview_keyboard_chunk()
        self._set_latest_action_chunk(chunk if self._has_active_generation_controls() else None)
        return {
            "ok": True,
            "type": "action",
            "seq": seq,
            "event": event,
            "key": key,
            "active_keys": sorted(self.latest_keys),
            "action_shape": [len(chunk), len(chunk[0])],
            "dvrk_prefix": chunk[-1][:20],
            "submit_generation": self._has_active_generation_controls(),
        }

    def reset_to_original(self) -> dict[str, Any]:
        self.generation_epoch += 1
        self.stop_requested = False
        frame = load_first_frame(
            self.first_frame_path,
            width=self.runtime.width,
            height=self.runtime.height,
        )
        self.video_source.publish_image(np.asarray(frame.image, dtype=np.uint8))
        self.last_action_chunk = None
        self.latest_keys = set()
        self.latest_seq = None
        self._reset_control_state()
        self.first_action_received.clear()
        self.last_inference_fps = None
        self.last_inference_duration_s = None
        self.last_model_duration_s = None
        self.last_decode_duration_s = None
        self.last_generated_frames = None
        self.last_error = None
        return {
            "ok": True,
            "type": "reset_original",
            "conditioning_seed": str(self.first_frame_path),
            "submit_generation": False,
        }

    async def hard_reset_to_original(self) -> dict[str, Any]:
        self.generation_epoch += 1
        await self.stop_generation_loop(cancel=False)
        result = self.reset_to_original()
        result["generation_stopped"] = True
        return result

    def _current_action_chunk(self) -> list[list[float]]:
        if self._has_active_generation_controls():
            chunk, left, right = self.keyboard.trajectory_chunk_from_keys(
                self.latest_keys,
                left_origin=self.left_control_state,
                right_origin=self.right_control_state,
                fps=self.runtime.fps,
            )
            self.left_control_state = left
            self.right_control_state = right
            self.last_action_chunk = [row[:] for row in chunk]
            return chunk
        if self.last_action_chunk is not None:
            return [row[:] for row in self.last_action_chunk]
        return self.keyboard.chunk_from_keys(set())

    async def _wait_for_active_action(self) -> bool:
        while not self.stop_requested:
            if self._has_active_generation_controls() and self.last_action_chunk is not None:
                return True
            self.first_action_received.clear()
            await self.first_action_received.wait()
        return False

    def _set_latest_action_chunk(self, chunk: list[list[float]] | None) -> None:
        self.last_action_chunk = [row[:] for row in chunk] if chunk is not None else None
        if self.last_action_chunk is None:
            self.first_action_received.clear()
        else:
            self.first_action_received.set()

    def _has_active_generation_controls(self) -> bool:
        return any(key in self.keyboard.bindings for key in self.latest_keys)

    def _chunk_has_generation_controls(self, chunk: list[list[float]]) -> bool:
        neutral = self.keyboard.chunk_from_keys(set())
        return any(
            abs(float(value) - float(neutral_value)) > 1e-8
            for row, neutral_row in zip(chunk, neutral, strict=True)
            for value, neutral_value in zip(row, neutral_row, strict=True)
        )

    def _preview_keyboard_chunk(self) -> list[list[float]]:
        chunk, _left, _right = self.keyboard.trajectory_chunk_from_keys(
            self.latest_keys,
            left_origin=self.left_control_state,
            right_origin=self.right_control_state,
            fps=self.runtime.fps,
        )
        return chunk

    def _reset_control_state(self) -> None:
        self.left_control_state = ToolCommand()
        self.right_control_state = ToolCommand()

    async def _generate_and_publish(
        self,
        *,
        action_chunk: list[list[float]],
        sample_name: str,
    ) -> None:
        validate_action_chunk(action_chunk)
        try:
            frame = load_first_frame(
                self.first_frame_path,
                width=self.runtime.width,
                height=self.runtime.height,
            )

            loop = asyncio.get_running_loop()

            def run_endpoint():
                return self.endpoint.generate(
                    first_frame=frame.image,
                    action_chunk=action_chunk,
                    prompt=self.prompt,
                    sample_name=sample_name,
                )

            endpoint_result = await loop.run_in_executor(self.generation_executor, run_endpoint)
            frames = await loop.run_in_executor(None, read_video_frames, endpoint_result.video_path)
            self.video_source.publish_stream(frames[1:] or frames)
            self.last_inference_duration_s = endpoint_result.duration_s
            self.last_inference_fps = len(frames) / endpoint_result.duration_s if endpoint_result.duration_s > 0 else None
            self.last_generated_frames = len(frames)
            video_path = str(endpoint_result.video_path)
            self.last_error = None
            self.broadcast(
                {
                    "ok": True,
                    "type": "chunk_done",
                    "video_path": video_path,
                    "frames": self.last_generated_frames,
                    "inference_fps": self.last_inference_fps,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            message = f"{type(exc).__name__}: {exc}"
            self.last_error = message
            self.broadcast({"ok": False, "type": "error", "error": message})


async def create_app(state: ServerState):
    from aiohttp import web

    async def index(_request: web.Request) -> web.Response:
        return web.Response(text=_asset_text("index.html"), content_type="text/html")

    async def video(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(_web_asset_path("video.mp4"))

    async def status(_request: web.Request) -> web.Response:
        return web.json_response(state.status_payload())

    async def packet(request: web.Request) -> web.Response:
        try:
            payload = json.loads(await request.text())
            result = (
                await state.hard_reset_to_original()
                if isinstance(payload, dict) and payload.get("type") == "reset_original"
                else state.ingest_payload(payload)
            )
            result["generation_started"] = (
                state.start_generation() if result.get("submit_generation", False) else False
            )
            return web.json_response(result)
        except (ValueError, json.JSONDecodeError) as exc:
            state.last_error = str(exc)
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    async def offer(request: web.Request) -> web.Response:
        try:
            from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
        except ImportError:
            return web.json_response({"ok": False, "error": "aiortc is required"}, status=503)

        payload = await request.json()
        ice_servers = []
        if state.runtime.turn_server:
            ice_servers.append(
                RTCIceServer(
                    urls=state.runtime.turn_server,
                    username=state.runtime.turn_username or None,
                    credential=state.runtime.turn_credential or None,
                )
            )
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
        state.peer_connections.add(pc)
        pc.addTrack(_make_video_track(state.video_source, fps=state.runtime.fps))

        @pc.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            state.data_channels.add(channel)
            @channel.on("message")
            def on_message(message: str | bytes) -> None:
                asyncio.create_task(_handle_datachannel_message(state, channel, message))

            @channel.on("close")
            def on_close() -> None:
                state.data_channels.discard(channel)
                if not state.data_channels:
                    asyncio.create_task(state.stop_generation_loop())

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if pc.connectionState in {"failed", "disconnected", "closed"}:
                await pc.close()
                state.peer_connections.discard(pc)
                if not state.peer_connections:
                    await state.stop_generation_loop()

        await pc.setRemoteDescription(RTCSessionDescription(sdp=payload["sdp"], type=payload["type"]))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_for_ice_gathering_complete(pc)
        description = pc.localDescription
        if description is None:
            return web.json_response({"error": "local description was not created"}, status=500)
        return web.json_response({"sdp": description.sdp, "type": description.type})

    async def on_shutdown(_app: web.Application) -> None:
        await asyncio.gather(*(pc.close() for pc in list(state.peer_connections)), return_exceptions=True)
        await state.close()
        state.peer_connections.clear()
        state.data_channels.clear()

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/request_session", index)
    app.router.add_get("/video.mp4", video)
    app.router.add_static("/assets", _web_asset_path("assets"))
    app.router.add_static("/static", _web_asset_path("static"))
    app.router.add_get("/status", status)
    app.router.add_post("/packet", packet)
    app.router.add_post("/offer", offer)
    app.router.add_post("/api/webrtc/offer", offer)
    app.on_shutdown.append(on_shutdown)
    return app


async def _wait_for_ice_gathering_complete(pc: Any, *, timeout_s: float = 5.0) -> None:
    if getattr(pc, "iceGatheringState", None) == "complete":
        return
    loop = asyncio.get_running_loop()
    done = loop.create_future()

    @pc.on("icegatheringstatechange")
    def on_icegatheringstatechange() -> None:
        if pc.iceGatheringState == "complete" and not done.done():
            done.set_result(None)

    try:
        await asyncio.wait_for(done, timeout=timeout_s)
    except TimeoutError:
        return


async def _handle_datachannel_message(state: ServerState, channel: Any, message: str | bytes) -> None:
    try:
        payload = json.loads(message)
        result = (
            await state.hard_reset_to_original()
            if isinstance(payload, dict) and payload.get("type") == "reset_original"
            else state.ingest_payload(payload)
        )
        result["generation_started"] = (
            state.start_generation() if result.get("submit_generation", False) else False
        )
        channel.send(json.dumps(result, sort_keys=True))
    except (ValueError, json.JSONDecodeError) as exc:
        channel.send(json.dumps({"ok": False, "type": "error", "error": str(exc)}, sort_keys=True))


def _make_video_track(source: VideoFrameSource, *, fps: int):
    from aiortc import VideoStreamTrack
    from av import VideoFrame

    class GeneratedVideoTrack(VideoStreamTrack):
        kind = "video"

        def __init__(self) -> None:
            super().__init__()
            self.queue = source.subscribe()

        async def recv(self):
            pts, time_base = await self.next_timestamp()
            frame_arr = await self.queue.get()
            frame = VideoFrame.from_ndarray(frame_arr, format="rgb24")
            frame.pts = pts
            frame.time_base = time_base
            return frame

        def stop(self) -> None:
            source.unsubscribe(self.queue)
            super().stop()

    return GeneratedVideoTrack()


def _asset_text(name: str) -> str:
    return _web_asset_path(name).read_text(encoding="utf-8")


def _web_asset_path(name: str) -> Path:
    return Path(__file__).with_name("web") / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cosmos3-super-webrtc.toml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cosmos3-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--first-frame", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/webrtc"))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--prompt")
    return parser


async def run_server(args: argparse.Namespace) -> None:
    from aiohttp import web

    runtime = RuntimeConfig.from_toml(args.config)
    if args.host:
        runtime = RuntimeConfig(**{**runtime.__dict__, "host": args.host})
    if args.port:
        runtime = RuntimeConfig(**{**runtime.__dict__, "port": args.port})
    keyboard = KeyboardConfig.from_toml(args.config)
    endpoint = SurgicalSimulatorEndpoint(
        runtime=runtime,
        checkpoint=args.checkpoint,
        cosmos3_root=args.cosmos3_root,
        project_root=args.project_root,
        output_root=args.output_root,
    )
    state = ServerState(
        runtime=runtime,
        keyboard=keyboard,
        endpoint=endpoint,
        first_frame_path=args.first_frame,
        prompt=args.prompt or runtime.prompt,
    )
    state.reset_to_original()
    app = await create_app(state)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, runtime.host, runtime.port)
    await site.start()
    print(json.dumps({"status": "listening", "host": runtime.host, "port": runtime.port}))
    while True:
        await asyncio.sleep(3600)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
