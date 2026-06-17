"""Browser keyboard to surgical simulator action mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .action import CHUNK_STEPS, ControlFlags, ToolCommand, build_action_vector, repeat_action

AXES = frozenset({"dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"})


@dataclass(frozen=True)
class AxisBinding:
    tool: str
    axis: str
    sign: float

    @classmethod
    def parse(cls, tool: str, value: str) -> "AxisBinding":
        if ":" not in value:
            raise ValueError(f"binding {value!r} must be '<axis>:<sign>'")
        axis, sign_text = value.split(":", 1)
        axis = axis.strip()
        if axis not in AXES:
            raise ValueError(f"unknown keyboard axis {axis!r}")
        return cls(tool=tool, axis=axis, sign=float(sign_text))


@dataclass(frozen=True)
class KeyboardConfig:
    repeat_hz: float = 30.0
    translation_step_m: float = 0.003
    rotation_step_rad: float = 0.025
    gripper_step: float = 0.04
    max_translation_m: float = 0.09
    max_rotation_rad: float = 0.8
    max_gripper: float = 1.0
    fine_scale: float = 0.25
    boost_scale: float = 2.0
    fine_key: str = "ShiftLeft"
    boost_key: str = "ShiftRight"
    hold_key: str = "Space"
    recenter_key: str = "Enter"
    left_clutch_key: str = "Digit1"
    right_clutch_key: str = "Digit2"
    bindings: dict[str, AxisBinding] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: Path) -> "KeyboardConfig":
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        keyboard = payload.get("keyboard", {})
        raw_bindings = keyboard.get("bindings", {})
        bindings: dict[str, AxisBinding] = {}
        for tool in ("left", "right"):
            for key_code, value in raw_bindings.get(tool, {}).items():
                bindings[str(key_code)] = AxisBinding.parse(tool, str(value))
        return cls(
            repeat_hz=float(keyboard.get("repeat_hz", cls.repeat_hz)),
            translation_step_m=float(keyboard.get("translation_step_m", cls.translation_step_m)),
            rotation_step_rad=float(keyboard.get("rotation_step_rad", cls.rotation_step_rad)),
            gripper_step=float(keyboard.get("gripper_step", cls.gripper_step)),
            max_translation_m=float(keyboard.get("max_translation_m", cls.max_translation_m)),
            max_rotation_rad=float(keyboard.get("max_rotation_rad", cls.max_rotation_rad)),
            max_gripper=float(keyboard.get("max_gripper", cls.max_gripper)),
            fine_scale=float(keyboard.get("fine_scale", cls.fine_scale)),
            boost_scale=float(keyboard.get("boost_scale", cls.boost_scale)),
            fine_key=str(keyboard.get("fine_key", cls.fine_key)),
            boost_key=str(keyboard.get("boost_key", cls.boost_key)),
            hold_key=str(keyboard.get("hold_key", cls.hold_key)),
            recenter_key=str(keyboard.get("recenter_key", cls.recenter_key)),
            left_clutch_key=str(keyboard.get("left_clutch_key", cls.left_clutch_key)),
            right_clutch_key=str(keyboard.get("right_clutch_key", cls.right_clutch_key)),
            bindings=bindings,
        )

    def command_from_keys(self, keys: set[str]) -> tuple[ToolCommand, ToolCommand, ControlFlags]:
        scale = 1.0
        if self.fine_key in keys:
            scale *= self.fine_scale
        if self.boost_key in keys:
            scale *= self.boost_scale

        left = ToolCommand()
        right = ToolCommand()
        for key in keys:
            binding = self.bindings.get(key)
            if binding is None:
                continue
            step = self._step_for_axis(binding.axis) * binding.sign * scale
            delta = self._axis_command(binding.axis, step)
            if binding.tool == "left":
                left = left.plus(delta)
            elif binding.tool == "right":
                right = right.plus(delta)
            else:
                raise ValueError(f"unknown tool binding {binding.tool!r}")

        flags = ControlFlags(
            hold=self.hold_key in keys,
            recenter=self.recenter_key in keys,
            left_clutch=self.left_clutch_key in keys,
            right_clutch=self.right_clutch_key in keys,
        )
        return left, right, flags

    def chunk_from_keys(self, keys: set[str]) -> list[list[float]]:
        left, right, flags = self.command_from_keys(keys)
        return repeat_action(build_action_vector(left, right, flags))

    def trajectory_chunk_from_keys(
        self,
        keys: set[str],
        *,
        left_origin: ToolCommand,
        right_origin: ToolCommand,
        fps: float,
    ) -> tuple[list[list[float]], ToolCommand, ToolCommand]:
        left_delta, right_delta, flags = self.command_from_keys(keys)
        fps = max(1.0, float(fps))
        per_frame_scale = self.repeat_hz / fps
        rows: list[list[float]] = []
        final_left = left_origin
        final_right = right_origin

        for frame_index in range(1, CHUNK_STEPS + 1):
            left = self._integrated_command(
                origin=left_origin,
                delta=left_delta,
                frame_scale=per_frame_scale * frame_index,
                frozen=flags.hold or flags.left_clutch,
            )
            right = self._integrated_command(
                origin=right_origin,
                delta=right_delta,
                frame_scale=per_frame_scale * frame_index,
                frozen=flags.hold or flags.right_clutch,
            )
            rows.append(build_action_vector(left, right, flags))
            final_left = left
            final_right = right

        return rows, final_left, final_right

    def browser_payload(self) -> dict[str, Any]:
        return {
            "repeat_hz": self.repeat_hz,
            "fine_key": self.fine_key,
            "boost_key": self.boost_key,
            "hold_key": self.hold_key,
            "recenter_key": self.recenter_key,
            "left_clutch_key": self.left_clutch_key,
            "right_clutch_key": self.right_clutch_key,
            "max_translation_m": self.max_translation_m,
            "max_rotation_rad": self.max_rotation_rad,
            "max_gripper": self.max_gripper,
            "bindings": {
                key: {"tool": binding.tool, "axis": binding.axis, "sign": binding.sign}
                for key, binding in sorted(self.bindings.items())
            },
        }

    def _step_for_axis(self, axis: str) -> float:
        if axis in {"dx", "dy", "dz"}:
            return self.translation_step_m
        if axis in {"droll", "dpitch", "dyaw"}:
            return self.rotation_step_rad
        if axis == "gripper":
            return self.gripper_step
        raise ValueError(f"unknown axis {axis!r}")

    @staticmethod
    def _axis_command(axis: str, value: float) -> ToolCommand:
        data = {name: 0.0 for name in AXES}
        data[axis] = value
        return ToolCommand(**data)

    def _integrated_command(
        self,
        *,
        origin: ToolCommand,
        delta: ToolCommand,
        frame_scale: float,
        frozen: bool,
    ) -> ToolCommand:
        if frozen:
            delta = ToolCommand(gripper=delta.gripper)
        return origin.plus(delta.scaled(frame_scale)).clamped(
            translation_limit=self.max_translation_m,
            rotation_limit=self.max_rotation_rad,
            gripper_limit=self.max_gripper,
        )


@dataclass(frozen=True)
class KeyboardPacket:
    seq: int
    timestamp: float
    keys: frozenset[str]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "KeyboardPacket":
        if payload.get("type") != "keyboard_state":
            raise ValueError("packet type must be keyboard_state")
        seq = payload.get("seq")
        timestamp = payload.get("timestamp")
        keys = payload.get("keys", [])
        if not isinstance(seq, int):
            raise ValueError("seq must be an integer")
        if not isinstance(timestamp, (int, float)):
            raise ValueError("timestamp must be numeric")
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            raise ValueError("keys must be a list of key codes")
        return cls(seq=seq, timestamp=float(timestamp), keys=frozenset(keys))
