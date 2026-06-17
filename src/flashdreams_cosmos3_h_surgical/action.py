"""Surgical simulator action tensor construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable, Sequence

ACTION_DIM = 44
CHUNK_STEPS = 12
DVRK_NATIVE_DIM = 20

LEFT_POSE_SLICE = slice(0, 9)
LEFT_GRIPPER_INDEX = 9
RIGHT_POSE_SLICE = slice(10, 19)
RIGHT_GRIPPER_INDEX = 19
PADDING_SLICE = slice(20, ACTION_DIM)


class ActionContractError(ValueError):
    """Raised when a simulator action packet does not match the runtime contract."""


@dataclass(frozen=True)
class ToolCommand:
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    droll: float = 0.0
    dpitch: float = 0.0
    dyaw: float = 0.0
    gripper: float = 0.0

    def scaled(self, value: float) -> "ToolCommand":
        return ToolCommand(
            dx=self.dx * value,
            dy=self.dy * value,
            dz=self.dz * value,
            droll=self.droll * value,
            dpitch=self.dpitch * value,
            dyaw=self.dyaw * value,
            gripper=self.gripper * value,
        )

    def plus(self, other: "ToolCommand") -> "ToolCommand":
        return ToolCommand(
            dx=self.dx + other.dx,
            dy=self.dy + other.dy,
            dz=self.dz + other.dz,
            droll=self.droll + other.droll,
            dpitch=self.dpitch + other.dpitch,
            dyaw=self.dyaw + other.dyaw,
            gripper=self.gripper + other.gripper,
        )

    def clamped(
        self,
        *,
        translation_limit: float,
        rotation_limit: float,
        gripper_limit: float,
    ) -> "ToolCommand":
        return ToolCommand(
            dx=clamp(self.dx, -translation_limit, translation_limit),
            dy=clamp(self.dy, -translation_limit, translation_limit),
            dz=clamp(self.dz, -translation_limit, translation_limit),
            droll=clamp(self.droll, -rotation_limit, rotation_limit),
            dpitch=clamp(self.dpitch, -rotation_limit, rotation_limit),
            dyaw=clamp(self.dyaw, -rotation_limit, rotation_limit),
            gripper=clamp(self.gripper, -gripper_limit, gripper_limit),
        )


@dataclass(frozen=True)
class ControlFlags:
    hold: bool = False
    recenter: bool = False
    left_clutch: bool = False
    right_clutch: bool = False


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def euler_xyz_to_rot6d(roll: float, pitch: float, yaw: float) -> list[float]:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r20 = -sp
    r21 = cp * sr
    return [r00, r10, r20, r01, r11, r21]


def tool_pose9(command: ToolCommand) -> list[float]:
    return [
        command.dx,
        command.dy,
        command.dz,
        *euler_xyz_to_rot6d(command.droll, command.dpitch, command.dyaw),
    ]


def build_action_vector(
    left: ToolCommand,
    right: ToolCommand,
    flags: ControlFlags | None = None,
) -> list[float]:
    flags = flags or ControlFlags()
    if flags.hold:
        left = ToolCommand(gripper=left.gripper)
        right = ToolCommand(gripper=right.gripper)
    if flags.left_clutch:
        left = ToolCommand(gripper=left.gripper)
    if flags.right_clutch:
        right = ToolCommand(gripper=right.gripper)

    vector = [0.0] * ACTION_DIM
    vector[LEFT_POSE_SLICE] = tool_pose9(left)
    vector[LEFT_GRIPPER_INDEX] = clamp(left.gripper, -1.0, 1.0)
    vector[RIGHT_POSE_SLICE] = tool_pose9(right)
    vector[RIGHT_GRIPPER_INDEX] = clamp(right.gripper, -1.0, 1.0)
    validate_action_vector(vector)
    return vector


def hold_vector() -> list[float]:
    return build_action_vector(ToolCommand(), ToolCommand(), ControlFlags(hold=True))


def repeat_action(vector: Sequence[float], steps: int = CHUNK_STEPS) -> list[list[float]]:
    if steps != CHUNK_STEPS:
        raise ActionContractError(f"action chunk length must be {CHUNK_STEPS}")
    validate_action_vector(vector)
    return [list(float(value) for value in vector) for _ in range(steps)]


def validate_action_vector(vector: Sequence[float]) -> None:
    if len(vector) != ACTION_DIM:
        raise ActionContractError(f"action vector must contain {ACTION_DIM} values")
    for index, value in enumerate(vector):
        if not isinstance(value, (int, float)):
            raise ActionContractError(f"action value {index} is not numeric")
        if not math.isfinite(float(value)):
            raise ActionContractError(f"action value {index} is not finite")
    for index, value in enumerate(vector[PADDING_SLICE], start=PADDING_SLICE.start):
        if abs(float(value)) > 1e-8:
            raise ActionContractError(f"padding action value {index} must be zero")


def validate_action_chunk(chunk: Sequence[Sequence[float]]) -> None:
    if len(chunk) != CHUNK_STEPS:
        raise ActionContractError(f"action chunk must contain {CHUNK_STEPS} rows")
    for row_index, row in enumerate(chunk):
        try:
            validate_action_vector(row)
        except ActionContractError as exc:
            raise ActionContractError(f"action chunk row {row_index}: {exc}") from exc


def chunk_from_rows(rows: Iterable[Iterable[float]]) -> list[list[float]]:
    chunk = [list(float(value) for value in row) for row in rows]
    validate_action_chunk(chunk)
    return chunk
