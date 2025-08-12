# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.utils.math_utils import axis_angle_diff, quat_diff_so3

XML_PATH = str(MODEL_PATH / "xml/caltech_leap_cube.xml")
SIM_XML_PATH = str(MODEL_PATH / "xml/caltech_leap_cube_sim.xml")
QPOS_HOME = np.array(
    [
        0.11, 0.005, 0.04, 1.0, 0.0, 0.0, 0.0,  # cube
        0.5, -0.75, 0.75, 0.25,  # index
        0.5, 0.0, 0.75, 0.25,  # middle
        0.5, 0.75, 0.75, 0.25,  # ring
        0.65, 0.9, 0.75, 0.6,  # thumb
    ]
)  # fmt: skip


@slider("w_pos", 0.0, 1000.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class CaltechLeapCubeConfig(LeapCubeConfig):
    """Reward configuration LEAP cube rotation task."""

    des_rot_rate: float = 1.0  # rad/s


class CaltechLeapCube(LeapCube):
    """Defines the LEAP cube rotation task."""

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str = SIM_XML_PATH) -> None:
        """Initializes the LEAP cube rotation task."""
        super(LeapCube, self).__init__(model_path, sim_model_path=sim_model_path)
        self.goal_pos = np.array([0.12, -0.01, 0.03])
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.qpos_home = QPOS_HOME
        self.reset_command = np.array(
            [
                0.5, -0.75, 0.75, 0.25,  # index
                0.5, 0.0, 0.75, 0.25,  # middle
                0.5, 0.75, 0.75, 0.25,  # ring
                0.65, 0.9, 0.75, 0.6,  # thumb
            ]
        )  # fmt: skip
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: LeapCubeConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the LEAP cube rotation task reward."""
        if system_metadata is None:
            system_metadata = {}

        # config/metadata
        goal_quat = system_metadata.get("goal_quat", self.goal_quat)  # (4,)
        omega = config.des_rot_rate

        # time processing
        times = system_metadata.get("times", None)
        if times is None:
            dt = float(system_metadata.get("dt", 0.02))
            T = states.shape[-2]  # time dimension
            times = np.arange(T, dtype=np.float64) * dt
        else:
            times = np.asarray(times, dtype=np.float64)

        # extract the trajectories
        # states shape assumed (B, T, ...), with quaternion in wxyz at indices 3:7 as in your code.
        qo_pos_traj = states[..., :3]  # (B, T, 3)
        qo_quat_traj = states[..., 3:7]  # (B, T, 4)
        B, T = qo_quat_traj.shape[:2]

        # start quat of goal traj is first step per batch
        q_start = qo_quat_traj[:, 0, :]  # (B,4)

        # build SLERP reference to goal at constant rate
        q_goal = np.broadcast_to(goal_quat[None, :], (B, 4))  # (B, 4)
        q_ref = slerp_path_from_rate(q_start, q_goal, times, omega, axis_angle_diff)  # (B, T, 4)

        # position cost is as before
        qo_pos_diff = qo_pos_traj - self.goal_pos[..., :3]  # (B, T, 3)

        # rotation: track the reference SLERP quaternion at every step instead of to a static goal
        rot_err = quat_diff_so3(qo_quat_traj, q_ref)  # (B, T, 3)

        w_pos = float(config.w_pos)
        w_rot = float(config.w_rot)

        pos_cost = w_pos * 0.5 * np.square(qo_pos_diff).sum(axis=-1).mean(axis=-1)  # (B,)
        rot_cost = w_rot * 0.5 * np.square(rot_err).sum(axis=-1).mean(axis=-1)  # (B,)

        rewards = -(pos_cost + rot_cost)
        return rewards


def _ensure_unit_quat(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Ensure that the input quaternion is unit length."""
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.clip(n, eps, None)
    return q / n


def quat_slerp(u: np.ndarray, v: np.ndarray, alpha: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Unit-quaternion slerp (wxyz), batched.

    Args:
        u: start quaternion, shape (..., 4)
        v: end quaternion, shape (..., 4)
        alpha: interpolation factor, shape (..., 1) or (...,)
        eps: numerical stability threshold

    Returns:
        out: slerped quaternion, shape (..., 4)
    """
    u = _ensure_unit_quat(u)
    v = _ensure_unit_quat(v)

    # make dot >= 0 to avoid the long path (antipodal handling)
    dot = (u * v).sum(axis=-1)
    v_adj = np.where(dot[..., None] < 0.0, -v, v)
    dot = np.abs(dot)

    # if very close, fall back to normalized lerp
    close = dot > (1.0 - 1e-7)
    omega = np.arccos(np.clip(dot, -1.0, 1.0))  # (...,)
    sin_omega = np.sin(omega)  # (...,)

    # broadcast alpha
    a = np.clip(alpha, 0.0, 1.0)[..., None]  # (...,1)

    # slerp
    s0 = np.sin((1.0 - a[..., 0]) * omega) / np.where(sin_omega < eps, 1.0, sin_omega)
    s1 = np.sin(a[..., 0] * omega) / np.where(sin_omega < eps, 1.0, sin_omega)
    out = s0[..., None] * u + s1[..., None] * v_adj

    # lerp for near-parallel
    lerp = _ensure_unit_quat((1.0 - a) * u + a * v_adj)
    return np.where(close[..., None], lerp, out)


def slerp_path_from_rate(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    times: np.ndarray,
    omega: float,
    angle_axis_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Return reference quats along shortest-path SLERP at constant angular speed.

    Args:
        q_start: start quaternion, shape (B, 4)
        q_goal: goal quaternion, shape (B, 4)
        times: time steps, shape (T,)
        omega: desired angular speed in rad/s
        angle_axis_fn: function to compute the angle-axis difference between quaternions
    """
    q_start = np.asarray(q_start)
    q_goal = np.asarray(q_goal)
    times = np.asarray(times, dtype=np.float64)
    if q_start.ndim == 1:
        q_start = q_start[None, :]
    if q_goal.ndim == 1:
        q_goal = q_goal[None, :]

    B = max(q_start.shape[0], q_goal.shape[0])
    if q_start.shape[0] == 1:
        q_start = np.repeat(q_start, B, axis=0)
    if q_goal.shape[0] == 1:
        q_goal = np.repeat(q_goal, B, axis=0)

    # align times to start at 0
    t0 = float(times[0])
    t = times - t0
    T = t.shape[0]

    # total shortest rotation angle per batch
    theta, _ = angle_axis_fn(q_start, q_goal)  # (B,)
    theta = np.clip(theta, 0.0, np.pi)
    T_reach = np.full((B,), np.inf) if omega <= 0 else np.where(theta < 1e-9, 0.0, theta / omega)

    # alpha in [0,1], shape (B, T)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(T_reach[:, None] > 0.0, t[None, :] / T_reach[:, None], 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # tile start and goal quats to match time steps
    U = np.broadcast_to(q_start[:, None, :], (B, T, 4)).reshape(B * T, 4)
    V = np.broadcast_to(q_goal[:, None, :], (B, T, 4)).reshape(B * T, 4)
    A = alpha.reshape(B * T)

    q_bt = quat_slerp(U, V, A)  # (B*T, 4)
    return q_bt.reshape(B, T, 4)
