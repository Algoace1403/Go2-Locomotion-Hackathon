"""
Mathematical Utility Functions
"""

import torch
import numpy as np


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-π, π]"""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute smallest signed angle difference"""
    diff = a - b
    return normalize_angle(diff)


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)

    Args:
        quat: Quaternion (w, x, y, z) shape (..., 4)

    Returns:
        Euler angles (roll, pitch, yaw) shape (..., 3)
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Roll (x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * np.pi / 2,
        torch.asin(sinp)
    )

    # Yaw (z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def euler_to_quat(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles to quaternion

    Args:
        euler: Euler angles (roll, pitch, yaw) shape (..., 3)

    Returns:
        Quaternion (w, x, y, z) shape (..., 4)
    """
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)


def quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector by quaternion

    Args:
        quat: Quaternion (w, x, y, z) shape (..., 4)
        vec: Vector shape (..., 3)

    Returns:
        Rotated vector shape (..., 3)
    """
    # Convert vector to quaternion (0, v)
    vec_quat = torch.cat([
        torch.zeros_like(vec[..., :1]),
        vec
    ], dim=-1)

    # q * v * q^(-1)
    quat_conj = quat.clone()
    quat_conj[..., 1:] = -quat_conj[..., 1:]

    result = quat_multiply(quat_multiply(quat, vec_quat), quat_conj)

    return result[..., 1:4]


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=-1)


def exp_kernel(error: torch.Tensor, sigma: float = 0.25) -> torch.Tensor:
    """Gaussian/exponential kernel for smooth reward shaping"""
    return torch.exp(-error / sigma)


def smooth_abs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Smooth approximation of absolute value"""
    return torch.sqrt(x ** 2 + eps)
