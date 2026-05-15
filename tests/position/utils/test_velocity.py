"""Unit tests for spyglass.position.utils.velocity.compute_velocity."""

import numpy as np
import pytest


def test_compute_velocity_straight_line():
    """Uniform motion in x: speed == dx/dt everywhere."""
    from spyglass.position.utils.velocity import compute_velocity

    n = 100
    fps = 20.0
    timestamps = np.arange(n) / fps
    position = np.column_stack([timestamps * 10.0, np.zeros(n)])  # 10 cm/s in x

    vel_2d, speed = compute_velocity(position, timestamps)

    # np.gradient gives central diffs; edge values differ slightly
    # interior points should be ~10 cm/s in x, ~0 in y
    np.testing.assert_allclose(vel_2d[1:-1, 0], 10.0, rtol=1e-6)
    np.testing.assert_allclose(vel_2d[1:-1, 1], 0.0, atol=1e-10)
    np.testing.assert_allclose(speed[1:-1], 10.0, rtol=1e-6)


def test_compute_velocity_matches_position_tools():
    """compute_velocity must match position_tools.get_velocity for same params."""
    from position_tools import get_velocity as pt_get_velocity

    from spyglass.position.utils.velocity import compute_velocity

    rng = np.random.default_rng(42)
    n = 200
    fps = 16.7
    timestamps = np.arange(n) / fps
    position = rng.standard_normal((n, 2)).cumsum(axis=0)
    std_dev = 0.1  # seconds

    vel_2d, speed = compute_velocity(
        position, timestamps, smooth_std_dev=std_dev
    )
    expected_vel = pt_get_velocity(
        position, time=timestamps, sigma=std_dev, sampling_frequency=fps
    )
    expected_speed = np.sqrt(np.sum(expected_vel**2, axis=1))

    np.testing.assert_allclose(vel_2d, expected_vel, rtol=1e-6)
    np.testing.assert_allclose(speed, expected_speed, rtol=1e-6)


def test_compute_velocity_turn_smoothing_order():
    """2D smoothing before magnitude ≠ scalar smoothing after magnitude.

    This test guards the regression: when an animal turns sharply, smoothing
    the 2D vector first (V1 approach, correct) yields lower apparent speed
    than smoothing the scalar magnitude (old V2 approach, wrong).
    """
    from spyglass.position.utils.velocity import compute_velocity

    # Build a sharp 90-degree turn at frame 50
    n = 100
    fps = 20.0
    timestamps = np.arange(n) / fps
    speed_val = 10.0  # cm/s
    x = np.concatenate(
        [np.arange(50) * (speed_val / fps), np.full(50, 50 * speed_val / fps)]
    )
    y = np.concatenate([np.zeros(50), np.arange(50) * (speed_val / fps)])
    position = np.column_stack([x, y])

    # 2D-smoothed (correct): velocity vector smoothed before magnitude
    vel_2d, speed_2d_smooth = compute_velocity(
        position, timestamps, smooth_std_dev=0.1
    )

    # Scalar-smoothed (old wrong approach): magnitude first, gaussian after
    from position_tools.core import gaussian_smooth

    raw_vel_2d = np.gradient(position, timestamps, axis=0)
    raw_speed = np.sqrt(np.sum(raw_vel_2d**2, axis=1))
    speed_scalar_smooth = gaussian_smooth(
        raw_speed, 0.1, fps, axis=0, truncate=8
    )

    # Near the turn, 2D-smoothed speed is lower (turn attenuated in vector)
    turn_region = slice(45, 56)
    assert np.max(speed_scalar_smooth[turn_region]) > np.max(
        speed_2d_smooth[turn_region]
    ), "scalar-smoothed peak should exceed 2D-smoothed peak at a sharp turn"


def test_compute_velocity_no_smoothing():
    """Without smooth_std_dev, result equals np.gradient speed."""
    from spyglass.position.utils.velocity import compute_velocity

    n = 50
    fps = 10.0
    timestamps = np.arange(n) / fps
    rng = np.random.default_rng(7)
    position = rng.standard_normal((n, 2)).cumsum(axis=0)

    vel_2d, speed = compute_velocity(position, timestamps)

    expected = np.gradient(position, timestamps, axis=0)
    np.testing.assert_allclose(vel_2d, expected, rtol=1e-12)
    np.testing.assert_allclose(
        speed, np.sqrt(np.sum(expected**2, axis=1)), rtol=1e-12
    )


def test_compute_velocity_nan_positions_no_inf():
    """NaN positions produce NaN velocity — never inf or divide-by-zero.

    Regression guard for high-NaN sleep epochs: long NaN gaps in position
    should propagate as NaN into velocity, not produce inf values from
    dividing large displacement by zero dt.
    """
    from spyglass.position.utils.velocity import compute_velocity

    n = 100
    fps = 20.0
    timestamps = np.arange(n) / fps
    rng = np.random.default_rng(13)
    position = rng.standard_normal((n, 2)).cumsum(axis=0)

    # Scatter NaN across 40 % of frames, including a contiguous 20-frame gap
    position[20:40] = np.nan
    nan_idx = rng.choice(n, 10, replace=False)
    position[nan_idx] = np.nan

    vel_2d, speed = compute_velocity(position, timestamps)

    assert not np.any(np.isinf(vel_2d)), "velocity_2d must not contain inf"
    assert not np.any(np.isinf(speed)), "speed must not contain inf"
    # Every finite value must be a valid number
    assert np.all(np.isfinite(vel_2d) | np.isnan(vel_2d))
    assert np.all(np.isfinite(speed) | np.isnan(speed))


def test_compute_velocity_all_nan_no_error():
    """All-NaN position (e.g. fully occluded animal) returns all-NaN cleanly."""
    from spyglass.position.utils.velocity import compute_velocity

    n = 50
    fps = 16.0
    timestamps = np.arange(n) / fps
    position = np.full((n, 2), np.nan)

    vel_2d, speed = compute_velocity(position, timestamps)

    assert vel_2d.shape == (n, 2)
    assert speed.shape == (n,)
    assert np.all(np.isnan(vel_2d))
    assert np.all(np.isnan(speed))
