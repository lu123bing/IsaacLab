# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the joint impedance controller command handling.

These exercise the controller in isolation, without launching the simulator.
"""

import pytest
import torch

from isaaclab.controllers.joint_impedance import JointImpedanceController
from isaaclab.controllers.joint_impedance_cfg import JointImpedanceControllerCfg

pytestmark = pytest.mark.unit

_NUM_ROBOTS = 4
_NUM_DOF = 7
_TARGET = 0.1
# a non-zero lower limit, so that a parameter below it is distinguishable from an unset one
_STIFFNESS_LIMITS = (10.0, 300.0)
_DAMPING_RATIO_LIMITS = (0.2, 1.0)


def _limits() -> torch.Tensor:
    """Return symmetric joint position limits wide enough that the clamp on them is not exercised."""
    return torch.stack(
        [-0.3 * torch.ones(_NUM_ROBOTS, _NUM_DOF), 0.3 * torch.ones(_NUM_ROBOTS, _NUM_DOF)], dim=-1
    )


def _controller(impedance_mode: str) -> JointImpedanceController:
    cfg = JointImpedanceControllerCfg(
        impedance_mode=impedance_mode,
        command_type="p_abs",
        stiffness=50.0,
        dof_pos_offset=[0.1] * _NUM_DOF,
        stiffness_limits=_STIFFNESS_LIMITS,
        damping_ratio_limits=_DAMPING_RATIO_LIMITS,
    )
    return JointImpedanceController(cfg, _NUM_ROBOTS, _limits(), "cpu")


def _command(controller: JointImpedanceController) -> torch.Tensor:
    """Return one command whose impedance parameters span above, below and on their configured limits."""
    stiffness_lower, stiffness_upper = controller.cfg.stiffness_limits
    command = torch.zeros(controller.num_actions)
    command[: _NUM_DOF] = _TARGET
    mid = 0.5 * (stiffness_lower + stiffness_upper)
    command[_NUM_DOF : _NUM_DOF * 2] = torch.tensor(
        [2.0 * stiffness_upper, 0.0, stiffness_upper, stiffness_lower, mid, mid, mid]
    )
    if controller.cfg.impedance_mode == "variable":
        ratio_lower, ratio_upper = controller.cfg.damping_ratio_limits
        ratio_mid = 0.5 * (ratio_lower + ratio_upper)
        command[_NUM_DOF * 2 :] = torch.tensor(
            [2.0 * ratio_upper, 0.0, ratio_upper, ratio_lower, ratio_mid, ratio_mid, ratio_mid]
        )
    return command


@pytest.mark.parametrize("impedance_mode", ["variable_kp", "variable"])
def test_set_command_clamps_impedance_parameters_to_their_limits(impedance_mode: str) -> None:
    """A parameter above, below or on a limit maps onto that limit, and an in-range one is kept."""
    controller = _controller(impedance_mode)
    command = _command(controller).repeat(_NUM_ROBOTS, 1)

    controller.set_command(command)

    expected_stiffness = torch.tensor([300.0, 10.0, 300.0, 10.0, 155.0, 155.0, 155.0])
    torch.testing.assert_close(controller._p_gains, expected_stiffness.repeat(_NUM_ROBOTS, 1))
    if impedance_mode == "variable":
        expected_ratio = torch.tensor([1.0, 0.2, 1.0, 0.2, 0.6, 0.6, 0.6])
    else:
        expected_ratio = torch.full_like(expected_stiffness, 1.0)
    expected_damping = 2.0 * expected_stiffness.sqrt() * expected_ratio
    torch.testing.assert_close(controller._d_gains, expected_damping.repeat(_NUM_ROBOTS, 1))


@pytest.mark.parametrize("impedance_mode", ["variable_kp", "variable"])
def test_set_command_accepts_read_only_command_tensors(impedance_mode: str) -> None:
    """The impedance parameters are clamped without writing through the caller command tensor.

    torch.tensor_split returns views that alias the command, so clamping in place raises for a command
    that is broadcast across robots, or that carries requires_grad.
    """
    controller = _controller(impedance_mode)
    command = _command(controller)

    # one command shared by every robot: the caller tensor is not writable
    controller.set_command(command.expand(_NUM_ROBOTS, -1))
    # a command that carries a gradient: its views are not writable either
    controller.set_command(command.repeat(_NUM_ROBOTS, 1).requires_grad_(True))


@pytest.mark.parametrize("impedance_mode", ["variable_kp", "variable"])
def test_set_command_leaves_a_writable_command_untouched(impedance_mode: str) -> None:
    """set_command reads its command, so a caller can keep using the buffer afterwards."""
    controller = _controller(impedance_mode)
    command = _command(controller).repeat(_NUM_ROBOTS, 1)
    command_before = command.clone()

    controller.set_command(command)

    torch.testing.assert_close(command, command_before)
