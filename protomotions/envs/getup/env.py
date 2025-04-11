# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import torch
from torch import Tensor
from isaac_utils import rotations, torch_utils
from protomotions.envs.base_env.env import BaseEnv
from protomotions.simulator.base_simulator.config import MarkerConfig, VisualizationMarker, MarkerState


class Getup(BaseEnv):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        self.real_self_obs = torch.zeros(
            (self.config.num_envs, self.config.getup_params.obs_size),
            device=self.device,
        )
        # self.last_actions = torch.zeros(
        #     (self.config.num_envs, self.get_action_size()),
        #     device=self.device,
        # )
        self.gravity_vec = torch_utils.to_torch(
            torch_utils.get_axis_params(-1.0, 2),
            device=self.device
        ).repeat((self.num_envs, 1))

    # def reset_defaults(self):
    #     # TODO: overwrite this to get getup init state
    #     raise NotImplementedError()

    # def reset(self, env_ids=None):
    #     return super().reset(env_ids)

    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        root_states = self.simulator.get_root_state(env_ids)
        dof_states = self.simulator.get_dof_state(env_ids)
        projected_gravity = rotations.quat_rotate_inverse(
            root_states.root_rot, self.gravity_vec[env_ids], w_last=True)

        self.real_self_obs[env_ids, :] = torch.cat([
            projected_gravity,
            root_states.root_ang_vel,
            dof_states.dof_pos,
            dof_states.dof_vel * 0.1,
            # self.last_actions[env_ids]
        ], dim=-1)

    def get_obs(self):
        obs = super().get_obs()
        obs.update({"real_self_obs": self.real_self_obs})
        return obs

    def compute_reward(self):
        root_states = self.simulator.get_root_state()
        dof_forces = self.simulator.get_dof_forces()
        dof_states = self.simulator.get_dof_state()
        self.rew_buf[:] = compute_getup_reward(
            root_states.root_pos,
            root_states.root_vel,
            dof_states.dof_vel,
            dof_forces,
            self.dt
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_getup_reward(
    root_pos: Tensor,
    root_vel: Tensor,
    dof_vel: Tensor,
    dof_forces: Tensor,
    dt: float,
) -> Tensor:
    """Compute the reward for the getup task.
    Args:
        root_pos (Tensor): Root position of the robot.
        root_vel (Tensor): Root velocity of the robot.
        dt (float): Time step.
    Returns:
        Tensor: Computed reward.
    """
    base_height_exp = torch.exp(
        -torch.norm(root_pos[:, 2:3] - 0.68, dim=-1) / 0.1
    )
    power = torch.abs(torch.multiply(dof_forces, dof_vel)).sum(dim=-1)
    base_xy_vel = torch.norm(root_vel[:, :2], dim=-1)
    return 5.0 * base_height_exp - 1.e-5 * power - 1.0 * base_xy_vel
