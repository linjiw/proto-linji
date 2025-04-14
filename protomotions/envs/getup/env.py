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
from protomotions.simulator.base_simulator.robot_state import RobotState
from protomotions.simulator.base_simulator.config import MarkerConfig, VisualizationMarker, MarkerState


class Getup(BaseEnv):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        self.real_self_obs = torch.zeros(
            (self.config.num_envs, self.config.getup_params.obs_size),
            device=self.device,
        )
        self.last_actions = torch.zeros(
            (self.config.num_envs, self.get_action_size()),
            device=self.device,
        )
        self.gravity_vec = torch_utils.to_torch(
            torch_utils.get_axis_params(-1.0, 2),
            device=self.device
        ).repeat((self.num_envs, 1))
        robot_config = self.simulator.robot_config
        self.head_id = robot_config.body_names.index(robot_config.head_body_name)

        # initialize at random initial progress
        self.progress_buf[:] = torch.randint(
            low=0,
            high=self.config.max_episode_length,
            size=(self.num_envs,),
            device=self.device,
        )
        non_termination_contact_bodies = robot_config.non_termination_contact_bodies
        penalize_contact_body_ids = [
            robot_config.body_names.index(body_name)
            for body_name in robot_config.body_names if body_name not in non_termination_contact_bodies
        ]
        self.penalize_contact_indices = torch.tensor(
            penalize_contact_body_ids,
            device=self.device,
            dtype=torch.long
        )

    def reset_default(self, env_ids):
        # Adjust root position
        default_state = self.default_state

        root_pos = default_state.root_pos[env_ids].clone()
        root_rot = default_state.root_rot[env_ids].clone()
        dof_pos = default_state.dof_pos[env_ids].clone()
        root_vel = default_state.root_vel[env_ids].clone()
        root_ang_vel = default_state.root_ang_vel[env_ids].clone()
        dof_vel = default_state.dof_vel[env_ids].clone()
        rigid_body_pos = default_state.rigid_body_pos[env_ids].clone()
        rigid_body_rot = default_state.rigid_body_rot[env_ids].clone()
        rigid_body_vel = default_state.rigid_body_vel[env_ids].clone()
        rigid_body_ang_vel = default_state.rigid_body_ang_vel[env_ids].clone()

        roll = (torch.rand(len(env_ids), device=self.device) - 0.5) * torch.pi * 0.8
        pitch = (torch.rand(len(env_ids), device=self.device) - 0.5) * torch.pi * 0.8
        yaw = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi

        base_quat = rotations.quat_from_euler_xyz(roll, pitch, yaw, w_last=True)
        p = torch.rand(len(env_ids), device=self.device)
        standing = p < 0.1
        base_quat[standing] = torch.tensor([0., 0., 0., 1.], device=self.device)[None, :]  # keep the initial orientation for lying robots

        # Estimate body height to keep feet near ground
        # Assume you know the body-to-foot lowest z-offset in local frame (e.g., self.foot_clearance)
        # Compute rotated local z offset
        local_foot_offset = torch.tensor([0.0, 0.0, 0.68], device=self.device).view(1, 3)
        rotated_offsets = rotations.quat_rotate(base_quat, local_foot_offset.repeat(len(env_ids), 1), w_last=True)
        adjusted_z = rotated_offsets[:, 2]  # height to make lowest foot touch ground
        root_pos[env_ids, 2] = adjusted_z + 0.1

        root_rot[env_ids] = base_quat
        root_pos[:, :2] = 0
        root_pos[:, :3] += self.get_envs_respawn_position(
            env_ids,
            rigid_body_pos=rigid_body_pos,
            offset=0,
        )

        # Transfer entire body to the proper coordinates
        # ZIFAN: rigid_body_pos is not used for reset
        rigid_body_pos[:, :, :3] -= (
            rigid_body_pos[:, 0, :3].unsqueeze(1).clone()
        )
        rigid_body_pos[:, :, :3] += root_pos.unsqueeze(1)
        
        new_states = RobotState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            rigid_body_pos=rigid_body_pos,
            rigid_body_rot=rigid_body_rot,
            rigid_body_vel=rigid_body_vel,
            rigid_body_ang_vel=rigid_body_ang_vel,
        )

        return new_states

    def step(self, actions):
        out = super().step(actions)
        self.last_actions[:] = actions
        return out
    
    def compute_reset(self):
        super().compute_reset()
        # terminate at large velocity
        root_vel = self.simulator.get_root_state().root_vel
        root_ang_vel = self.simulator.get_root_state().root_ang_vel
        self.terminate_buf[:] = torch.logical_or(
            root_vel.norm(dim=-1) > 8.0,
            self.terminate_buf
        ).to(self.device)
        self.terminate_buf[:] = torch.logical_or(
            root_ang_vel.norm(dim=-1) > 16.0,
            self.terminate_buf
        ).to(self.device)

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
            self.last_actions[env_ids]
        ], dim=-1)

    def get_obs(self):
        obs = super().get_obs()
        obs.update({"real_self_obs": self.real_self_obs})
        return obs

    def compute_reward(self):
        root_states = self.simulator.get_root_state()
        bodies_states = self.simulator.get_bodies_state()
        dof_forces = self.simulator.get_dof_forces()
        dof_states = self.simulator.get_dof_state()
        contact_forces = self.simulator.get_bodies_contact_buf()

        base_height_exp = torch.exp(
            (root_states.root_pos[:, 2] - self.config.getup_params.target_base_height).clip(max=0.0) / 0.1
        )
        head_height_exp = torch.exp(
            (bodies_states.rigid_body_pos[:, self.head_id, 2] - self.config.getup_params.target_head_height).clip(max=0.0) / 0.1
        )
        base_height_norm = (root_states.root_pos[:, 2] / self.config.getup_params.target_base_height).clip(min=0.0, max=1.0)
        head_height_norm = (bodies_states.rigid_body_pos[:, self.head_id, 2] / self.config.getup_params.target_head_height).clip(min=0.0, max=1.0)
        power = torch.abs(torch.multiply(dof_forces, dof_states.dof_vel.clip(min=-100.0, max=100.))).sum(dim=-1)
        base_vel = torch.square(root_states.root_vel).sum(dim=-1).clip(max=100.0)
        contact_penalty = (contact_forces[:, self.penalize_contact_indices].sum(dim=-1) > 0.1).sum(dim=-1).float()

        self.log_dict["raw/base_height_exp"] = base_height_exp.mean()
        self.log_dict["raw/head_height_exp"] = head_height_exp.mean()
        self.log_dict["raw/base_height_norm"] = base_height_norm.mean()
        self.log_dict["raw/head_height_norm"] = head_height_norm.mean()
        self.log_dict["raw/power"] = power.mean()
        self.log_dict["raw/base_vel"] = base_vel.mean()
        self.log_dict["raw/contact_penalty"] = contact_penalty.mean()

        # self.rew_buf[:] = 2.0 * base_height_exp + 2.0 * head_height_exp - 1.e-5 * power - 0.5 * contact_penalty - 0.2 * base_vel
        self.rew_buf[:] = \
            self.config.getup_params.reward_scales.base_height_norm * base_height_norm + \
            self.config.getup_params.reward_scales.base_height_norm * head_height_norm - \
            self.config.getup_params.reward_scales.contact_penalty * contact_penalty # - 1.e-5 * power