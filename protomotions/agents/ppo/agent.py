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

import torch
import os
import logging
import numpy as np
from omegaconf import DictConfig

from torch import Tensor

import time
import math
from pathlib import Path
from typing import Optional, Tuple, Dict

from lightning.fabric import Fabric

from hydra.utils import instantiate
from isaac_utils import torch_utils

from protomotions.utils.time_report import TimeReport
from protomotions.utils.average_meter import AverageMeter, TensorAverageMeterDict
from protomotions.agents.utils.data_utils import DictDataset, ExperienceBuffer
from protomotions.agents.ppo.model import PPOModel
from protomotions.agents.common.common import weight_init, get_params
from protomotions.envs.base_env.env import BaseEnv
from protomotions.utils.running_mean_std import RunningMeanStd
from rich.progress import track
from protomotions.agents.ppo.utils import discount_values, bounds_loss

try:
    from torch_scatter import scatter_mean, scatter_add
except ImportError:
    print("torch_scatter not found. PLR score aggregation will be less efficient.")
    # Define fallback functions or raise an error if scatter ops are crucial
    def scatter_mean(src, index, dim_size, dim=0):
        # Basic fallback - might be slow
        out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
        counts = torch.zeros(dim_size, device=src.device, dtype=torch.long)
        scatter_add(src, index, dim=dim, out=out)
        scatter_add(torch.ones_like(src), index, dim=dim, out=counts.unsqueeze(-1).expand_as(out)) # Hacky way to count
        counts = torch.clamp(counts, min=1)
        return out / counts
    def scatter_add(src, index, dim_size, dim=0, out=None):
         # Basic fallback
        if out is None:
            out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
        for i in range(src.shape[dim]):
            out[index[i]] += src[i]
        return out

log = logging.getLogger(__name__)


class PPO:
    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    def __init__(self, fabric: Fabric, env: BaseEnv, config):
        self.fabric = fabric
        self.device: torch.device = fabric.device
        self.env = env
        self.motion_lib = self.env.motion_lib
        self.config = config

        self.num_envs: int = self.env.config.num_envs
        self.num_steps: int = config.num_steps
        self.gamma: float = config.gamma
        self.tau: float = config.tau
        self.e_clip: float = config.e_clip
        self.num_mini_epochs: int = config.num_mini_epochs
        self.task_reward_w: float = config.task_reward_w
        self._should_stop: bool = False

        if self.config.normalize_values:
            self.running_val_norm = RunningMeanStd(
                shape=(1,),
                device=self.device,
                clamp_value=self.config.normalized_val_clamp_value,
            )
        else:
            self.running_val_norm = None

        # timer
        self.time_report = TimeReport()

        self.current_lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.current_rewards = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        self.episode_reward_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.episode_env_tensors = TensorAverageMeterDict()
        self.step_count = 0
        self.current_epoch = 0
        self.fit_start_time = None
        self.best_evaluated_score = None

        self.force_full_restart = False

        # --- PLR Initialization ---
        self.plr_config = config.get('plr', None)
        self.plr_enabled = self.plr_config is not None and self.plr_config.get('enabled', False)
        print(f'config: {config}')
        print(f"PLR enabled: {self.plr_enabled}")
        if self.plr_enabled:
            print("PLR Motion Sampling Enabled")
            if not hasattr(self.env, 'motion_lib') or not hasattr(self.env, 'motion_manager'):
                 raise AttributeError("Environment must have 'motion_lib' and 'motion_manager' for PLR.")

            self.num_motions = self.env.motion_lib.num_motions()
            if self.num_motions <= 0:
                raise ValueError("MotionLib reported zero motions. Cannot enable PLR.")

            # Initialize PLR state tensors
            self.motion_scores = torch.zeros(self.num_motions, dtype=torch.float32, device=self.device)
            self.motion_staleness = torch.zeros(self.num_motions, dtype=torch.float32, device=self.device) # Use float for potential power transform
            # Initialize scores pessimistically (e.g., high value) to encourage exploration initially? Or zero? Let's start with zero.

            # Store last sampled IDs for staleness update
            self._last_sampled_motion_ids_in_batch = None

            # Validate PLR config
            if self.plr_config.strategy not in ['value_l1', 'gae_abs']:
                raise ValueError(f"Unsupported PLR strategy: {self.plr_config.strategy}")
            # Add more validation for transform types, etc.

    @property
    def should_stop(self):
        return self.fabric.broadcast(self._should_stop)

    def setup(self):
        model: PPOModel = instantiate(self.config.model)
        model.apply(weight_init)
        actor_optimizer = instantiate(
            self.config.model.config.actor_optimizer,
            params=list(model._actor.parameters()),
        )
        critic_optimizer = instantiate(
            self.config.model.config.critic_optimizer,
            params=list(model._critic.parameters()),
        )

        self.model, self.actor_optimizer, self.critic_optimizer = self.fabric.setup(
            model, actor_optimizer, critic_optimizer
        )
        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value")

    def load(self, checkpoint: Path):
        if checkpoint is not None:
            checkpoint = Path(checkpoint).resolve()
            print(f"Loading model from checkpoint: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.load_parameters(state_dict)
            
            env_checkpoint = checkpoint.resolve().parent / f"env_{self.fabric.global_rank}.ckpt"
            # if env_checkpoint.exists():
            #     print(f"Loading env checkpoint: {env_checkpoint}")
            #     env_state_dict = torch.load(env_checkpoint, map_location=self.device)
            #     self.env.load_state_dict(env_state_dict)

    def load_parameters(self, state_dict):
        self.current_epoch = state_dict["epoch"]

        if "step_count" in state_dict:
            self.step_count = state_dict["step_count"]
        if "run_start_time" in state_dict:
            self.fit_start_time = state_dict["run_start_time"]

        self.best_evaluated_score = state_dict.get("best_evaluated_score", None)

        self.model.load_state_dict(state_dict["model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

        if self.config.normalize_values:
            self.running_val_norm.load_state_dict(state_dict["running_val_norm"])

        self.episode_reward_meter.load_state_dict(state_dict["episode_reward_meter"])
        self.episode_length_meter.load_state_dict(state_dict["episode_length_meter"])

    # -----------------------------
    # Model Saving and State Dict
    # -----------------------------
    def get_state_dict(self, state_dict):
        extra_state_dict = {
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "epoch": self.current_epoch,
            "step_count": self.step_count,
            "run_start_time": self.fit_start_time,
            "episode_reward_meter": self.episode_reward_meter.state_dict(),
            "episode_length_meter": self.episode_length_meter.state_dict(),
            "best_evaluated_score": self.best_evaluated_score,
        }

        if self.config.normalize_values:
            extra_state_dict["running_val_norm"] = self.running_val_norm.state_dict()
        state_dict.update(extra_state_dict)
        return state_dict

    def save(self, path=None, name="last.ckpt", new_high_score=False):
        if path is None:
            path = self.fabric.loggers[0].log_dir
        root_dir = Path.cwd() / Path(self.fabric.loggers[0].root_dir)
        save_dir = Path.cwd() / Path(path)
        state_dict = self.get_state_dict({})
        self.fabric.save(save_dir / name, state_dict)

        if self.fabric.global_rank == 0:
            if root_dir != save_dir:
                if (root_dir / "last.ckpt").is_symlink():
                    (root_dir / "last.ckpt").unlink()
                # Make root_dir / "last.ckpt" point to the new checkpoint.
                # Calculate the relative path and create a symbolic link.
                relative_path = Path(os.path.relpath(save_dir / name, root_dir))
                (root_dir / "last.ckpt").symlink_to(relative_path)
                log.info(f"saved checkpoint, {root_dir / 'last.ckpt'}")
        self.fabric.barrier()
        
        # Save env state for all ranks to the same directory.
        rank_0_path = (root_dir / "last.ckpt").resolve().parent
        env_checkpoint = rank_0_path / f"env_{self.fabric.global_rank}.ckpt"
        env_state_dict = self.env.get_state_dict()
        torch.save(env_state_dict, env_checkpoint)

        # Check if new high score flag is consistent across devices.
        gathered_high_score = self.fabric.all_gather(new_high_score)
        assert all(
            [x == gathered_high_score[0] for x in gathered_high_score]
        ), "New high score flag should be the same across all ranks."

        if new_high_score:
            score_based_name = "score_based.ckpt"
            self.fabric.save(save_dir / score_based_name, state_dict)
            print(
                f"New best performing controller found with score {self.best_evaluated_score}. Model saved to {save_dir / score_based_name}."
            )
            if self.fabric.global_rank == 0:
                if root_dir != save_dir:
                    if (root_dir / "score_based.ckpt").is_symlink():
                        (root_dir / "score_based.ckpt").unlink()
                    # Create symlink for the best score checkpoint.
                    relative_path = Path(os.path.relpath(save_dir / name, root_dir))
                    (root_dir / "score_based.ckpt").symlink_to(relative_path)

    def export_jit(self, checkpoint: Path):
        from protomotions.agents.ppo.utils import ScriptablePolicyWrapper
        # Export the model to TorchScript.
        self.model.eval()
        scripted_model = ScriptablePolicyWrapper(self.model)
        scripted_model.eval()
        script = torch.jit.script(scripted_model)
        # save jit script
        save_path = Path(checkpoint).resolve().parent / "exported.pt"
        script.save(save_path)
        print(f"Exported JIT script to {save_path}")

    # -----------------------------
    # Experience Buffer and Training Loop
    # -----------------------------
    def register_extra_experience_buffer_keys(self):
        pass

    def fit(self):
        # Setup experience buffer
        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps).to(
            self.device
        )
        self.experience_buffer.register_key(
            "self_obs", shape=(self.env.config.robot.self_obs_size,)
        )
        self.experience_buffer.register_key(
            "actions", shape=(self.env.config.robot.number_of_actions,)
        )
        self.experience_buffer.register_key("rewards")
        self.experience_buffer.register_key("extra_rewards")
        self.experience_buffer.register_key("total_rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.experience_buffer.register_key("values")
        self.experience_buffer.register_key("next_values")
        self.experience_buffer.register_key("returns")
        self.experience_buffer.register_key("advantages")
        self.experience_buffer.register_key("neglogp")
        self.register_extra_experience_buffer_keys()

        if self.config.get("extra_inputs", None) is not None:
            obs = self.env.get_obs()
            for key in self.config.extra_inputs.keys():
                assert (
                    key in obs
                ), f"Key {key} not found in obs returned from env: {obs.keys()}"
                env_tensor = obs[key]
                shape = env_tensor.shape
                dtype = env_tensor.dtype
                self.experience_buffer.register_key(key, shape=shape[1:], dtype=dtype)

        # *** PLR: Register motion_ids key if PLR is enabled ***
        if self.plr_enabled:
            # Assuming motion_id is a single long integer per environment
            self.experience_buffer.register_key("motion_ids", shape=(), dtype=torch.long)
            # Point self.storage to self.experience_buffer for PLR methods
            self.storage = self.experience_buffer
            log.info("Registered 'motion_ids' key in ExperienceBuffer for PLR.")
        else:
            # Set storage to None if PLR is not enabled, PLR methods will check
            self.storage = None

        # Force reset on fit start
        done_indices = None
        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        self.fabric.call("on_fit_start", self)

        while self.current_epoch < self.config.max_epochs:
            self.epoch_start_time = time.time()

            # Set networks in eval mode so that normalizers are not updated
            self.eval()
            with torch.no_grad():
                self.fabric.call("before_play_steps", self)

                for step in track(
                    range(self.num_steps),
                    description=f"Epoch {self.current_epoch}, collecting data...",
                ):
                    obs = self.handle_reset(done_indices)
                    self.experience_buffer.update_data("self_obs", step, obs["self_obs"])
                    if self.config.get("extra_inputs", None) is not None:
                        for key in self.config.extra_inputs:
                            self.experience_buffer.update_data(key, step, obs[key])

                    action, neglogp, value = self.model.get_action_and_value(obs)
                    self.experience_buffer.update_data("actions", step, action)
                    self.experience_buffer.update_data("neglogp", step, neglogp)
                    if self.config.normalize_values:
                        value = self.running_val_norm.normalize(value, un_norm=True)
                    self.experience_buffer.update_data("values", step, value)

                    # Check for NaNs in observations and actions
                    for key in obs.keys():
                        if torch.isnan(obs[key]).any():
                            print(f"NaN in {key}: {obs[key]}")
                            raise ValueError("NaN in obs")
                    if torch.isnan(action).any():
                        raise ValueError(f"NaN in action: {action}")

                    # Step the environment
                    next_obs, rewards, dones, terminated, extras = self.env_step(action)

                    # *** PLR: Store motion_ids ***
                    if self.plr_enabled:
                        if 'motion_ids' not in extras:
                            raise KeyError("Environment 'extras' dictionary must contain 'motion_ids' when PLR is enabled.")
                        # Ensure motion_ids has the correct shape [num_envs]
                        motion_ids_for_step = extras['motion_ids'].to(self.device)
                        if motion_ids_for_step.ndim > 1:
                             motion_ids_for_step = motion_ids_for_step.squeeze() # Remove trailing dims if needed
                        self.experience_buffer.update_data("motion_ids", step, motion_ids_for_step)

                    all_done_indices = dones.nonzero(as_tuple=False)
                    done_indices = all_done_indices.squeeze(-1)

                    # Update logging metrics with the environment feedback
                    self.post_train_env_step(rewards, dones, done_indices, extras, step)

                    self.experience_buffer.update_data("rewards", step, rewards)
                    self.experience_buffer.update_data("dones", step, dones)

                    next_value = self.model._critic(next_obs).flatten()
                    if self.config.normalize_values:
                        next_value = self.running_val_norm.normalize(
                            next_value, un_norm=True
                        )
                    next_value = next_value * (1 - terminated.float())
                    self.experience_buffer.update_data("next_values", step, next_value)

                    self.step_count += self.get_step_count_increment()

                # After data collection, compute rewards, advantages, and returns.
                rewards = self.experience_buffer.rewards
                extra_rewards = self.calculate_extra_reward()
                self.experience_buffer.batch_update_data("extra_rewards", extra_rewards)
                total_rewards = rewards + extra_rewards
                self.experience_buffer.batch_update_data("total_rewards", total_rewards)

                advantages = discount_values(
                    self.experience_buffer.dones,
                    self.experience_buffer.values,
                    total_rewards,
                    self.experience_buffer.next_values,
                    self.gamma,
                    self.tau,
                )
                returns = advantages + self.experience_buffer.values
                self.experience_buffer.batch_update_data("returns", returns)

                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                self.experience_buffer.batch_update_data("advantages", advantages)

            # --- Model Optimization ---
            # This call updates the actor and critic networks
            training_log_dict = self.optimize_model()
            training_log_dict["epoch"] = self.current_epoch

            # --- PLR: Update State and Sampling Weights ---
            if self.plr_enabled:
                # Calculates scores from the *just processed* batch (via self.storage)
                # and updates EMA score state. Updates staleness based on the
                # sampled IDs from that batch.
                self._update_plr_state()
                # Calculates new weights based on updated scores/staleness
                # and pushes them to MotionManager for the *next* data collection phase.
                self._update_motion_sampling_weights()
                log.info(f"PLR state and weights updated for epoch {self.current_epoch}")

            self.fabric.call("after_train", self) # Callbacks after training step

            # --- Epoch End Tasks ---
            self.current_epoch += 1

            # Save model checkpoint at specified intervals before evaluation.
            if self.current_epoch % self.config.manual_save_every == 0:
                self.save()

            # Evaluation
            if (
                self.config.eval_metrics_every is not None
                and self.current_epoch > 0
                and self.current_epoch % self.config.eval_metrics_every == 0
            ):
                eval_log_dict, evaluated_score = self.calc_eval_metrics()
                evaluated_score = self.fabric.broadcast(evaluated_score, src=0)
                if evaluated_score is not None:
                    if (
                        self.best_evaluated_score is None
                        or evaluated_score >= self.best_evaluated_score
                    ):
                        self.best_evaluated_score = evaluated_score
                        self.save(new_high_score=True)
                training_log_dict.update(eval_log_dict)

            # Logging
            self.post_epoch_logging(training_log_dict)
            self.env.on_epoch_end(self.current_epoch)

            # Check for termination signal
            if self.should_stop:
                self.save()
                return

        # --- End of Training Loop ---
        self.time_report.report()
        self.save()
        self.fabric.call("on_fit_end", self)

    # -----------------------------
    # Environment Interaction Helpers
    # -----------------------------
    def handle_reset(self, done_indices=None):
        if self.force_full_restart:
            done_indices = None
            self.force_full_restart = False
        obs = self.env.reset(done_indices)
        return obs

    def env_step(self, actions):
        obs, rewards, dones, extras = self.env.step(actions)
        rewards = rewards * self.task_reward_w
        terminated = extras["terminate"]
        return obs, rewards, dones, terminated, extras

    def post_train_env_step(self, rewards, dones, done_indices, extras, step):
        self.current_rewards += rewards
        self.current_lengths += 1

        self.episode_reward_meter.update(self.current_rewards[done_indices])
        self.episode_length_meter.update(self.current_lengths[done_indices])

        not_dones = 1.0 - dones.float()
        self.current_rewards = self.current_rewards * not_dones
        self.current_lengths = self.current_lengths * not_dones

        self.episode_env_tensors.add(extras["to_log"])

    # -----------------------------
    # Optimization
    # -----------------------------
    def optimize_model(self) -> Dict:
        dataset = self.process_dataset(self.experience_buffer.make_dict())
        self.train()
        training_log_dict = {}

        for batch_idx in track(
            range(self.max_num_batches()),
            description=f"Epoch {self.current_epoch}, training...",
        ):
            iter_log_dict = {}
            dataset_idx = batch_idx % len(dataset)

            # Reshuffle dataset at the beginning of each mini epoch if configured.
            if dataset_idx == 0 and batch_idx != 0 and dataset.do_shuffle:
                dataset.shuffle()
            batch_dict = dataset[dataset_idx]

            # Check for NaNs in the batch.
            for key in batch_dict.keys():
                if torch.isnan(batch_dict[key]).any():
                    print(f"NaN in {key}: {batch_dict[key]}")
                    raise ValueError("NaN in training")

            # Update actor
            actor_loss, actor_loss_dict = self.actor_step(batch_dict)
            iter_log_dict.update(actor_loss_dict)
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(actor_loss)
            actor_grad_clip_dict = self.handle_model_grad_clipping(
                self.model._actor, self.actor_optimizer, "actor"
            )
            iter_log_dict.update(actor_grad_clip_dict)
            self.actor_optimizer.step()

            # Update critic
            critic_loss, critic_loss_dict = self.critic_step(batch_dict)
            iter_log_dict.update(critic_loss_dict)
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(critic_loss)
            critic_grad_clip_dict = self.handle_model_grad_clipping(
                self.model._critic, self.critic_optimizer, "critic"
            )
            iter_log_dict.update(critic_grad_clip_dict)
            self.critic_optimizer.step()

            # Extra optimization steps if needed.
            extra_opt_steps_dict = self.extra_optimization_steps(batch_dict, batch_idx)
            iter_log_dict.update(extra_opt_steps_dict)

            for k, v in iter_log_dict.items():
                if k in training_log_dict:
                    training_log_dict[k][0] += v
                    training_log_dict[k][1] += 1
                else:
                    training_log_dict[k] = [v, 1]

        for k, v in training_log_dict.items():
            training_log_dict[k] = v[0] / v[1]

        self.eval()
        return training_log_dict

    def actor_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        dist = self.model._actor(batch_dict)
        
        # if has logstd
        if hasattr(self.model._actor, "logstd"):
            logstd = self.model._actor.logstd
            std = torch.exp(logstd)
            neglogp = self.model.neglogp(batch_dict["actions"], dist.mean, std, logstd)
        else:
            neglogp = -dist.log_prob(batch_dict["actions"]).sum(dim=-1)

        # Compute probability ratio between new and old policy.
        ratio = torch.exp(batch_dict["neglogp"] - neglogp)
        surr1 = batch_dict["advantages"] * ratio
        surr2 = batch_dict["advantages"] * torch.clamp(
            ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
        )
        ppo_loss = torch.max(-surr1, -surr2)
        clipped = torch.abs(ratio - 1.0) > self.e_clip
        clipped = clipped.detach().float().mean()

        if self.config.bounds_loss_coef > 0:
            b_loss: Tensor = bounds_loss(dist.mean) * self.config.bounds_loss_coef
        else:
            b_loss = torch.zeros(self.num_envs, device=self.device)

        actor_ppo_loss = ppo_loss.mean()
        b_loss = b_loss.mean()
        extra_loss, extra_actor_log_dict = self.calculate_extra_actor_loss(batch_dict, dist)
        actor_loss = actor_ppo_loss + b_loss + extra_loss

        log_dict = {
            "actor/ppo_loss": actor_ppo_loss.detach(),
            "actor/bounds_loss": b_loss.detach(),
            "actor/extra_loss": extra_loss.detach(),
            "actor/clip_frac": clipped.detach(),
            "losses/actor_loss": actor_loss.detach(),
        }
        log_dict.update(extra_actor_log_dict)
        return actor_loss, log_dict

    def calculate_extra_actor_loss(self, batch_dict, dist) -> Tuple[Tensor, Dict]:
        return torch.tensor(0.0, device=self.device), {}

    def critic_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        values = self.model._critic(batch_dict).flatten()
        if self.config.clip_critic_loss:
            critic_loss_unclipped = (values - batch_dict["returns"]).pow(2)
            v_clipped = batch_dict["values"] + torch.clamp(
                values - batch_dict["values"],
                -self.config.e_clip,
                self.config.e_clip,
            )
            critic_loss_clipped = (v_clipped - batch_dict["returns"]).pow(2)
            critic_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
            critic_loss = 0.5 * critic_loss_max.mean()
        else:
            critic_loss = 0.5 * (batch_dict["returns"] - values).pow(2).mean()
        log_dict = {"losses/critic_loss": critic_loss.detach()}
        return critic_loss, log_dict

    def handle_model_grad_clipping(self, model, optimizer, model_name):
        params = get_params(list(model.parameters()))
        grad_norm_before_clip = torch_utils.grad_norm(params)
        if self.config.check_grad_mag:
            bad_grads = (
                torch.isnan(grad_norm_before_clip) or grad_norm_before_clip > 1000000.0
            )
        else:
            bad_grads = torch.isnan(grad_norm_before_clip)

        bad_grads_count = 0
        if bad_grads:
            if self.config.fail_on_bad_grads:
                all_params = torch.cat(
                    [p.grad.view(-1) for p in params if p.grad is not None],
                    dim=0,
                )
                raise ValueError(
                    f"NaN gradient in {model_name}"
                    + f" {all_params.isfinite().logical_not().float().mean().item()}"
                    + f" {all_params.abs().min().item()}"
                    + f" {all_params.abs().max().item()}"
                    + f" {grad_norm_before_clip.item()}"
                )
            else:
                bad_grads_count = 1
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()

        if self.config.gradient_clip_val > 0:
            self.fabric.clip_gradients(
                model,
                optimizer,
                max_norm=self.config.gradient_clip_val,
                error_if_nonfinite=True,
            )
        grad_norm_after_clip = torch_utils.grad_norm(params)
        clip_dict = {
            f"{model_name}/grad_norm_before_clip": grad_norm_before_clip.detach(),
            f"{model_name}/grad_norm_after_clip": grad_norm_after_clip.detach(),
            f"{model_name}/bad_grads_count": bad_grads_count,
        }
        return clip_dict

    # -----------------------------
    # Evaluation and Logging
    # -----------------------------
    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
        return {}, None

    @torch.no_grad()
    def evaluate_policy(self):
        self.eval()
        done_indices = None  # Force reset on first entry
        step = 0
        obs_hist = []
        while self.config.max_eval_steps is None or step < self.config.max_eval_steps:
            obs = self.handle_reset(done_indices)
            # Obtain actor predictions
            actions = self.model.act(obs)
            # Step the environment
            obs, rewards, dones, terminated, extras = self.env_step(actions)
            # save plot
            if self.env.simulator._user_is_recording and "real_self_obs" in obs:
                obs_hist.append(obs["real_self_obs"][0].cpu().detach().numpy())
                print("recording step obs")
            if not self.env.simulator._user_is_recording and len(obs_hist) > 0:
                # save plot to file
                import numpy as np
                from matplotlib import pyplot as plt
                plot_save_path = self.env.simulator._curr_user_recording_name
                obs_hist = np.array(obs_hist)
                fig, axes = plt.subplots(9, 9, figsize=(32, 36))
                for i in range(obs_hist.shape[1]):
                    axes[i // 9, i % 9].plot(obs_hist[:, i])
                plt.tight_layout()
                plt.savefig(f"{plot_save_path}_obs.png")
                print(f"Obs plot saves to {plot_save_path}_obs.png")
                obs_hist = []
            # save plot 
            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
            step += 1
        

    def post_epoch_logging(self, training_log_dict: Dict):
        end_time = time.time()
        log_dict = {
            "info/episode_length": self.episode_length_meter.get_mean().item(),
            "info/episode_reward": self.episode_reward_meter.get_mean().item(),
            "info/frames": torch.tensor(self.step_count),
            "info/gframes": torch.tensor(self.step_count / (10**9)),
            "times/fps_last_epoch": (self.num_steps * self.get_step_count_increment())
            / (end_time - self.epoch_start_time),
            "times/fps_total": self.step_count / (end_time - self.fit_start_time),
            "times/training_hours": (end_time - self.fit_start_time) / 3600,
            "times/training_minutes": (end_time - self.fit_start_time) / 60,
            "times/last_epoch_seconds": (end_time - self.epoch_start_time),
            "rewards/task_rewards": self.experience_buffer.rewards.mean().item(),
            "rewards/extra_rewards": self.experience_buffer.extra_rewards.mean().item(),
            "rewards/total_rewards": self.experience_buffer.total_rewards.mean().item(),
        }
        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"env/{k}": v for k, v in env_log_dict.items()}
        if len(env_log_dict) > 0:
            log_dict.update(env_log_dict)
        log_dict.update(training_log_dict)

        # --- PLR Logging ---
        if self.plr_enabled:
            log_dict['plr/mean_score'] = self.motion_scores.mean().item()
            log_dict['plr/max_score'] = self.motion_scores.max().item()
            log_dict['plr/min_score'] = self.motion_scores.min().item()
            log_dict['plr/mean_staleness'] = self.motion_staleness.mean().item()
            log_dict['plr/max_staleness'] = self.motion_staleness.max().item()

            # Log sampling distribution entropy
            current_weights = self.env.motion_manager.motion_weights # Get current weights
            entropy = -(current_weights * torch.log(current_weights + 1e-9)).sum()
            log_dict['plr/sampling_entropy'] = entropy.item()

            # --- Add Top K Weights to WandB Log ---
            # Get top k weights again (or store from _update_motion_sampling_weights if needed)
            top_k_log_data = self._log_top_k_weights(current_weights, k=5) # Recalculate for logging
            log_dict.update(top_k_log_data) # Add top_k data to the main log dict

        # Log combined dictionary
        self.fabric.log_dict(log_dict, step=self.current_epoch) # Ensure logging uses the correct step

    # -----------------------------
    # Helper Functions
    # -----------------------------
    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    @torch.no_grad()
    def calculate_extra_reward(self):
        return torch.zeros(self.num_steps, self.num_envs, device=self.device)

    def max_num_batches(self):
        return math.ceil(
            self.num_envs * self.num_steps * self.num_mini_epochs / self.config.batch_size
        )

    def get_step_count_increment(self):
        return self.num_envs * self.fabric.world_size  # fabric.world_size = num gpu * num nodes

    def extra_optimization_steps(self, batch_dict, batch_idx: int):
        return {}

    def terminate_early(self):
        self._should_stop = True

    @torch.no_grad()
    def process_dataset(self, dataset):
        if self.config.normalize_values:
            self.running_val_norm.update(dataset["values"])
            self.running_val_norm.update(dataset["returns"])

            dataset["values"] = self.running_val_norm.normalize(dataset["values"])
            dataset["returns"] = self.running_val_norm.normalize(dataset["returns"])

        dataset = DictDataset(self.config.batch_size, dataset, shuffle=True)
        return dataset

    # --- PLR Helper Methods ---

    def _score_transform(self, scores, transform_type, temperature, eps=1e-8):
        """Applies score transformation and temperature."""
        if transform_type == "rank":
            # Higher score = lower rank = higher probability
            sorted_indices = torch.argsort(scores, descending=True)
            ranks = torch.zeros_like(scores, dtype=torch.float32)
            ranks[sorted_indices] = torch.arange(len(scores), device=self.device, dtype=torch.float32)
            transformed_scores = 1.0 / (ranks + 1.0 + eps) # +1 for 1-based rank
        elif transform_type == "power":
            # Ensure scores are non-negative if using power
            transformed_scores = torch.clamp(scores, min=0) + eps
        elif transform_type == "softmax":
            # Softmax handles temperature directly
            if temperature <= 0: temperature = eps
            return torch.softmax(scores / temperature, dim=0)
        else:
            raise ValueError(f"Unknown score transform: {transform_type}")

        # Apply temperature for rank and power
        if temperature <= 0: temperature = eps
        transformed_scores = transformed_scores**(1.0 / temperature)
        return transformed_scores

    def _calculate_plr_weights(self):
        """Calculates the final sampling weights based on scores and staleness."""
        if self.num_motions == 0: return None # Should not happen if init checks pass

        # 1. Calculate Score-based Weights (P_S)
        ps_weights = self._score_transform(
            self.motion_scores,
            self.plr_config.score_transform,
            self.plr_config.temperature
        )
        ps_sum = ps_weights.sum()
        if ps_sum > 1e-8:
            ps_weights /= ps_sum
        else:
            ps_weights = torch.ones_like(ps_weights) / self.num_motions # Fallback to uniform

        # 2. Calculate Staleness-based Weights (P_C)
        pc_weights = torch.zeros_like(ps_weights)
        if self.plr_config.staleness_coef > 0:
            pc_weights = self._score_transform(
                self.motion_staleness,
                self.plr_config.staleness_transform,
                self.plr_config.staleness_temperature
            )
            pc_sum = pc_weights.sum()
            if pc_sum > 1e-8:
                pc_weights /= pc_sum
            else:
                 # If staleness sum is zero (e.g., first step), fallback to uniform staleness
                pc_weights = torch.ones_like(pc_weights) / self.num_motions

        # 3. Mix
        mixed_weights = (1.0 - self.plr_config.staleness_coef) * ps_weights \
                      + self.plr_config.staleness_coef * pc_weights

        # 4. Final Normalization (handle potential floating point inaccuracies)
        mixed_weights = torch.clamp(mixed_weights, min=0) # Ensure non-negative
        final_sum = mixed_weights.sum()
        if final_sum > 1e-8:
            normalized_weights = mixed_weights / final_sum
        else:
            normalized_weights = torch.ones_like(mixed_weights) / self.num_motions # Fallback

        # Ensure no NaNs
        normalized_weights = torch.nan_to_num(normalized_weights, nan=0.0)
        renorm_sum = normalized_weights.sum()
        if renorm_sum > 1e-8:
             normalized_weights = normalized_weights / renorm_sum
        else:
             normalized_weights = torch.ones_like(normalized_weights) / self.num_motions


        return normalized_weights

    def _update_plr_state(self):
        """Updates scores and staleness based on the last completed rollout."""
        if not self.plr_enabled or self.storage is None:
            return

        # --- Update Scores ---
        with torch.no_grad():
            # Ensure returns and advantages are computed
            # Check if required attributes exist on the storage object
            required_attrs = ['advantages', 'returns', 'values', 'rewards', 'motion_ids']
            for attr in required_attrs:
                 if not hasattr(self.storage, attr) or getattr(self.storage, attr) is None:
                      log.warning(f"Attribute '{attr}' not found or is None in storage. Cannot update PLR scores.")
                      return

            # Get relevant data - reshape to (num_steps * num_envs, ...)
            num_steps, num_envs = self.storage.rewards.shape[:2]
            # Check if buffer is filled enough (at least num_steps)
            if num_steps < self.num_steps:
                 log.warning(f"Storage has only {num_steps} steps, expected {self.num_steps}. Skipping PLR update this time.")
                 return

            values = self.storage.values[:num_steps].reshape(-1)
            returns = self.storage.returns[:num_steps].reshape(-1)
            advantages = self.storage.advantages[:num_steps].reshape(-1)

            # Get motion IDs - Assuming storage has 'motion_ids' [steps, envs, 1 or 0 dim]
            # Flatten motion IDs to match other tensors
            motion_ids_flat = self.storage.motion_ids[:num_steps].reshape(-1)


            # Calculate scores per step
            if self.plr_config.strategy == 'value_l1':
                step_scores = (returns - values).abs()
            elif self.plr_config.strategy == 'gae_abs':
                step_scores = advantages.abs()
            else: # Should have been caught in init
                 log.error(f"Invalid PLR strategy: {self.plr_config.strategy}")
                 return # Return early if strategy is invalid

            # Aggregate scores per motion ID using scatter_mean
            # Need unique motion IDs present and their inverse mapping if not using scatter
            # scatter_mean is much cleaner:
            try:
                # Ensure indices are within bounds
                valid_indices = (motion_ids_flat >= 0) & (motion_ids_flat < self.num_motions)
                if not valid_indices.all():
                    log.error(f"Invalid motion IDs detected in PLR update: min={motion_ids_flat.min()}, max={motion_ids_flat.max()}, num_motions={self.num_motions}")
                    # Optionally filter out invalid ones, or just return
                    # motion_ids_flat = motion_ids_flat[valid_indices]
                    # step_scores = step_scores[valid_indices]
                    return # Safer to just skip the update if indices are bad

                batch_motion_scores = scatter_mean(step_scores[valid_indices], motion_ids_flat[valid_indices], dim_size=self.num_motions)
                # Handle motions not present in batch (scatter_mean gives 0, which is fine for EMA)
            except NameError:
                 # Fallback without torch_scatter (less efficient)
                 log.warning("Using fallback for scatter_mean in PLR (torch_scatter not found).")
                 batch_motion_scores = torch.zeros_like(self.motion_scores)
                 unique_ids, inverse_indices = torch.unique(motion_ids_flat, return_inverse=True)
                 for i, motion_id in enumerate(unique_ids):
                      # Ensure motion_id is valid before indexing
                     if motion_id < 0 or motion_id >= self.num_motions:
                         log.error(f"Invalid motion ID {motion_id} during fallback scatter_mean. Skipping.")
                         continue
                     mask = (motion_ids_flat == motion_id)
                     if mask.sum() > 0: # Ensure there are scores to average
                         batch_motion_scores[motion_id] = step_scores[mask].mean()


            # Apply EMA update
            alpha = self.plr_config.ema_alpha
            self.motion_scores = (1.0 - alpha) * self.motion_scores + alpha * batch_motion_scores
            # Ensure scores aren't NaN after update
            self.motion_scores = torch.nan_to_num(self.motion_scores, nan=0.0)


            # --- Store Sampled IDs for Staleness ---
            # Get unique motion IDs sampled in this batch
            # Ensure motion_ids_flat is valid before calling unique
            if motion_ids_flat.numel() > 0:
                self._last_sampled_motion_ids_in_batch = torch.unique(motion_ids_flat)
            else:
                self._last_sampled_motion_ids_in_batch = None


        # --- Update Staleness (Happens *before* next sampling) ---
        if self._last_sampled_motion_ids_in_batch is not None and self._last_sampled_motion_ids_in_batch.numel() > 0:
             self.motion_staleness += 1.0 # Increment staleness for all
             # Ensure indices are valid before using them for assignment
             valid_sampled_ids = self._last_sampled_motion_ids_in_batch[
                 (self._last_sampled_motion_ids_in_batch >= 0) & (self._last_sampled_motion_ids_in_batch < self.num_motions)
             ]
             if valid_sampled_ids.numel() > 0:
                # Reset staleness for motions sampled in the *last* batch
                self.motion_staleness[valid_sampled_ids] = 0.0
             self._last_sampled_motion_ids_in_batch = None # Clear it

    def _update_motion_sampling_weights(self):
        """Calculates and pushes new PLR weights to the MotionManager."""
        if not self.plr_enabled:
            return

        new_weights = self._calculate_plr_weights()
        if new_weights is not None:
            self.env.motion_manager.update_sampling_weights(new_weights)

            # --- Add PLR Logging Here ---
            # Log top K probabilities for inspection
            self._log_top_k_weights(new_weights, k=5) # Log top 5

            # Optional: Log the entropy of the distribution to see how peaky it gets
            # We will log this in post_epoch_logging instead for WandB

    # Add a helper for logging topk
    def _log_top_k_weights(self, weights, k=5):
        """Logs the top k weights and their indices."""
        if k <= 0 or weights is None or len(weights) == 0:
            return {}

        k = min(k, len(weights))
        top_weights, top_indices = torch.topk(weights, k)

        log_data = {}
        for i in range(k):
            log_data[f'plr/top_{i+1}_motion_id'] = top_indices[i].item()
            log_data[f'plr/top_{i+1}_motion_prob'] = top_weights[i].item()

        # Log basic info
        log.info(f"PLR Top {k} Probabilities:")
        for i in range(k):
            log.info(f"  Motion ID {top_indices[i].item()}: {top_weights[i].item():.4e}")

        return log_data
