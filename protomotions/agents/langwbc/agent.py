import torch
from torch import Tensor
import logging
import time
from lightning.fabric import Fabric
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from typing import Tuple, Dict, Optional, Deque
from pathlib import Path
from rich.progress import track
from collections import deque

# Assuming CLIP and PPOModel are importable or handled elsewhere
try:
    from transformers import CLIPModel, CLIPProcessor, CLIPTextModelWithProjection, AutoTokenizer
except ImportError:
    print("Warning: transformers library not found. CLIP encoding will not work.")
    CLIPModel, CLIPProcessor, CLIPTextModelWithProjection, AutoTokenizer = None, None, None, None


from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.model import PPOModel # Needed for loading teacher
from protomotions.agents.langwbc.model import LangWBCModel
from protomotions.agents.utils.data_utils import DictDataset, ExperienceBuffer # Use existing buffer for simplicity first
from protomotions.envs.base_env.env import BaseEnv # Use BaseEnv for type hint
from protomotions.agents.common.common import weight_init

log = logging.getLogger(__name__)

class LangWBCAgent(PPO):
    # Override type hint for env if needed, e.g., if specific methods are required
    # env: SpecificLangWBCEnv

    def __init__(self, fabric: Fabric, env: BaseEnv, config: DictConfig):
        # Initialize PPO base, but many RL parts won't be used
        super().__init__(fabric, env, config)

        # --- LangWBC Specific Initializations ---
        self.history_len = self.config.history_len
        self.history_interval = self.config.history_interval
        # Use deque for efficient history buffering per environment
        self.obs_history_buffer: Deque[Dict[str, torch.Tensor]] = deque(maxlen=self.history_len * self.history_interval) # Store raw obs
        self.current_history_view = torch.zeros(
            self.num_envs,
            self.history_len * self.config.model.obs_size, # obs_size from model config
            device=self.device,
            dtype=torch.float
        )
        # Need placeholders for text embeddings (how these are provided needs clarification - config? env?)
        self.text_embeddings = torch.zeros(
            self.num_envs,
            self.config.model.text_embedding_dim,
            device=self.device,
            dtype=torch.float
        )
        # --- End LangWBC Specific Initializations ---

        # PPO specific things we disable or don't need for BC student
        self.running_val_norm = None # No value prediction needed
        self.actor_optimizer = None # Single optimizer for CVAE
        self.critic_optimizer = None


    def setup(self):
        """Instantiate Student CVAE model, Teacher model, CLIP, and Optimizer."""
        # --- Instantiate Student CVAE Model ---
        student_model: LangWBCModel = instantiate(self.config.model)
        student_model.apply(weight_init)
        student_optimizer = instantiate(
            self.config.model.optimizer, # Optimizer defined in model config
            params=list(student_model.parameters()),
        )
        self.model, self.optimizer = self.fabric.setup(student_model, student_optimizer)
        # Mark methods used during data collection/inference if needed by Fabric
        self.model.mark_forward_method("forward")
        self.model.mark_forward_method("get_action_and_latent")


        # --- Load Pre-trained Teacher Model ---
        if self.config.teacher_model_path is None:
            raise ValueError("LangWBCAgent requires 'teacher_model_path' in config.")

        teacher_path = Path(self.config.teacher_model_path)
        teacher_config_path = teacher_path / "config.yaml"
        teacher_checkpoint_path = teacher_path / self.config.teacher_checkpoint_name

        if not teacher_config_path.exists() or not teacher_checkpoint_path.exists():
             raise FileNotFoundError(f"Could not find teacher config ({teacher_config_path}) or checkpoint ({teacher_checkpoint_path})")

        log.info(f"Loading teacher config from: {teacher_config_path}")
        teacher_full_config = OmegaConf.load(teacher_config_path)
        # We need the *agent's model* config part from the teacher's config
        teacher_model_config = teacher_full_config.agent.config.model

        log.info(f"Loading teacher model from: {teacher_checkpoint_path}")
        teacher_state_dict = torch.load(teacher_checkpoint_path, map_location="cpu") # Load to CPU first

        # Instantiate the teacher model (likely a PPOModel)
        # Ensure the teacher model's config matches the saved checkpoint structure
        teacher_model_instance: PPOModel = instantiate(teacher_model_config)

        # Setup with Fabric BEFORE loading state_dict if teacher used Fabric
        # If teacher was *not* trained with Fabric, setup might not be needed, but harmless
        self.teacher_model = self.fabric.setup_module(teacher_model_instance)
        # Load the state dict for the 'model' part
        self.teacher_model.load_state_dict(teacher_state_dict["model"])

        # Set teacher to eval mode and freeze parameters
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        log.info("Teacher model loaded and frozen.")

        # --- Load CLIP Model ---
        if CLIPModel is None:
             log.warning("Transformers library not installed. Cannot load CLIP model.")
             self.clip_processor = None
             self.clip_text_model = None
        else:
             log.info(f"Loading CLIP model: {self.config.clip_model_name}")
             # Using AutoTokenizer and CLIPTextModelWithProjection as common choices
             self.clip_processor = AutoTokenizer.from_pretrained(self.config.clip_model_name)
             # Load to device via Fabric setup
             clip_instance = CLIPTextModelWithProjection.from_pretrained(self.config.clip_model_name)
             self.clip_text_model = self.fabric.setup_module(clip_instance)
             self.clip_text_model.eval() # Set to eval mode
             for param in self.clip_text_model.parameters():
                  param.requires_grad = False # Freeze CLIP
             log.info("CLIP model loaded and frozen.")


    def _update_history(self, current_obs: Dict[str, Tensor]):
        """Appends current observation and maintains the history view."""
        # Store only the proprioceptive part needed for the model input
        # Assumes 'self_obs' contains the 90-dim proprioceptive state
        proprio_obs = current_obs.get("self_obs")
        if proprio_obs is None:
             # Adapt this if the key for proprioceptive state is different
             raise KeyError("Expected 'self_obs' key in observations for history.")

        # Append raw obs dict (or just the tensor)
        self.obs_history_buffer.append({'self_obs': proprio_obs.clone()}) # Store necessary parts

        # Rebuild the flattened history view needed for the encoder
        # This could be optimized, but is clear
        if len(self.obs_history_buffer) >= self.history_len * self.history_interval:
             temp_hist = []
             # Sample history at the desired interval
             buffer_list = list(self.obs_history_buffer)
             for i in range(self.history_len):
                 # Get obs from correct time point, sampling backwards with interval
                 idx = len(buffer_list) - 1 - (i * self.history_interval)
                 if idx >= 0:
                      temp_hist.append(buffer_list[idx]['self_obs'])
                 else:
                      # Pad with zeros if history is not full yet
                      temp_hist.append(torch.zeros_like(proprio_obs))

             # Reverse to get chronological order [t-hist_len*interval, ..., t-interval, t] sampled
             # We sample backwards, so the list is [t, t-interval, ..., t-hist*interval]
             # We need it as [t-hist*interval, ..., t-interval, t] for typical RNN/Transformer inputs
             # Or just flatten appropriately for MLP. Let's reverse and flatten.
             temp_hist.reverse()
             self.current_history_view = torch.cat(temp_hist, dim=-1)
        else:
             # History not full yet, maybe pad or handle differently?
             # For now, let's just use zeros implicitly via the initial self.current_history_view
             pass


    @torch.no_grad()
    def _get_text_embeddings(self, text_commands: List[str]) -> torch.Tensor:
        """Encodes a list of text commands using the loaded CLIP model."""
        if self.clip_text_model is None or self.clip_processor is None:
            # Return zeros if CLIP is not available
            log.warning("CLIP model not available, returning zero text embeddings.")
            return torch.zeros(len(text_commands), self.config.model.text_embedding_dim, device=self.device)

        inputs = self.clip_processor(text=text_commands, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.clip_text_model(**inputs)
        text_embeds = outputs.text_embeds # Use text_embeds (projection output)
        # Normalize? Paper doesn't specify, but often done.
        # text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds

    def handle_reset(self, done_indices=None):
        """Reset environment and history buffer."""
        # Reset PPO base stuff (like motion sampling if PLR used, logging meters)
        # Important: PPO handle_reset calls env.reset()
        obs = super(PPO, self).handle_reset(done_indices) # Call grandparent's method if PPO overrides it

        if done_indices is None:
            done_indices = torch.arange(self.num_envs, device=self.device)

        # --- LangWBC: Reset history for environments that are done ---
        # This is tricky with deque. Simplest is to clear and refill on full reset,
        # or handle per-env histories (more complex buffer).
        # For now, let's assume a full reset clears history, or handle approximate history.
        # A more robust way requires per-environment history deques.
        if done_indices.numel() == self.num_envs:
            self.obs_history_buffer.clear()
            self.current_history_view.zero_()
            # Pre-fill history with initial obs? Needs multiple steps.
            # Or accept initial steps have partial history.
        else:
            # Partial resets are harder to manage perfectly with a single deque
            # without masking/per-env logic within the history view.
            # Let's log a warning and proceed with potentially stale history for non-reset envs.
             log.warning("Partial resets detected. History buffer might contain stale data for non-reset envs.")
             # A possible fix: Zero out history for `done_indices` in self.current_history_view,
             # but this requires knowing the structure.

        # --- Get Text Commands and Embeddings ---
        # How text commands are associated with envs needs defining.
        # Example: Assuming env provides text in `extras` or a fixed mapping
        # Let's use a placeholder - replace with actual logic
        text_commands = ["walk forward"] * self.num_envs # Placeholder
        self.text_embeddings = self._get_text_embeddings(text_commands)

        # Update history with the *first* observation after reset
        self._update_history(obs)

        # Add history and text embedding to the observation dict for the model
        obs['obs_history'] = self.current_history_view.clone()
        obs['text_embedding'] = self.text_embeddings.clone()
        # Need current obs separate for decoder
        obs['current_proprio'] = obs['self_obs'].clone()

        return obs

    def env_step(self, actions):
        """Step environment, update history, add history/text to obs."""
        # Call grandparent's env_step to avoid PPO logic if necessary
        next_obs_base, rewards, dones, terminated, extras = super(PPO, self).env_step(actions)

        # --- LangWBC: Update history and add to observation ---
        self._update_history(next_obs_base)

        # Update text commands if they change per step (unlikely but possible)
        # text_commands = ... get from extras or state ...
        # self.text_embeddings = self._get_text_embeddings(text_commands)

        next_obs = next_obs_base
        next_obs['obs_history'] = self.current_history_view.clone()
        next_obs['text_embedding'] = self.text_embeddings.clone()
        next_obs['current_proprio'] = next_obs['self_obs'].clone() # Make sure current obs is available

        return next_obs, rewards, dones, terminated, extras

    def fit(self):
        """Modified training loop for DAgger + CVAE."""
        # --- Setup Experience Buffer ---
        # We need to store: obs_history, text_embedding, current_proprio, teacher_action
        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps).to(self.device)
        # History is large, store flattened
        self.experience_buffer.register_key("obs_history", shape=(self.history_len * self.config.model.obs_size,))
        self.experience_buffer.register_key("text_embedding", shape=(self.config.model.text_embedding_dim,))
        # Store current obs separately for decoder input during training
        self.experience_buffer.register_key("current_proprio", shape=(self.config.model.obs_size,))
        # Store TEACHER actions, not student actions
        self.experience_buffer.register_key("teacher_actions", shape=(self.config.model.action_size,))
        self.experience_buffer.register_key("rewards") # Keep for logging/diagnostics
        self.experience_buffer.register_key("dones", dtype=torch.long)

        # --- Training Loop (adapted from PPO.fit) ---
        done_indices = None
        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        self.fabric.call("on_fit_start", self)

        obs = self.handle_reset(done_indices) # Initial reset includes history update

        while self.current_epoch < self.config.max_epochs:
            self.epoch_start_time = time.time()
            self.eval() # Teacher is always eval, student might be too during collection?

            with torch.no_grad(): # No gradients needed during data collection
                self.fabric.call("before_play_steps", self)
                for step in track(
                    range(self.num_steps),
                    description=f"Epoch {self.current_epoch}, collecting DAgger data...",
                ):
                    # Prepare observation for teacher and student
                    # Teacher needs full state + reference motion (s_t, s^ref_t)
                    # Student needs history + text embedding + current obs
                    # Assuming 'obs' contains all necessary keys after handle_reset/env_step

                    # --- DAgger: Query Teacher ---
                    # Teacher needs reference motion state (s^ref_t).
                    # Environment *must* provide this in 'extras'.
                    if "reference_motion_state" not in obs:
                         # Using 'obs' as a proxy for teacher input s_t (might need adjustment)
                         # if teacher used privileged info.
                         log.warning("No 'reference_motion_state' in obs, teacher query might be inaccurate. Using 'self_obs'.")
                         # This is a placeholder - proper teacher input state construction is needed.
                         teacher_input_state = obs['self_obs'] # Likely incorrect placeholder
                         teacher_reference_state = torch.zeros_like(teacher_input_state) # Placeholder
                    else:
                         # Construct teacher input properly based on its training regime
                         teacher_input_state = obs # Or combine parts of obs
                         teacher_reference_state = obs["reference_motion_state"]

                    # Query teacher - handle potential differences in input format
                    # This assumes teacher_model.act() takes the state dict like PPOModel
                    try:
                        teacher_action = self.teacher_model.act(teacher_input_state, teacher_reference_state) # Hypothetical call signature
                    except Exception as e:
                        log.error(f"Error querying teacher model: {e}. Check teacher input requirements.")
                        # Use zero action as fallback? Or student action? Needs strategy.
                        teacher_action = torch.zeros(self.num_envs, self.config.model.action_size, device=self.device)


                    # --- Store Data for Student Training ---
                    self.experience_buffer.update_data("obs_history", step, obs['obs_history'])
                    self.experience_buffer.update_data("text_embedding", step, obs['text_embedding'])
                    self.experience_buffer.update_data("current_proprio", step, obs['current_proprio'])
                    self.experience_buffer.update_data("teacher_actions", step, teacher_action) # Store TEACHER action

                    # --- Step Environment with STUDENT Action ---
                    # Get student action (using mean for stepping)
                    student_action = self.model.forward(
                        obs['obs_history'], obs['text_embedding'], obs['current_proprio']
                    )

                    # Step env
                    next_obs, rewards, dones, terminated, extras = self.env_step(student_action)

                    # Store rewards/dones for logging
                    self.experience_buffer.update_data("rewards", step, rewards)
                    self.experience_buffer.update_data("dones", step, dones)

                    all_done_indices = dones.nonzero(as_tuple=False)
                    done_indices = all_done_indices.squeeze(-1)

                    # Update logging meters (episode reward/length)
                    self.post_train_env_step(rewards, dones, done_indices, extras, step) # Use PPO's logging method

                    self.step_count += self.get_step_count_increment()
                    obs = next_obs # Prepare for next step

            # --- Optimize Student Model ---
            training_log_dict = self.optimize_model() # Uses the collected buffer
            training_log_dict["epoch"] = self.current_epoch

            # --- PLR (If Enabled) ---
            # Note: PLR logic from PPO base might need adaptation or disabling
            # if PLR is intended for the *student's* learning process.
            # Original PLR updates weights based on student's value estimates/GAE.
            # Here, student doesn't have value head. PLR needs rethinking if used.
            # Let's assume PLR is OFF for LangWBC student by default unless specified.
            # if self.plr_enabled: ... # Add PLR logic if needed

            self.fabric.call("after_train", self)
            self.current_epoch += 1

            # --- Saving and Evaluation ---
            # Use PPO's saving logic. Evaluation needs custom logic if using LangWBC.
            if self.current_epoch % self.config.manual_save_every == 0:
                self.save()

            if (
                self.config.eval_metrics_every is not None
                and self.current_epoch > 0
                and self.current_epoch % self.config.eval_metrics_every == 0
            ):
                 # Default calc_eval_metrics is likely PPO/RL focused.
                 # Need LangWBC-specific evaluation (e.g., tracking error against teacher on test commands).
                 log.warning("Evaluation logic (calc_eval_metrics) needs implementation for LangWBC.")
                 eval_log_dict, evaluated_score = self.calc_eval_metrics() # Placeholder call
                 evaluated_score = self.fabric.broadcast(evaluated_score, src=0)
                 # ... (Saving based on score) ...
                 training_log_dict.update(eval_log_dict)

            self.post_epoch_logging(training_log_dict) # Use PPO's logging
            self.env.on_epoch_end(self.current_epoch)

            if self.should_stop:
                self.save()
                return

        # --- End of Training ---
        self.time_report.report()
        self.save()
        self.fabric.call("on_fit_end", self)


    def optimize_model(self) -> Dict:
        """Optimizes the CVAE student model using the collected experience."""
        # Dataset uses keys: obs_history, text_embedding, current_proprio, teacher_actions
        dataset = DictDataset(self.config.batch_size, self.experience_buffer.make_dict(), shuffle=True)
        self.train() # Set student model to train mode

        training_log_dict = {}
        for batch_idx in track(
            range(self.max_num_batches()), # Reuse PPO's calculation
            description=f"Epoch {self.current_epoch}, training student...",
        ):
            iter_log_dict = {}
            dataset_idx = batch_idx % len(dataset)
            if dataset_idx == 0 and batch_idx != 0 and dataset.do_shuffle:
                dataset.shuffle()
            batch_dict = dataset[dataset_idx]

            # --- CVAE Forward and Loss Calculation ---
            # Model outputs student action, mu, logvar
            student_actions, mu, logvar = self.model.get_action_and_latent(
                batch_dict["obs_history"],
                batch_dict["text_embedding"],
                batch_dict["current_proprio"],
            )

            # 1. Behavior Cloning (BC) Loss
            teacher_actions = batch_dict["teacher_actions"]
            bc_loss = torch.nn.functional.mse_loss(student_actions, teacher_actions)

            # 2. KL Divergence Loss
            kl_loss = self.model.kl_loss(mu, logvar).mean() # Mean over batch

            # Total Loss (Eq. 8 from paper)
            total_loss = bc_loss + self.config.lambda_kl * kl_loss

            iter_log_dict["losses/student_bc_loss"] = bc_loss.detach()
            iter_log_dict["losses/student_kl_loss"] = kl_loss.detach()
            iter_log_dict["losses/student_total_loss"] = total_loss.detach()
            iter_log_dict["info/lambda_kl"] = self.config.lambda_kl

            # --- Optimization Step ---
            self.optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(total_loss)
            # Use PPO's gradient clipping method
            grad_clip_dict = self.handle_model_grad_clipping(
                self.model, self.optimizer, "student_model"
            )
            iter_log_dict.update(grad_clip_dict)
            self.optimizer.step()

            # Aggregate logs
            for k, v in iter_log_dict.items():
                if k in training_log_dict:
                    training_log_dict[k][0] += v
                    training_log_dict[k][1] += 1
                else:
                    training_log_dict[k] = [v, 1]

        # Average logs
        for k, v in training_log_dict.items():
            training_log_dict[k] = v[0] / v[1]

        self.eval() # Set student model back to eval mode
        return training_log_dict


    def get_state_dict(self, state_dict):
        """Save student model and optimizer state."""
        # Overrides PPO state dict saving
        extra_state_dict = {
            "model": self.model.state_dict(), # Student model
            "optimizer": self.optimizer.state_dict(), # Student optimizer
            "epoch": self.current_epoch,
            "step_count": self.step_count,
            "run_start_time": self.fit_start_time,
            "episode_reward_meter": self.episode_reward_meter.state_dict(),
            "episode_length_meter": self.episode_length_meter.state_dict(),
            "best_evaluated_score": self.best_evaluated_score,
            # Add history buffer? Might be large.
        }
        state_dict.update(extra_state_dict)
        return state_dict

    def load_parameters(self, state_dict):
        """Load student model and optimizer state."""
        # Overrides PPO parameter loading
        self.current_epoch = state_dict["epoch"]
        self.step_count = state_dict.get("step_count", self.step_count)
        self.fit_start_time = state_dict.get("run_start_time", self.fit_start_time)
        self.best_evaluated_score = state_dict.get("best_evaluated_score", None)

        self.model.load_state_dict(state_dict["model"]) # Load student model
        self.optimizer.load_state_dict(state_dict["optimizer"]) # Load student optimizer

        # Load logging meters
        self.episode_reward_meter.load_state_dict(state_dict["episode_reward_meter"])
        self.episode_length_meter.load_state_dict(state_dict["episode_length_meter"])
        # Load history buffer if saved?


    # --- Methods from PPO that might need overriding or disabling ---

    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
        """Needs custom implementation for LangWBC evaluation."""
        # Example: Run episodes with fixed text commands, measure tracking error vs teacher.
        log.warning("calc_eval_metrics not implemented for LangWBC. Returning empty dict.")
        # Placeholder - return empty dict and no score
        return {}, None

    # Disable PPO-specific optimization steps if they exist
    def actor_step(self, batch_dict): raise NotImplementedError("LangWBC uses optimize_model directly")
    def critic_step(self, batch_dict): raise NotImplementedError("LangWBC uses optimize_model directly")
    def calculate_extra_reward(self): return torch.zeros(self.num_steps, self.num_envs, device=self.device) # No extra RL rewards
    def process_dataset(self, dataset): return dataset # No value normalization needed


