# ProtoMotions with IsaacGym for Motion Mimicry

This document provides an introduction to using the ProtoMotions framework with the IsaacGym simulator, specifically focusing on motion mimicry tasks like the Full Body Tracker and MaskedMimic.

## 1. Overview

**ProtoMotions** is a framework designed for creating interactive, physics-based virtual characters. It aims to enable agents to learn complex behaviors through reinforcement learning, often by mimicking reference motion data.

**Key Features:**

*   **Simulator Agnosticism:** Supports multiple physics simulators (IsaacGym, IsaacLab, Genesis). The core logic is separated from simulator-specific implementations.
*   **Modularity:** Built with configurable components for environments, agents, rewards, observations, etc.
*   **Configuration:** Heavily relies on [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/) for managing experiments and parameters. This allows for easy composition and modification of settings.

## 2. Installation for IsaacGym

Follow these steps to set up ProtoMotions with IsaacGym (referencing the main `README.md`):

1.  **Install IsaacGym:** Download and install IsaacGym Preview 4 from the NVIDIA website.
    ```bash
    # Example download command (check NVIDIA website for the latest link)
    # wget https://developer.nvidia.com/isaac-gym-preview-4
    # tar -xvzf isaac-gym-preview-4
    ```
2.  **Install IsaacGym Python API:**
    ```bash
    pip install -e isaacgym/python
    ```
3.  **Install ProtoMotions and Dependencies:** From the ProtoMotions repository root:
    ```bash
    pip install -e .
    pip install -r requirements_isaacgym.txt
    pip install -e isaac_utils
    pip install -e poselib
    ```
4.  **Set `PYTHON_PATH` Alias (Optional but Recommended):**
    ```bash
    alias PYTHON_PATH=python
    ```
5.  **Potential Issues:** If you encounter linking errors (`GLIBCXX` issues), try setting the `LD_LIBRARY_PATH`:
    ```bash
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
    ```

## 3. Configuration System (Hydra)

ProtoMotions uses Hydra for configuration.

*   **Config Files:** Configuration files are located in the `protomotions/config/` directory. Key subdirectories include:
    *   `config/exp/`: Defines specific experiments (e.g., `full_body_tracker`, `masked_mimic`).
    *   `config/robot/`: Defines robot parameters (e.g., `smpl.yaml`, `h1.yaml`).
    *   `config/simulator/`: Defines simulator settings (e.g., `isaacgym.yaml`).
    *   `config/agent/`: Defines agent parameters (e.g., PPO settings).
*   **Command-Line Overrides:** Configurations are composed and selected using command-line arguments. The `+` syntax adds or overrides configuration groups.
    *   `+exp=...`: Selects the experiment configuration.
    *   `+robot=...`: Selects the robot model.
    *   `+simulator=isaacgym`: Selects the IsaacGym simulator backend.
    *   `+experiment_name=my_experiment`: Sets the directory name for saving results and checkpoints (`results/my_experiment`). This is crucial for organizing runs and resuming training.
    *   Other parameters (e.g., `num_envs=1024`, `motion_file=...`) can be overridden directly.

## 4. IsaacGym Integration

*   **Simulator Abstraction:** The core training logic in `protomotions/train_agent.py` instantiates the environment and simulator based on the configuration.
*   **IsaacGym Backend:** When `+simulator=isaacgym` is specified:
    *   The `isaacgym` Python package is imported early (handled in `train_agent.py`).
    *   The simulator configuration is loaded from `config/simulator/isaacgym.yaml`.
    *   An instance of `protomotions.simulator.isaacgym.simulator.IsaacGymSimulator` is created. This class wraps the IsaacGym API calls.
*   **`IsaacGymSimulator` Class:** Located in `protomotions/simulator/isaacgym/simulator.py`, this class is responsible for:
    *   Initializing the Gym simulation (`gymapi`).
    *   Creating the simulation planes and environments.
    *   Loading assets (robots, objects) into the simulation.
    *   Setting actor states (positions, rotations, DOF states).
    *   Stepping the physics simulation.
    *   Retrieving simulation data (actor states, contact forces, etc.).
    *   Handling camera and visualization logic.
*   **Environment Interaction:** The environment class (e.g., `protomotions.envs.mimic.env.MimicEnv`) interacts with the `IsaacGymSimulator` instance to apply actions, step physics, and get observations relevant to the task.

## 5. Motion Mimicry Tasks

Motion mimicry involves training an agent to replicate movements from a reference motion dataset (e.g., MoCap data). ProtoMotions provides implementations based on DeepMimic and MaskedMimic concepts.

### 5.1. Full Body Tracker (DeepMimic Style)

*   **Goal:** Train a policy to track a given reference motion as closely as possible using physics simulation.
*   **Experiment Config:** Use configurations under `+exp=full_body_tracker/...` (e.g., `+exp=full_body_tracker/transformer_flat_terrain`).
*   **Key Files:**
    *   `protomotions/envs/mimic/env.py`: Defines `MimicEnv`, which handles loading motion data, calculating observations (like target poses, current poses, velocities), and computing rewards based on tracking error.
    *   `protomotions/envs/mimic/mimic_utils.py`: Contains helper functions for motion processing, state initialization, and reward calculations used by `MimicEnv`.
    *   `protomotions/agents/ppo/agent.py`: The PPO agent optimizes the policy based on rewards from `MimicEnv`.
*   **Input:** Requires a `motion_file` argument pointing to the reference motion data (e.g., `motion_file=data/motions/smpl_humanoid_walk.npy`).
*   **Training:** A single-stage process using PPO.
    ```bash
    PYTHON_PATH protomotions/train_agent.py \
        +exp=full_body_tracker/transformer_flat_terrain \
        +robot=smpl \
        +simulator=isaacgym \
        motion_file=data/motions/smpl_humanoid_walk.npy \
        +experiment_name=smpl_tracker_isaacgym
    ```

### 5.2. MaskedMimic

*   **Goal:** Train a unified, physics-based controller that can generate diverse behaviors by predicting masked portions of future motion, conditioned on the unmasked parts and potentially task objectives. It leverages an expert policy (usually a Full Body Tracker) for supervision.
*   **Experiment Config:** Use configurations under `+exp=masked_mimic/...` (e.g., `+exp=masked_mimic/flat_terrain`).
*   **Key Files:** Similar to the Full Body Tracker, relying heavily on `MimicEnv` and related components, but with different agent configurations and potentially reward structures.
*   **Two-Stage Training:**
    1.  **Train Expert (Full Body Tracker):** First, train a full body tracker as described above. Note the `experiment_name` used.
        ```bash
        # (Run command from section 5.1)
        # Results saved in results/smpl_tracker_isaacgym/
        ```
    2.  **Train MaskedMimic Agent:** Train the MaskedMimic policy, providing the path to the *folder* containing the expert checkpoint.
        ```bash
        PYTHON_PATH protomotions/train_agent.py \
            +exp=masked_mimic/flat_terrain \
            +robot=smpl \
            +simulator=isaacgym \
            motion_file=data/motions/smpl_humanoid_walk.npy \
            agent.config.expert_model_path=results/smpl_tracker_isaacgym \
            +experiment_name=smpl_masked_mimic_isaacgym
        ```
*   **Inference/Evaluation:** Trained MaskedMimic agents can be used for tasks like user control.
    ```bash
    PYTHON_PATH protomotions/eval_agent.py \
        +robot=smpl \
        +simulator=isaacgym \
        +opt=[masked_mimic/tasks/user_control] \
        checkpoint=results/smpl_masked_mimic_isaacgym/last.ckpt
    ```

## 6. Detailed Mimicry Workflow in IsaacGym

This section details the step-by-step process of how the `MimicEnv` interacts with IsaacGym and the motion data to train a mimicry policy.

**Key Classes Involved:**

*   `protomotions.envs.mimic.env.MimicEnv`: The main environment class orchestrating the mimicry task.
*   `protomotions.envs.mimic.components.MimicMotionManager`: Handles loading, sampling, and advancing reference motions.
*   `protomotions.envs.mimic.components.MimicObs`: Calculates standard mimicry observations (current vs. target state).
*   `protomotions.envs.mimic.components.MaskedMimicObs` (Optional): Calculates observations specific to MaskedMimic (masked future poses, conditioning signals).
*   `protomotions.envs.mimic.mimic_utils`: Contains utility functions for reward calculation (`exp_tracking_reward`) and coordinate transformations.
*   `protomotions.simulator.isaacgym.simulator.IsaacGymSimulator`: The wrapper around the IsaacGym API.

**Simulation Step Breakdown:**

1.  **Action Application (`pre_physics_step` in `MimicEnv`):**
    *   The RL agent (e.g., PPO) produces an action based on the current observation.
    *   This action often represents target joint positions or velocities.
    *   If `config.mimic_residual_control` is enabled, the action is interpreted as a residual added to the target pose from the reference motion (`residual_actions_to_actual` in `MimicEnv`).
    *   The environment passes the processed action (e.g., target DOF positions) to the `IsaacGymSimulator`.
    *   `IsaacGymSimulator` applies these actions as DOF targets or forces using IsaacGym API calls (e.g., `set_dof_position_target_tensor`, `set_dof_actuation_force_tensor`).

2.  **Physics Simulation (`step_physics` in `IsaacGymSimulator`):**
    *   The `IsaacGymSimulator` calls the IsaacGym API (`gym.simulate`, `gym.fetch_results`) to advance the physics simulation by one timestep (`dt`).
    *   IsaacGym calculates the resulting character states (positions, velocities, contact forces, etc.) based on the applied actions and physics interactions.

3.  **State Retrieval and Processing (`post_physics_step`):**
    *   `IsaacGymSimulator` retrieves the updated simulation state (DOF state, rigid body states, etc.) from IsaacGym.
    *   `MimicMotionManager` advances the time (`motion_times`) for the reference motion in each environment. It checks if motions have ended or need resetting.
    *   If `MaskedMimic` is enabled, `MaskedMimicObs` updates its internal state (e.g., shifting future pose masks, handling time gaps) in its `post_physics_step`.
    *   Standard environment updates (e.g., updating observation history buffers in `SelfObservation`) occur.

4.  **Observation Calculation (`compute_observations`, called before agent acts):**
    *   The environment requests the current physics state (e.g., `get_dof_state`, `get_bodies_state`) from the `IsaacGymSimulator`.
    *   `MimicMotionManager` provides the current reference motion IDs (`motion_ids`) and times (`motion_times`).
    *   The `MotionLib` (managed by `MimicMotionManager`) is queried to get the target reference state (poses, velocities) corresponding to the current `motion_ids` and `motion_times`.
    *   `MimicObs` calculates observations:
        *   Current character state (root pose, joint rotations, velocities) often transformed into a root-relative frame.
        *   Target character state from the reference motion, also often in a root-relative frame.
        *   Phase information (e.g., sine/cosine encoding of the motion time).
    *   If `MaskedMimic` is enabled, `MaskedMimicObs` calculates additional observations:
        *   Sparse future target poses relative to the current pose.
        *   Masks indicating which future joints/poses are visible/conditioned upon.
        *   Historical pose observations.
        *   Conditioning signals like text embeddings or far-future target pose information, along with their respective masks.
    *   These observations are compiled into a dictionary and returned.

5.  **Reward Calculation (`compute_reward` in `MimicEnv`):**
    *   Retrieves the current physics state (global translation `gt`, global rotation `gr`, velocities `gv`/`gav`, DOF velocities `dv`) from the simulator.
    *   Retrieves the corresponding reference motion state (`ref_gt`, `ref_gr`, etc.) from the `MotionLib` via the `MimicMotionManager`.
    *   Transforms states as needed (e.g., calculating root-relative key body positions `kb`, converting DOF positions to local rotations `lr`). Coordinates are often adjusted to account for respawn offsets and terrain height (`respawn_offset_relative_to_data`).
    *   Calls `exp_tracking_reward` (from `mimic_utils`) which calculates exponential rewards based on the difference between the current and reference states for various components:
        *   Key body position error (`kb_pos_rew`).
        *   Root position/rotation error (`root_pos_rew`, `root_rot_rew`).
        *   Local joint rotation error (`joint_rot_rew`).
        *   Velocity errors (root, body, DOF: `root_vel_rew`, `root_ang_vel_rew`, `joint_vel_rew`).
        *   End-effector position error (`ee_pos_rew`).
    *   Calculates penalties (e.g., action rate, power consumption `pow_rew`).
    *   Sums the weighted reward components to get the final reward (`rew_buf`).
    *   Logs various raw and scaled reward components and error metrics (`log_dict`, `mimic_info_dict`).

6.  **Reset Calculation (`compute_reset`):**
    *   Checks for standard termination conditions (e.g., maximum episode length, root height limits).
    *   Checks if the reference motion has ended (`motion_manager.get_done_tracks`).
    *   Checks for early termination based on poor tracking performance (`config.mimic_early_termination`), comparing metrics stored in `mimic_info_dict` against thresholds. Early termination is often ignored during a grace period after a reset or on complex terrain where tracking is harder.
    *   Sets `reset_buf` and `terminate_buf` for environments that need resetting.

7.  **Reset Execution (`reset`):**
    *   For environments flagged in `reset_buf`:
        *   `MimicMotionManager` samples a new motion and start time.
        *   `MaskedMimicObs` (if enabled) resets its tracking state (`reset_track`).
        *   The environment determines a new spawn location, potentially considering terrain and scene requirements (`get_envs_respawn_position`). This also calculates `respawn_offset_relative_to_data`.
        *   The environment calls `IsaacGymSimulator` to reset the actor states (root pose, DOF positions/velocities) to the initial state of the new reference motion using IsaacGym API calls (`set_actor_root_state_tensor_indexed`, `set_dof_state_tensor_indexed`).

This cycle repeats, allowing the RL agent to learn a policy that minimizes the tracking error defined by the reward function by controlling the character within the IsaacGym physics simulation.

## 7. Training and Evaluation Flow

*   **Training:** Initiated via `protomotions/train_agent.py`.
    *   Loads configuration using Hydra based on command-line args.
    *   Instantiates Fabric (for multi-GPU/distributed training).
    *   Instantiates the simulator (e.g., `IsaacGymSimulator`).
    *   Instantiates the environment (e.g., `MimicEnv`).
    *   Instantiates the agent (e.g., `PPO`).
    *   Loads checkpoints if resuming (`config.checkpoint` or checks `results/<experiment_name>/last.ckpt`).
    *   Starts the training loop (`agent.fit()`), which involves collecting experience from the environment and updating the policy.
    *   Saves checkpoints periodically and logs results (e.g., to WandB if `+opt=wandb` is used).
*   **Evaluation:** Initiated via `protomotions/eval_agent.py`.
    *   Loads configuration (often inferred from the checkpoint or specified via command line).
    *   Instantiates the simulator and environment.
    *   Loads the trained agent policy from the specified `checkpoint`.
    *   Runs the policy in the environment for visualization and interaction (keyboard controls available).

## 8. Data Preparation

Motion mimicry relies on motion capture data. The general process (detailed in `README.md`) is:

1.  **Download:** Obtain raw motion data (e.g., AMASS) and SMPL/SMPL-X models.
2.  **Convert:** Use scripts like `data/scripts/convert_amass_to_isaac.py` to convert data into a format usable by PoseLib (used internally). Retargeting to specific robots (like H1) might be needed (`--robot-type=h1 --force-retarget`).
3.  **YAML Definition:** Create `.yaml` files (like those in `data/yaml_files/`) to list motions, their FPS, and potentially labels. Scripts like `data/scripts/create_motion_fps_yaml.py` can help.
4.  **Package:** Pre-process the data using `data/scripts/package_motion_lib.py` into `.pt` files for faster loading during training. This script takes the YAML file and the motion data directory as input.

The `motion_file` argument in training commands usually points to either a single converted `.npy` file or a packaged `.pt` file generated from a YAML definition.

## 9. Key Code Locations Summary

*   **IsaacGym Simulator:** `protomotions/simulator/isaacgym/`
*   **Mimicry Environment:** `protomotions/envs/mimic/`
*   **Training Script:** `protomotions/train_agent.py`
*   **Evaluation Script:** `protomotions/eval_agent.py`
*   **Agent Logic (PPO):** `protomotions/agents/ppo/`
*   **Configuration:** `protomotions/config/`
*   **Data Processing:** `data/scripts/`

This document should serve as a starting point for understanding how ProtoMotions leverages IsaacGym for motion mimicry tasks. Refer to the specific code files and the main `README.md` for more in-depth details.
