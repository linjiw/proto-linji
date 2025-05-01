# MaskedMimic Training Process

MaskedMimic is a technique for training versatile, physics-based character controllers. It involves a two-stage training process:

1.  **Stage 1: Full Body Tracker (Expert Policy Training):** Train a policy to accurately mimic a large dataset of motion capture (mocap) data. This policy becomes the "expert".
2.  **Stage 2: MaskedMimic Policy Training:** Train the final policy to *imitate the actions* of the Stage 1 expert, but using sparse, masked future pose information and optional text descriptions as input.

This document focuses primarily on understanding **Stage 1**.

## Stage 1: Full Body Tracker (Expert Policy)

**Goal:** To train a robust reinforcement learning (RL) agent that can make a physically simulated character follow diverse reference motion clips as closely as possible. This is heavily inspired by the [DeepMimic](https://github.com/xbpeng/DeepMimic) approach.

**How it Works (Mechanics):**

1.  **RL Setup:**
    *   **Agent:** Uses Proximal Policy Optimization (PPO).
    *   **Environment:** A physics simulation (IsaacGym in this case) containing multiple instances of the character (`+robot=smpl`). Each instance runs in parallel.
    *   **Policy Network:** Takes observations from the environment and outputs actions.

2.  **Observations:** At each simulation step, the policy receives information about:
    *   **Current Character State:** Joint positions, joint velocities, root position/rotation relative to the terrain, root linear/angular velocities, potentially end-effector positions/velocities. These are typically represented in a character-centric coordinate frame.
    *   **Target Motion State:** Crucially, it also receives the target state (joint rotations, root position/rotation, velocities) from the *reference motion clip* for the *next* simulation timestep (`t+1`). This gives the policy a target to aim for.

3.  **Actions:** The policy outputs target joint rotations (or PD gains/torques depending on the configuration) for the character's actuators (motors). The simulation applies these actions, resulting in the character's movement.

4.  **Reward Function:** This is key to driving the mimicking behavior. The agent receives rewards based on how closely its *current* state matches the *reference motion's* state at the *current* time `t`:
    *   **Joint Rotation Similarity:** Rewards matching the reference joint rotations (e.g., using quaternion distance).
    *   **Root State Similarity:** Rewards matching the reference root position and orientation.
    *   **Velocity Similarity:** Rewards matching reference joint angular velocities and root linear/angular velocities.
    *   **End-Effector Similarity:** Rewards matching the reference positions of key body parts like hands and feet.
    *   **Center of Mass (CoM) Similarity:** Rewards matching the reference CoM position and velocity.
    *   (The exact components and weights are defined in the configuration files).

5.  **Learning:** PPO uses the rewards collected over time to update the policy network, making it progressively better at generating actions that lead to states closely matching the reference motion, thereby maximizing the cumulative reward.

**Motion Scheduling and Sampling:**

How does the agent learn from *many* different motions?

1.  **Motion Library:** The training process uses a collection of motion clips defined by the `motion_file` argument. This can be:
    *   A YAML file (like `data/yaml_files/hml3d_smpl.yaml`) that lists individual `.npy` motion files and their properties (FPS, sub-segments, labels).
    *   A pre-packaged `.pt` file (created by `package_motion_lib.py`) for faster loading.

2.  **Environment Resets:** In the multi-environment IsaacGym setup, each environment simulates one character independently. When an episode in a specific environment ends (e.g., the character falls, deviates too much, reaches the end of its current motion clip, or hits a time limit), that environment resets.

3.  **Random Motion Sampling:** Upon reset, the environment **randomly samples a new motion clip** (and potentially a starting time within that clip) from the motion library (`MotionLib`).

4.  **Episode Trajectory:** The character in that environment then attempts to mimic this newly sampled motion clip from the chosen starting time. The reference motion time advances frame-by-frame along with the simulation time for that environment.

5.  **Variety:** This constant resetting and random resampling across hundreds or thousands of parallel environments ensures that the policy is trained to track a wide variety of motions present in the dataset, rather than overfitting to just one or a few.

**Training Command (Recap):**

```bash
PYTHON_PATH protomotions/train_agent.py \
    +exp=full_body_tracker/transformer_flat_terrain \
    +robot=smpl \
    +simulator=isaacgym \
    motion_file=data/yaml_files/hml3d_smpl.yaml \
    +experiment_name=my_smpl_full_body_tracker_phase1
    # Optional: Add 'task.motion_data_dir=/path/to/npy/files' if needed
```

## Stage 2: MaskedMimic Policy

**Goal:** To train a policy that can generate diverse and controllable behaviors based on *sparse* future information and potentially text commands, by learning to predict the *actions* of the Stage 1 expert.

**Key Differences from Stage 1:**

*   **Input/Observations:** Instead of seeing the dense target state for `t+1`, the MaskedMimic policy sees:
    *   The character's current state (similar to Stage 1).
    *   *Masked* future target poses at various time steps ahead (e.g., `t+10`, `t+30`, `t+60`). "Masked" means only *some* joints' target positions/rotations might be provided, indicated by accompanying mask tensors. (See `MaskedMimicObs` component).
    *   Optional text embeddings corresponding to the motion description.
    *   Historical observations of the character's own past states.
*   **Learning Target:** It's trained to predict the *actions* that the previously trained Stage 1 expert policy *would have taken* given the *full, unmasked* state information.
*   **Reward:** Typically uses L1 or L2 loss between the MaskedMimic policy's predicted action and the Stage 1 expert's actual action.

**Training Command (Example):**

```bash
PYTHON_PATH protomotions/train_agent.py \
    +exp=masked_mimic/flat_terrain \
    +robot=smpl \
    +simulator=isaacgym \
    motion_file=data/yaml_files/hml3d_smpl.yaml \
    agent.config.expert_model_path=results/my_smpl_full_body_tracker_phase1/ \
    +experiment_name=my_smpl_masked_mimic_phase2
    # Optional: Add 'task.motion_data_dir=/path/to/npy/files' if needed
```
*Note the `agent.config.expert_model_path` pointing to the Stage 1 results folder.*

## Why Two Stages?

*   **Stable Target:** The Stage 1 expert provides a consistent, high-quality target (expert actions) for Stage 2 training. Directly learning complex behaviors from sparse inputs can be unstable.
*   **Generalization:** Stage 2 learns to achieve similar results as the expert but using less information (sparse/masked inputs), leading to a more general and controllable final policy.
