# ProtoMotions with IsaacGym for Steering Task

This document provides an overview of the basic steering task within the ProtoMotions framework using the IsaacGym simulator.

## 1. Overview

The steering task trains a physics-based character to follow target velocity commands (linear and angular) without relying on reference motion data. The agent learns to walk or run in a specified direction at a specified speed, purely based on task rewards.

**Key Characteristics:**

*   **Task-Driven:** Learning is driven by rewards designed to encourage matching target velocities, maintaining balance, minimizing effort, etc.
*   **No Motion Reference:** Unlike mimicry, this task does not use pre-recorded motion capture data for guidance.
*   **Command Following:** The agent receives target velocity/heading commands as part of its observation and is rewarded for matching them.

## 2. Configuration (`+exp=steering_mlp`)

The `steering_mlp` experiment configuration typically involves:

*   **Environment:** `protomotions.envs.steering.env.SteeringEnv`. This environment defines the core logic for the task.
*   **Agent:** A standard Reinforcement Learning agent like PPO (`protomotions.agents.ppo.agent.PPO`).
*   **Network:** Multi-Layer Perceptron (MLP) networks for the policy (actor) and value function (critic).

## 3. IsaacGym Integration for Steering

The integration with IsaacGym follows the general principles outlined in the main `isaacgym_mimic.md` document, with the primary difference being the environment class used:

*   **Environment:** Instead of `MimicEnv`, the `SteeringEnv` is instantiated.
*   **Simulation Loop:**
    1.  **Observations:** `SteeringEnv` computes observations including the robot's state (joint positions/velocities, root orientation/velocity) and the current target velocity/heading command.
    2.  **Action:** The MLP policy takes these observations and outputs target joint actions.
    3.  **Physics:** `IsaacGymSimulator` applies actions and steps the physics simulation.
    4.  **State Retrieval:** `IsaacGymSimulator` retrieves the updated physical state.
    5.  **Rewards:** `SteeringEnv` calculates rewards based on:
        *   How closely the robot's actual root velocity matches the target command.
        *   Maintaining an upright posture (e.g., penalizing low torso height or excessive tilt).
        *   Potentially penalizing high joint velocities, accelerations, or torques (effort).
        *   Survival rewards.
    6.  **Reset:** Environments are reset based on termination conditions like falling, reaching episode limits, etc.

## 4. Training Command Example

The command provided trains an H1 robot using an MLP policy on the steering task in IsaacGym:

```bash
python protomotions/train_agent.py +exp=steering_mlp +robot=h1 +simulator=isaacgym +experiment_name=h1_steering_test +opt=wandb
```

*   `+exp=steering_mlp`: Selects the steering task configuration.
*   `+robot=h1`: Specifies the Unitree H1 robot.
*   `+simulator=isaacgym`: Uses the IsaacGym backend.
*   `+experiment_name=h1_steering_test`: Defines the output directory for checkpoints and logs (`results/h1_steering_test/`).
*   `+opt=wandb`: Enables logging to Weights & Biases (optional).

## 5. Visualization / Evaluation

To visualize the trained agent's behavior, use the `eval_agent.py` script with the corresponding checkpoint:

```bash
python protomotions/eval_agent.py +robot=h1 +simulator=isaacgym checkpoint=results/h1_steering_test/last.ckpt
```

This command loads the latest checkpoint from the `h1_steering_test` experiment and runs the learned policy in the IsaacGym visualizer. You should see the H1 robot attempting to execute the learned steering behavior.

## 6. Key Code Locations

*   **Steering Environment:** `protomotions/envs/steering/env.py`
*   **IsaacGym Simulator:** `protomotions/simulator/isaacgym/`
*   **Base Environment Logic:** `protomotions/envs/base_env/env.py`
*   **Training Script:** `protomotions/train_agent.py`
*   **Evaluation Script:** `protomotions/eval_agent.py`
*   **Agent Logic (PPO):** `protomotions/agents/ppo/`
*   **Configuration:** `protomotions/config/exp/steering_mlp.yaml`, `protomotions/config/env/steering.yaml`
