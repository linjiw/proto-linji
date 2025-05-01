Introduction to Prioritized Level Replay (PLR)
Reinforcement learning (RL) agents often struggle to generalize beyond their training experiences, especially in environments with procedurally generated content (PCG) where each "level" is a unique instance. Standard training methods typically sample these levels uniformly. However, the learning potential of a level depends on the agent's current policy.

Prioritized Level Replay (PLR) is a framework designed to improve sample efficiency and generalization in RL by moving away from uniform level sampling. Instead of picking the next training level randomly, PLR selectively samples levels, prioritizing those estimated to offer the highest learning potential if revisited. It essentially creates an emergent curriculum by focusing on levels that are appropriately challenging for the agent's current capabilities.

PLR is designed to be a general method that can be combined with various RL algorithms and doesn't require control over the level generation process itself, only the ability to replay a specific level using its identifier (like a seed).

How PLR Works
PLR modifies the experience collection part of the RL training loop. Here's a breakdown of its core mechanics:

Level Sampling Decision
At the start of collecting new experiences, PLR decides whether to:

Sample a New, Unseen Level: Pick a level from the training set that the agent hasn't encountered before. The probability of doing this often decreases as more levels are seen.
Replay a Seen Level: Select a level the agent has already visited, sampled from a special "replay distribution" $P_{replay}$.
The Replay Distribution ($P_{replay}$)
This distribution determines which previously seen level to replay. It's a mix of two components:

Score-based Prioritization ($P_S$): Prioritizes levels based on their estimated future learning potential. Levels with higher scores are more likely to be selected.
Staleness-based Prioritization ($P_C$): Prioritizes levels that haven't been replayed recently. This prevents scores from becoming too "off-policy" as the agent learns and ensures diversity.
The final probability is a weighted sum: $P_{replay} = (1−ρ) \cdot P_S + ρ \cdot P_C$, where $ρ$ is a "staleness coefficient" hyperparameter.

Scoring Levels ($S_i$)
After the agent completes an episode (or trajectory segment) $τ$ on a level $l_i$, PLR calculates a score $S_i$ to estimate the learning potential of replaying that level.

TD-Errors as Proxy: The core idea is to use Temporal Difference (TD) errors as a proxy for learning potential. High magnitude TD-errors suggest a discrepancy between the agent's value estimates and actual outcomes, indicating room for learning.
GAE Magnitude / L1 Value Loss: The paper finds that using the average magnitude of the Generalized Advantage Estimate (GAE) over the trajectory is particularly effective. This is often equivalent to the L1 value loss used in algorithms like PPO. The formula is:
$$S_i = \frac{1}{T} \sum_{t=0}^{T} \left|\sum_{k=t}^{T} (γλ)^{k−t} δ_k\right|$$

Value Correction Hypothesis: In sparse reward settings, prioritizing levels with high value loss guides the agent towards "threshold levels" – those at the edge of its current capabilities – creating a natural curriculum.
Prioritization Schemes
How scores translate into probabilities in $P_S$:

Rank Prioritization: Probabilities are based on the rank of a level's score ($h(S_i) = \frac{1}{rank(S_i)}$). This is often found to be more stable than using the raw scores directly.
Proportional Prioritization: Probabilities are directly proportional to the scores ($h(S_i) = S_i$).
A temperature parameter $β$ can tune the sharpness of the distribution $P_S(l_i) \propto h(S_i)^{1/β}$.

Staleness Calculation ($P_C$)
This assigns probability mass proportional to how long ago a level was last sampled ($P_C(l_i) \propto c−C_i$), where $c$ is the current global episode count and $C_i$ is the episode count when level $l_i$ was last sampled.

Organization into "Code" (Based on Pseudocode)
While I cannot generate runnable Python code, the paper provides algorithms that show how PLR integrates into a typical RL loop.

Algorithm 1: Policy-Gradient Training Loop with PLR
Initialization: - Initialize policy $π_θ$. - Initialize empty lists/arrays for seen levels ($Λ_{seen}$), their scores ($S$), and last-visited timestamps ($C$). - Initialize global episode counter $c$.

Training Loop: 1. Experience Collection: Instead of standard sampling, call a function like collect_experiences (which uses Algorithm 2 internally) to gather a batch of trajectories $B$. 2. Policy Update: Update the policy $θ$ using the collected batch $B$ (e.g., using PPO).

Algorithm 2: Experience Collection with PLR
Input: Training levels, $Λ_{seen}$, policy $π$, scores $S$, timestamps $C$, counter $c$.
Output: A sampled trajectory $τ$.

Steps: 1. Increment global episode counter $c$. 2. Decide Replay vs. New: Sample $d \sim P_D(d)$. - If New Level ($d=0$ and unseen levels exist): - Sample $l_i$ uniformly from unseen levels ($Λ_{train} \setminus Λ_{seen}$). - Add $l_i$ to $Λ_{seen}$, initialize its score $S_i$ and timestamp $C_i$. - Else (Replay Level): - Calculate $P_S$ using current scores $S$ and prioritization method (e.g., rank-based).
- Calculate $P_C$ using timestamps $C$ and current count $c$.
- Form $P_{replay} = (1−ρ)P_S + ρP_C$.
- Sample level $l_i \sim P_{replay}$. 3. Sample Trajectory: Collect trajectory $τ$ by running policy $π$ on level $l_i$. 4. Update Score & Timestamp: Calculate the score $S_i = score(τ,π)$ (e.g., using Eq. 2). Update timestamp $C_i ← c$.
5. Return $τ$.

(Note: Algorithms 3 & 4 in the paper adapt this for T-step rollouts instead of full episodes).

Plug-and-Play Module for Curriculum RL
PLR acts as a "plug-and-play" module by replacing the standard level sampling mechanism within an existing RL training framework.

Integration Point: It primarily modifies the part of the code responsible for deciding which environment instance (level) to run the policy in to collect the next batch of experiences.

Requirements:
Identifiable Levels: The environment must allow specifying and resetting to distinct levels using an identifier (e.g., a seed).
Shared Dynamics: Levels should share underlying dynamics or structure so learning on one can generalize to others.
Scoring Mechanism: The RL algorithm needs to provide the necessary information to compute the chosen score (e.g., value estimates $V(s)$ to calculate TD-errors or GAE). PPO with GAE is a common choice compatible with the L1 value loss score.
Implementation:
Maintain the data structures: $Λ_{seen}$ (list of seen level IDs), $S$ (list of scores), $C$ (list of timestamps).
Implement the sampleNextLevel logic from Algorithm 2/4. This involves calculating $P_S$, $P_C$, mixing them into $P_{replay}$, sampling a level ID, and potentially adding new levels to the tracked lists.
Implement the chosen score function (e.g., Eq. 2 using GAE magnitudes calculated during the policy update phase).
Modify the training loop to call sampleNextLevel when a new episode/rollout begins in an environment worker/instance.
Update the score $S_i$ and timestamp $C_i$ for the level $l_i$ after its trajectory $τ$ is collected.
By selectively sampling levels based on learning potential and staleness, PLR adaptively focuses the agent's training on the most informative levels at any given time, effectively inducing a curriculum without needing explicit difficulty definitions.

Prioritized Level Replay (PLR)
Understanding Prioritized Level Replay
The Problem
Standard Reinforcement Learning (RL) often struggles with generalization, especially in procedurally generated environments where each level/task instance is unique (like Procgen or MiniGrid).
Typically, training levels are sampled uniformly at random. However, not all levels offer the same learning value at different stages of training. Some might be too easy, some too hard.
The Core Idea
PLR aims to improve sample efficiency and generalization by sampling levels non-uniformly.
It prioritizes revisiting levels that are estimated to have the highest learning potential for the agent's current skill level.
This creates an emergent curriculum: the agent focuses on levels that are "just right" – challenging but not impossible – adapting the difficulty as it learns.
Crucially, PLR doesn't need to modify the level generation process itself, only the ability to replay a level given its identifier (e.g., a seed).
How it Works (Mechanics)
Integration Point: PLR primarily modifies the part of the RL training loop where the next level/environment instance is chosen for collecting experience.
Key Decision: When a new episode/trajectory needs to be collected, PLR decides between:
Sampling a New Level: Picking a level the agent hasn't seen before (probability P_D(d=0)). This often happens more frequently early in training. The level_sampler.py uses unseen_seed_weights to track this. When a level is sampled, its weight becomes 0. New levels are sampled uniformly from the remaining unseen ones (_sample_unseen_level).
Replaying a Seen Level: Selecting a level from the set the agent has already visited (Λ_seen). The probability of doing this increases as more levels are seen (controlled by rho and nu hyperparameters in level_sampler.py). The selection is governed by a replay distribution (P_replay).
The Replay Distribution (P_replay): This is the heart of the prioritization. It's a mix of two factors:

Score-based Priority (P_S): Favors levels estimated to offer high learning potential. Calculated based on seed_scores in level_sampler.py.
Staleness-based Priority (P_C): Favors levels that haven't been replayed recently. This ensures diversity and prevents scores from becoming outdated ("off-policy") as the agent improves. Calculated based on seed_staleness.
Mixing: The final probability is a weighted sum:

$$P_replay = (1 - staleness_coef) \cdot P_S + staleness_coef \cdot P_C$$

(staleness_coef corresponds to ρ in the explanation). The level_sampler.py calculates this within the sample_weights method.

Scoring Levels (S_i): How do we estimate "learning potential"?

Proxy: The paper suggests using metrics derived from the agent's learning process, like TD-errors or related quantities. High errors often indicate the agent was surprised by the outcome, suggesting room for learning.
Common Scores (Implemented in level_sampler.py):
value_l1: Average absolute L1 value loss (often equivalent to the magnitude of GAE). This is a core method highlighted in the paper (_average_value_l1). It uses returns and value_preds from the rollout.
gae: Average GAE (_average_gae).
Others (less common for PLR but exploring other potential signals): policy_entropy, least_confidence, min_margin, one_step_td_error.
Update: After a trajectory τ is collected from level l_i, its score S_i is calculated (e.g., using _average_value_l1) and updated in seed_scores. The level_sampler.py uses an exponential moving average controlled by alpha:

$$new_score = (1 - alpha) \cdot old_score + alpha \cdot calculated_score$$

in the update_seed_score method. The update_with_rollouts method orchestrates extracting trajectory segments and calling the appropriate score function.

Score-to-Probability Transformation (P_S): Raw scores need to be converted into a probability distribution.

Methods (Implemented in _score_transform):
rank: Probability proportional to $1 / rank(score)^{(1/temperature)}$. More robust to score magnitudes.
power: Probability proportional to $(score + eps)^{(1/temperature)}$. Direct use of scores.
softmax: Probability proportional to $exp(score / temperature)$.
eps_greedy: Mostly pick the best, with small chance eps of random.
Temperature (temperature): Controls the "sharpness" of the distribution. Low temperature focuses heavily on the highest-scored levels; high temperature makes it closer to uniform.
Staleness (P_C):
Tracking: seed_staleness array increments for all levels at each sampling step, but resets to 0 for the chosen level (_update_staleness).
Probability: Calculated similarly to scores using _score_transform (often power or rank) with its own staleness_temperature. Levels not sampled recently get higher staleness values and thus higher probability under P_C.
Implementation Structure
The provided level_replay/level_sampler.py is essentially the PLR module. Let's map the concepts to its structure:

# Simplified Conceptual Structure of level_sampler.py

class LevelSampler:
    def __init__(self, seeds, ..., strategy, replay_schedule, score_transform, temperature, rho, nu, alpha, staleness_coef, ...):
        # Store hyperparameters
        self.strategy = strategy # e.g., 'value_l1', 'random'
        self.replay_schedule = replay_schedule # 'fixed' or 'proportionate'
        self.score_transform = score_transform # 'rank', 'power', etc.
        self.temperature = temperature
        self.rho = rho # Threshold for starting replay
        self.nu = nu # Fixed probability of sampling unseen (if replay_schedule='fixed')
        self.alpha = alpha # EMA coefficient for score updates
        self.staleness_coef = staleness_coef # Weight for staleness term in P_replay
        # ... other hyperparameters ...

        # --- Core Data Structures ---
        self.seeds = np.array(seeds) # All possible level IDs
        self.seed2index = {seed: i for i, seed in enumerate(seeds)} # Map seed to array index

        self.unseen_seed_weights = np.array([1.0] * len(seeds)) # 1.0 if unseen, 0.0 if seen
        self.seed_scores = np.array([0.0] * len(seeds)) # Stores S_i for each level
        self.seed_staleness = np.array([0.0] * len(seeds)) # Stores staleness counter for P_C

        # (Partial scores/steps are for handling updates from multiple parallel actors)
        self.partial_seed_scores = ...
        self.partial_seed_steps = ...

    def sample(self) -> int:
        """Decides whether to sample a new or replay level and returns the chosen seed."""
        # 1. Check if random/sequential strategy
        if self.strategy == 'random' or self.strategy == 'sequential':
            # ... handle simple cases ...
            return seed

        # 2. Determine proportion of seen levels
        num_unseen = (self.unseen_seed_weights > 0).sum()
        proportion_seen = (len(self.seeds) - num_unseen) / len(self.seeds)

        # 3. Decide: Replay or New? (Based on replay_schedule, rho, nu, proportion_seen)
        sample_new = True
        if self.replay_schedule == 'fixed':
            if proportion_seen >= self.rho and (np.random.rand() > self.nu or num_unseen == 0):
                sample_new = False
        else: # Proportionate
             if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                 sample_new = False

        # 4. Sample accordingly
        if sample_new and num_unseen > 0:
            seed = self._sample_unseen_level()
        else:
            seed = self._sample_replay_level()

        return seed

    def _sample_unseen_level(self) -> int:
        """Uniformly sample from levels where unseen_seed_weights > 0."""
        # ... implementation ...
        # Find indices where unseen_seed_weights > 0
        # Choose one uniformly
        # Update staleness for the chosen index
        # Return seeds[chosen_index]
        pass # Simplified

    def _sample_replay_level(self) -> int:
        """Sample from seen levels based on P_replay."""
        # 1. Calculate combined weights (P_replay)
        weights = self.sample_weights() # This calculates P_S, P_C and mixes them
        # 2. Sample index based on weights
        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=weights)[0]
        # 3. Update staleness
        self._update_staleness(seed_idx)
        # 4. Return corresponding seed
        return int(self.seeds[seed_idx])

    def sample_weights(self) -> np.ndarray:
        """Calculates the P_replay probability distribution over seen levels."""
        # 1. Calculate score-based weights (P_S)
        #    - Apply score transform (e.g., rank) to self.seed_scores
        #    - Apply temperature
        #    - Mask out unseen levels (multiply by (1 - self.unseen_seed_weights))
        #    - Normalize
        score_weights = self._score_transform(self.score_transform, self.temperature, self.seed_scores)
        score_weights = score_weights * (1 - self.unseen_seed_weights)
        # ... normalization ...

        # 2. Calculate staleness-based weights (P_C) (if staleness_coef > 0)
        staleness_weights = 0
        if self.staleness_coef > 0:
            #   - Apply staleness transform to self.seed_staleness
            #   - Apply staleness temperature
            #   - Mask out unseen levels
            #   - Normalize
            staleness_weights = self._score_transform(self.staleness_transform, self.staleness_temperature, self.seed_staleness)
            staleness_weights = staleness_weights * (1 - self.unseen_seed_weights)
            # ... normalization ...

        # 3. Mix P_S and P_C
        final_weights = (1 - self.staleness_coef) * score_weights + self.staleness_coef * staleness_weights
        # ... ensure normalization ...
        return final_weights

    def _score_transform(self, transform, temperature, scores) -> np.ndarray:
        """Applies transformations like rank, power, softmax."""
        # ... implementation for 'rank', 'power', 'softmax', etc. ...
        pass # Simplified

    def _update_staleness(self, selected_idx):
        """Increment staleness for all, reset for selected."""
        if self.staleness_coef > 0:
            self.seed_staleness += 1
            self.seed_staleness[selected_idx] = 0

    # --- Score Calculation and Update ---

    def update_with_rollouts(self, rollouts):
        """Processes a RolloutStorage object to update scores for completed episodes."""
        if self.strategy == 'random': return # No scoring needed

        # Determine score function based on self.strategy ('value_l1', 'gae', etc.)
        score_function = self._get_score_function()

        # Iterate through actors and completed episodes within the rollout buffer
        level_seeds = rollouts.level_seeds # Shape: (num_steps, num_actors, 1)
        # ... other data like values, returns, log_probs from rollouts ...
        dones = ~(rollouts.masks > 0) # Shape: (num_steps, num_actors, 1)

        for actor_index in range(num_actors):
            # Find episode boundaries using 'dones'
            # For each completed episode:
                # Get the seed: seed = level_seeds[start_t, actor_index].item()
                # Get the corresponding index: seed_idx = self.seed2index[seed]
                # Extract data for the episode (values, returns, etc.)
                # Calculate score: score = score_function(**episode_data)
                # Update the score: self.update_seed_score(actor_index, seed_idx, score, num_steps)
        # Handle partial episodes at the end of the rollout buffer (_partial_update_seed_score)
        pass # Simplified structure

    def _get_score_function(self):
        """Returns the appropriate private method for calculating scores."""
        if self.strategy == 'value_l1':
            return self._average_value_l1
        elif self.strategy == 'gae':
            return self._average_gae
        # ... other strategies ...
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def _average_value_l1(self, returns, value_preds, **kwargs) -> float:
        """Calculates average L1 value loss ( |returns - value_preds| )."""
        advantages = returns - value_preds
        return advantages.abs().mean().item()

    # ... implementations for _average_gae, _average_entropy, etc. ...

    def update_seed_score(self, actor_index, seed_idx, score, num_steps):
        """Updates the score for a given seed_idx using EMA."""
        # (Handle merging partial scores first if applicable)
        # score = self._partial_update_seed_score(...)

        # Mark level as seen
        self.unseen_seed_weights[seed_idx] = 0.0

        # Apply EMA update
        old_score = self.seed_scores[seed_idx]
        self.seed_scores[seed_idx] = (1 - self.alpha) * old_score + self.alpha * score

    def after_update(self):
        """Called after the policy gradient update. Resets partial scores."""
        # (Merge any remaining partial scores into main scores)
        # Reset partial score/step buffers
        self.partial_seed_scores.fill(0)
        self.partial_seed_steps.fill(0)
PLR as a Plug-and-Play Module for PPO Curriculum Learning
The LevelSampler class is designed to be relatively self-contained, making it suitable as a plug-and-play component. Here's how you'd integrate it into a typical PPO training loop:

Minimal Requirements for Integration
Environment Interaction: Your environment setup needs to allow resetting to a specific level identified by a seed. The make_lr_venv function and the custom VecPyTorchProcgen/VecPyTorchMinigrid wrappers handle this.
PPO Implementation: Your PPO algorithm needs to compute and store necessary values in its rollout buffer (like value_preds and returns if using value_l1 or gae scoring). The RolloutStorage class in level_replay/storage.py does this.
Integration Steps (Simplified)
# Assume standard PPO setup with a RolloutStorage `rollouts`
# and parallel environments `envs`

# --- 1. Initialization ---
all_training_seeds = [...] # List of all possible level seeds
level_sampler_args = {
    'strategy': 'value_l1', # Or 'gae', 'rank', etc.
    'replay_schedule': 'fixed',
    'score_transform': 'rank',
    'temperature': 0.1,
    'rho': 0.2, # Start replay when 20% levels seen
    'nu': 0.5, # 50% chance to sample unseen after rho threshold
    'alpha': 1.0, # Score update factor (1.0 means replace old score)
    'staleness_coef': 0.1, # 10% weight to staleness
    # ... other args ...
}
level_sampler = LevelSampler(
    seeds=all_training_seeds,
    obs_space=envs.observation_space,
    action_space=envs.action_space,
    num_actors=args.num_processes,
    **level_sampler_args
)

# Initial reset using sequential sampling to see initial levels
obs, level_seeds = envs.reset() # Assuming envs uses level_sampler internally now
rollouts.obs[0].copy_(obs)
rollouts.to(device)

# --- 2. Training Loop ---
for j in range(num_updates):
    # --- 2a. Collect Rollouts ---
    for step in range(args.num_steps):
        # Act using policy
        value, action, action_log_dist, recurrent_hidden_states = actor_critic.act(...)

        # Step environment
        # *** MODIFICATION POINT 1 (Handled inside VecEnvWrapper) ***
        # The VecEnvWrapper's step_wait method now handles resets.
        # When an env is 'done', it calls level_sampler.sample() to get the next seed
        # and resets that specific environment instance with the new seed.
        obs, reward, done, infos = envs.step(action)

        # Store transition in rollouts
        # Need to store the level_seed associated with each step/transition
        current_level_seeds = torch.tensor([info['level_seed'] for info in infos]).unsqueeze(-1).to(device)
        rollouts.insert(..., level_seeds=current_level_seeds) # Add level_seeds storage

    # --- 2b. Compute Returns (Standard PPO) ---
    with torch.no_grad():
        next_value = actor_critic.get_value(...)
    rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

    # --- 2c. *** MODIFICATION POINT 2: Update Level Sampler *** ---
    # Provide the collected rollouts to the sampler to update scores/staleness
    # This needs GAE/returns calculated in the previous step.
    level_sampler.update_with_rollouts(rollouts)

    # --- 2d. Update Policy (Standard PPO) ---
    value_loss, action_loss, dist_entropy = agent.update(rollouts) # agent is PPO

    # --- 2e. *** MODIFICATION POINT 3: Sampler Post-Update Hook *** ---
    # Allows sampler to reset internal state if needed after policy weights change
    level_sampler.after_update()

    # (Standard logging, checkpointing etc.)
PLR Plugin Integration Key Points
Replacing the environment's default (often random) reset mechanism with calls to level_sampler.sample(). This is neatly handled within the custom VecEnvWrapper (VecPyTorchProcgen, VecPyTorchMinigrid) in the provided code.
Adding a call to level_sampler.update_with_rollouts(rollouts) after collecting experience and computing returns/advantages.
Adding a call to level_sampler.after_update() after the PPO policy update.
The LevelSampler class encapsulates all the logic for tracking seen levels, scores, staleness, and implementing the sampling strategies, requiring only the rollout data from the main PPO loop to function.