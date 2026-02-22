# -*- coding: utf-8 -*-
"""
Reinforcement Learning — HVAC Maintenance Optimization (Phase 5).
================================================================================

Pedagogical demonstration of Reinforcement Learning (RL) applied to
optimizing HVAC maintenance scheduling across 96 French departments.

RL Concepts:
    - **Agent**: Decision-making system that observes the state and chooses an action.
    - **Environment**: HVAC simulator returning state + reward.
    - **State**: Vector [temperature, humidity, age, demand, season, budget].
    - **Action**: {do nothing, preventive, corrective, replacement}.
    - **Reward**: Quality signal (+preventive before failure, -failure, -cost).
    - **Policy**: State -> action mapping maximizing cumulative reward.

Q-Learning Algorithm:
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    Model-free, off-policy algorithm. Discretized Q-table, epsilon-greedy
    exploration with exponential decay.

Scaling Up:
    - Stable-Baselines3 (PPO/SAC) for Deep RL with PyTorch
    - Ray RLlib for distributed multi-agent training

Dependencies: numpy, matplotlib, gymnasium (optional)

Usage:
    python src/models/reinforcement_learning_demo.py
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- Optional Gymnasium management ---
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    warnings.warn(
        "gymnasium is not installed. Simplified interface used. "
        "Install via: pip install gymnasium"
    )

# --- Matplotlib (non-interactive mode for headless servers) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Custom Environment — HVACMaintenanceEnv
# =============================================================================

# HVAC domain constants
ACTIONS = {0: "no_maintenance", 1: "preventive_maintenance",
           2: "corrective_repair", 3: "equipment_replacement"}
ACTION_COSTS = {0: 0.0, 1: 0.15, 2: 0.40, 3: 0.70}
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


class HVACMaintenanceEnv:
    """Gymnasium environment simulating HVAC maintenance over 12 months.

    Observation (Box, 6 dims): [temp, humidity, age, demand, season, budget]
    Actions (Discrete 4): none / preventive / corrective / replacement
    Episode: 12 steps (1 year), reward based on decision quality.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        """Initialize the HVAC environment."""
        super().__init__()
        self.render_mode = render_mode
        self.observation_space_low = np.array([0.0] * 6, dtype=np.float32)
        self.observation_space_high = np.array([1.0] * 6, dtype=np.float32)
        self.n_actions = 4

        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=self.observation_space_low,
                high=self.observation_space_high, dtype=np.float32)
            self.action_space = spaces.Discrete(self.n_actions)

        # Internal variables
        self.current_month: int = 0
        self.remaining_budget: float = 1.0
        self.equipment_age: float = 0.0
        self.equipment_condition: float = 1.0  # 1.0 = new, 0.0 = out of service

    def _generate_weather(self, month_idx: int) -> Tuple[float, float]:
        """Generate realistic temperature and humidity (French temperate climate)."""
        # Sinusoidal temperature: ~3 C in January, ~22 C in July
        temp = 12.5 + 10.0 * np.sin(2 * np.pi * (month_idx - 3) / 12)
        temp += self.np_random.normal(0, 2.0)
        temperature = float(np.clip((temp + 10) / 50, 0.0, 1.0))
        # Humidity: higher in winter
        hum = 0.7 - 0.15 * np.sin(2 * np.pi * (month_idx - 3) / 12)
        humidity = float(np.clip(hum + self.np_random.normal(0, 0.05), 0.0, 1.0))
        return temperature, humidity

    def _calculate_demand(self, temperature: float, month_idx: int) -> float:
        """HVAC demand in normalized units: high in winter (heating) and summer (AC)."""
        temp_reelle = temperature * 50 - 10
        demand = abs(temp_reelle - 20) / 30 + self.np_random.normal(0, 0.05)
        return float(np.clip(demand, 0.0, 1.0))

    def _failure_probability(self) -> float:
        """Failure probability increasing with age, decreasing with equipment condition."""
        prob = 0.05 + 0.4 * (self.equipment_age ** 2) * (1 - self.equipment_condition * 0.5)
        return float(np.clip(prob, 0.0, 0.95))

    def _build_observation(self, temp: float, hum: float, dem: float) -> np.ndarray:
        """Build the observation vector [temp, hum, age, dem, season, budget]."""
        season = (np.sin(2 * np.pi * self.current_month / 12) + 1) / 2
        return np.array([temp, hum, self.equipment_age, dem,
                         season, self.remaining_budget], dtype=np.float32)

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the episode with random initial conditions."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.current_month = 0
        self.remaining_budget = 1.0
        self.equipment_age = float(self.np_random.uniform(0.0, 0.6))
        self.equipment_condition = float(np.clip(
            1.0 - self.equipment_age * 0.5 + self.np_random.normal(0, 0.1), 0.2, 1.0))

        temp, hum = self._generate_weather(0)
        obs = self._build_observation(temp, hum, self._calculate_demand(temp, 0))
        return obs, {"month": MONTHS[0], "initial_age": self.equipment_age}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute an action and advance by one month. Returns (obs, rew, done, trunc, info)."""
        assert 0 <= action < self.n_actions, f"Invalid action: {action}"
        reward = 0.0
        info: Dict[str, Any] = {"month": MONTHS[self.current_month], "action": ACTIONS[action]}

        # 1. Action cost
        cost = ACTION_COSTS[action]
        self.remaining_budget = max(0.0, self.remaining_budget - cost)
        reward -= cost

        # 2. Action effect + failures
        failure_occurred = False
        failure_prob = self._failure_probability()
        info["failure_prob"] = round(failure_prob, 3)

        if action == 0:
            # No maintenance — failure risk
            if self.np_random.random() < failure_prob:
                failure_occurred = True
                reward -= 5.0
                self.equipment_condition = max(0.1, self.equipment_condition - 0.4)
        elif action == 1:
            # Preventive maintenance
            if self.equipment_age > 0.3:
                reward += 2.0  # Good decision on aging equipment
                self.equipment_condition = min(1.0, self.equipment_condition + 0.2)
                if self.np_random.random() < failure_prob:
                    reward += 3.0  # Bonus: failure avoided
            else:
                reward -= 0.5  # Unnecessary maintenance, equipment is new
        elif action == 2:
            # Corrective repair
            if self.equipment_condition < 0.6:
                self.equipment_condition = min(1.0, self.equipment_condition + 0.35)
                reward += 1.0
            else:
                reward -= 0.3  # Unnecessary repair
        elif action == 3:
            # Full replacement
            if self.equipment_age > 0.7:
                self.equipment_age, self.equipment_condition = 0.0, 1.0
                reward += 2.5
            else:
                self.equipment_age, self.equipment_condition = 0.0, 1.0
                reward -= 1.0  # Premature replacement

        # 3. Natural aging
        self.equipment_age = min(1.0, self.equipment_age + 1.0 / 15)
        self.equipment_condition = max(0.1, self.equipment_condition - 0.03)

        # 4. Time advancement
        self.current_month += 1
        done_flag = self.current_month >= 12

        # End-of-episode bonus: good budget management
        if done_flag:
            reward += self.remaining_budget * 2.0

        # 5. New state
        obs_month = min(self.current_month, 11)
        temp, hum = self._generate_weather(obs_month)
        obs = self._build_observation(temp, hum, self._calculate_demand(temp, obs_month))
        info.update({"failure": failure_occurred, "equipment_condition": round(self.equipment_condition, 3),
                     "remaining_budget": round(self.remaining_budget, 3)})
        return obs, float(reward), done_flag, False, info


# =============================================================================
# 2. Q-Learning Agent (implemented from scratch, no external RL library)
# =============================================================================

class AgentQLearning:
    """Tabular Q-Learning agent with state discretization and epsilon-greedy policy.

    Exploration vs exploitation: epsilon starts at 1.0 (full exploration)
    and decays exponentially toward epsilon_min (dominant exploitation).
    """

    def __init__(self, n_bins: int = 6, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995):
        """Configure Q-Learning hyperparameters."""
        self.n_bins = n_bins
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon   # Exploration probability
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = 4
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def _discretize_state(self, observation: np.ndarray) -> Tuple[int, ...]:
        """Discretize the continuous observation into n_bins per dimension."""
        indices = np.clip((observation * self.n_bins).astype(int), 0, self.n_bins - 1)
        return tuple(indices.tolist())

    def _get_q_values(self, state: Tuple[int, ...]) -> np.ndarray:
        """Return Q-values for a state (initialized to 0 if unknown)."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=np.float64)
        return self.q_table[state]

    def get_nearest_q(self, observation: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Return Q-values from the exact bin or the nearest neighbor if unexplored.

        Useful for displaying the policy on unvisited states.
        Returns (q_values, exact_match).
        """
        state = self._discretize_state(observation)
        if state in self.q_table:
            return self.q_table[state].copy(), True
        # Search for the nearest neighbor (Manhattan distance)
        if not self.q_table:
            return np.zeros(self.n_actions), False
        state_arr = np.array(state)
        best_match, min_dist = None, float("inf")
        for visited in self.q_table:
            d = np.sum(np.abs(np.array(visited) - state_arr))
            if d < min_dist:
                min_dist, best_match = d, visited
        return self.q_table[best_match].copy(), False

    def choose_action(self, observation: np.ndarray) -> int:
        """Epsilon-greedy: random exploration or exploitation of max Q."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state = self._discretize_state(observation)
        return int(np.argmax(self._get_q_values(state)))

    def update(self, obs: np.ndarray, action: int, reward: float,
                      next_obs: np.ndarray, done_flag: bool) -> None:
        """Q-Learning update: Q(s,a) += alpha * [target - Q(s,a)]."""
        state = self._discretize_state(obs)
        q_vals = self._get_q_values(state)
        q_next = self._get_q_values(self._discretize_state(next_obs))
        target = reward if done_flag else reward + self.gamma * np.max(q_next)
        q_vals[action] += self.alpha * (target - q_vals[action])

    def decay_epsilon(self) -> None:
        """Exponential epsilon decay after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# 3. Training loops and comparison with random policy
# =============================================================================

def train_agent(env: HVACMaintenanceEnv, agent: AgentQLearning,
                    n_episodes: int = 1000) -> List[float]:
    """Train the Q-Learning agent. Returns the reward history."""
    history: List[float] = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        cumulative = 0.0
        for _ in range(12):
            action = agent.choose_action(obs)
            next_obs, rew, done, trunc, _ = env.step(action)
            agent.update(obs, action, rew, next_obs, done)
            cumulative += rew
            obs = next_obs
            if done or trunc:
                break
        agent.decay_epsilon()
        history.append(cumulative)
        if (episode + 1) % 200 == 0:
            logger.info(f"Episode {episode+1}/{n_episodes} — "
                        f"Avg(100): {np.mean(history[-100:]):.2f} — "
                        f"Eps: {agent.epsilon:.3f}")
    return history


def evaluate_random_policy(env: HVACMaintenanceEnv,
                                n_episodes: int = 1000) -> List[float]:
    """Baseline: purely random policy for comparison."""
    history: List[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        cumulative = 0.0
        for _ in range(12):
            obs, rew, done, trunc, _ = env.step(np.random.randint(4))
            cumulative += rew
            if done or trunc:
                break
        history.append(cumulative)
    return history


# =============================================================================
# 4. Visualizations
# =============================================================================

def plot_learning_curves(rewards_ql: List[float],
                                 rewards_random: List[float],
                                 window: int = 50,
                                 path: str = "reports/figures/rl_learning_curve.png") -> None:
    """Learning curve: rolling average Q-Learning vs random."""
    fig, ax = plt.subplots(figsize=(12, 6))
    avg_ql = np.convolve(rewards_ql, np.ones(window) / window, mode="valid")
    avg_rand = np.convolve(rewards_random, np.ones(window) / window, mode="valid")
    ax.plot(avg_ql, label="Q-Learning (rolling avg)", color="#2196F3", linewidth=2)
    ax.plot(avg_rand, label="Random Policy", color="#F44336", linewidth=2, linestyle="--")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.set_title("Reinforcement Learning — HVAC Maintenance\n"
                 "Q-Learning vs Random Policy Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Curve saved: {path}")


def plot_q_values_heatmap(agent: AgentQLearning,
                            path: str = "reports/figures/rl_heatmap_q_values.png") -> None:
    """Heatmap of Q-values for representative HVAC fleet states."""
    # Representative states: [temp, hum, age, demand, season, budget]
    key_states = {
        "Summer/New/Budget+":          np.array([0.7, 0.5, 0.1, 0.8, 0.8, 0.9]),
        "Summer/Old/Budget+":          np.array([0.7, 0.5, 0.7, 0.8, 0.8, 0.7]),
        "Winter/New/Budget+":          np.array([0.2, 0.7, 0.1, 0.7, 0.2, 0.8]),
        "Winter/Old/Budget+":          np.array([0.2, 0.7, 0.7, 0.7, 0.2, 0.6]),
        "Winter/Old/Budget-":          np.array([0.2, 0.7, 0.7, 0.7, 0.2, 0.2]),
        "Mid-season/Medium/Budget=":   np.array([0.5, 0.6, 0.4, 0.4, 0.5, 0.5]),
        "Summer/VeryOld/Budget+":      np.array([0.7, 0.5, 0.9, 0.8, 0.8, 0.6]),
        "Winter/VeryOld/Budget-":      np.array([0.2, 0.8, 0.9, 0.9, 0.2, 0.1]),
    }
    action_names = ["No\nmaint.", "Prev.\nmaint.",
                    "Correct.\nrepair", "Replace-\nment"]
    names = list(key_states.keys())
    matrix = np.zeros((len(names), 4))
    for i, obs in enumerate(key_states.values()):
        matrix[i], _ = agent.get_nearest_q(obs)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(action_names, fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    # Annotate cells
    range_val = matrix.max() - matrix.min() if matrix.max() != matrix.min() else 1.0
    for i in range(len(names)):
        for j in range(4):
            c = "white" if abs(matrix[i, j]) > range_val * 0.4 else "black"
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color=c, fontweight="bold")
    ax.set_title("Learned Q-values by state and action\n"
                 "(green = favorable, red = unfavorable)", fontsize=13)
    fig.colorbar(im, ax=ax, label="Q-value")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Heatmap saved: {path}")


def display_policy_summary(agent: AgentQLearning) -> None:
    """Display the learned policy for representative scenarios."""
    print("\n" + "=" * 70)
    print("LEARNED POLICY SUMMARY")
    print("=" * 70)
    scenarios = [
        ("New equipment, winter, high budget",       [0.2, 0.7, 0.1, 0.7, 0.2, 0.9]),
        ("New equipment, summer, high budget",       [0.7, 0.5, 0.1, 0.8, 0.8, 0.9]),
        ("Medium equipment, winter, medium budget",  [0.2, 0.7, 0.4, 0.7, 0.2, 0.5]),
        ("Old equipment, winter, high budget",       [0.2, 0.7, 0.7, 0.7, 0.2, 0.7]),
        ("Old equipment, summer, low budget",        [0.7, 0.5, 0.7, 0.8, 0.8, 0.2]),
        ("Very old equipment, winter, high budget",  [0.2, 0.7, 0.9, 0.9, 0.2, 0.6]),
        ("Very old equipment, summer, low budget",   [0.7, 0.5, 0.9, 0.8, 0.8, 0.1]),
    ]
    for desc, obs_list in scenarios:
        obs = np.array(obs_list, dtype=np.float32)
        q_vals, exact = agent.get_nearest_q(obs)
        best = int(np.argmax(q_vals))
        marker = "" if exact else " (nearest state)"
        print(f"\n  {desc}{marker}")
        print(f"    -> Action: {ACTIONS[best]}")
        print(f"    -> Q: {', '.join(f'{v:.2f}' for v in q_vals)}")
    print("\n" + "=" * 70)


# =============================================================================
# 5. Main entry point
# =============================================================================

def main() -> None:
    """Train the Q-Learning agent, compare to random policy, and visualize."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    N_EPISODES = 1000
    np.random.seed(42)

    print("=" * 70)
    print("REINFORCEMENT LEARNING — HVAC MAINTENANCE")
    print("Optimization of scheduling across 96 French departments")
    print("=" * 70)

    # --- Environment creation ---
    env = HVACMaintenanceEnv()
    print(f"\nEnvironment: {env.n_actions} actions, 6 state variables, 12 months/episode")

    # --- Q-Learning training ---
    print(f"\n--- Q-Learning Training ({N_EPISODES} episodes) ---")
    agent = AgentQLearning(n_bins=6, alpha=0.1, gamma=0.95,
                           epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)
    rewards_ql = train_agent(env, agent, n_episodes=N_EPISODES)
    avg_ql = np.mean(rewards_ql[-100:])
    print(f"\nQ-Learning — Avg (last 100): {avg_ql:.2f}")
    print(f"Q-table: {len(agent.q_table)} states | Final epsilon: {agent.epsilon:.4f}")

    # --- Random policy (baseline) ---
    print(f"\n--- Random policy ({N_EPISODES} episodes) ---")
    rewards_rand = evaluate_random_policy(env, n_episodes=N_EPISODES)
    avg_rand = np.mean(rewards_rand)
    print(f"Random — Avg: {avg_rand:.2f}")

    # --- Comparison ---
    improvement = ((avg_ql - avg_rand) / abs(avg_rand)) * 100 if avg_rand != 0 else 0
    print(f"\n--- Comparison ---")
    print(f"  Q-Learning: {avg_ql:.2f}  |  Random: {avg_rand:.2f}  |  {improvement:+.1f}%")

    # --- Visualizations ---
    print("\n--- Generating visualizations ---")
    try:
        import os
        os.makedirs("reports/figures", exist_ok=True)
        plot_learning_curves(rewards_ql, rewards_rand)
        plot_q_values_heatmap(agent)
        print("Figures saved in reports/figures/")
    except Exception as e:
        logger.warning(f"Unable to save figures: {e}")

    # --- Policy summary ---
    display_policy_summary(agent)

    # --- Methodological note ---
    print("\n" + "-" * 70)
    print("METHODOLOGICAL NOTE:")
    print("-" * 70)
    print("  This prototype uses tabular Q-Learning, suited for demonstration.\n"
          "  To scale up (96 departments, real data):\n"
          "    - Stable-Baselines3 (PPO/SAC) for Deep RL with PyTorch\n"
          "    - Ray RLlib for distributed multi-agent training\n"
          "    - Integration of real weather data (Meteo-France API)\n"
          "    - Historical maintenance data from ADEME/CSTB")
    print("-" * 70)


if __name__ == "__main__":
    main()
