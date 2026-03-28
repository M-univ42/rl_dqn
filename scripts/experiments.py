import os
import csv
import numpy as np
from train import train

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# we want 5 runs per config to get a better approximation of the average learning curve
# seeds 0-4 for reproducibility and consistency across configs
SEEDS = [0, 1, 2, 3, 4]

TRAINING_CONFIGS = {
    # DQN: no buffer, no target network, updates after every step
    "dqn_no_buffer": dict(
        n_steps=1_00_000,
        lr=1e-4,
        epsilon_max=1.0,
        epsilon_min=0.01,
        gamma=0.99,
        replay_buffer_size=-1,
        batch_size=-1,
    ),
    # DQN with experience replay
    "dqn_replay": dict(
        n_steps=1_00_000,
        lr=1e-4,
        epsilon_max=1.0,
        epsilon_min=0.01,
        gamma=0.99,
        replay_buffer_size=10_000,
        batch_size=64,
    ),
}


def save_csv(filename, ep_rewards, ep_steps, rolling_window=10):
    path = os.path.join(DATA_DIR, filename)
    smoothed = []
    for i, r in enumerate(ep_rewards):
        start = max(0, i - rolling_window + 1)
        smoothed.append(np.mean(ep_rewards[start:i+1]))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode_Return", "Episode_Return_smooth", "env_step"])
        for r, s, step in zip(ep_rewards, smoothed, ep_steps):
            writer.writerow([r, round(s, 3), step])

    print(f"\nSaved → {path}")


def run_experiment(name, config, seed):
    print(f"\n{'='*60}")
    print(f"Running: {name}  |  seed={seed}")
    print(f"Config:  {config}")
    print(f"{'='*60}")
    n_steps, lr, epsilon_max, epsilon_min, gamma, replay_buffer_size, batch_size = config.values()
    ep_rewards, ep_steps = train(n_steps, lr, epsilon_max, epsilon_min, gamma,
                                 replay_buffer_size, batch_size, seed=seed)
    save_csv(f"{name}_seed{seed}.csv", ep_rewards, ep_steps)


def run_all():
    for name, config in TRAINING_CONFIGS.items():
        for seed in SEEDS:
            run_experiment(name, config, seed)


if __name__ == "__main__":
    run_all()
