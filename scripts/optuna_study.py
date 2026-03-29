
import os
import sys
import numpy as np
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_intermediate_values,
    plot_contour,
    plot_parallel_coordinate,
)

sys.path.insert(0, os.path.dirname(__file__))
from environment import CartPoleEnv
from dqn import MLP_DQN

N_STEPS         = 1_000_000   # steps per trial
REPORT_INTERVAL = 1_000    # report intermediate value every N steps
N_TRIALS        = 50        # total Optuna trials
SEED            = 42        # random seed for reproducibility
GAMMA           = 0.99      # discount factor
REPLAY_BUFFER   = -1        # -1 means no replay buffer, just vanilla DQN updates every step
BATCH_SIZE      = 64        # batch size for experience replay updates

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots", "optuna")
os.makedirs(PLOTS_DIR, exist_ok=True)

LR_VALUES           = [1e-5, 1e-4, 1e-3]
UPDATE_FREQ_VALUES  = [1, 4, 16]
NETWORK_SIZE_VALUES = ["small", "medium", "large"]
EPSILON_MIN_VALUES  = [0.001, 0.01, 0.1]


def objective(trial: optuna.Trial) -> float:
    lr           = trial.suggest_categorical("lr",           LR_VALUES)
    update_freq  = trial.suggest_categorical("update_freq",  UPDATE_FREQ_VALUES)
    network_size = trial.suggest_categorical("network_size", NETWORK_SIZE_VALUES)
    epsilon_min  = trial.suggest_categorical("epsilon_min",  EPSILON_MIN_VALUES)

    env       = CartPoleEnv(render_mode="rgb_array", verbose=False, seed=SEED)
    agent     = MLP_DQN(lr=lr, epsilon_max=1.0, epsilon_min=epsilon_min,
                        replay_buffer_size=REPLAY_BUFFER, batch_size=BATCH_SIZE,
                        network_size=network_size)

    state      = env.reset()
    ep_reward  = 0.0
    ep_rewards = []
    epsilon    = 1.0

    for step in range(1, N_STEPS + 1):
        epsilon = max(epsilon_min, 1.0 - (1.0 - epsilon_min) * step / N_STEPS)

        a = agent.select_action(state, policy="e-greedy", epsilon=epsilon)
        next_state, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        ep_reward += reward

        agent.add_to_replay_buffer(state, a, reward, next_state, done)
        if step % update_freq == 0:
            batch = agent.sample_replay_buffer()
            if batch is not None:
                agent.update(*batch, gamma=GAMMA)
            else:
                agent.update([state], [a], [reward], [next_state], [done], gamma=GAMMA)

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            state = env.reset()
        else:
            state = next_state

        # report intermediate value for pruning and learning-curve plots
        if step % REPORT_INTERVAL == 0 and ep_rewards:
            mean_return = float(np.mean(ep_rewards[-50:]))
            trial.report(mean_return, step=step)
            if trial.should_prune():
                env.terminate()
                raise optuna.TrialPruned()

    env.terminate()

    # objective: mean return over last 100 episodes
    return float(np.mean(ep_rewards[-100:])) if len(ep_rewards) >= 100 else float(np.mean(ep_rewards))


def run_study() -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.NopPruner(),
        sampler=sampler,
        study_name="dqn_ablation",
        storage="sqlite:///millionrun.sqlite3",
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest trial:  #{study.best_trial.number}")
    print(f"Best value:  {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")

    return study


def save_plots(study: optuna.Study):
    figures = {
        "optimization_history":  plot_optimization_history(study),
        "param_importances":     plot_param_importances(study),
        "intermediate_values":   plot_intermediate_values(study),  # learning curves per trial
        "contour_lr_update":     plot_contour(study, params=["lr", "update_freq"]),
        "contour_lr_epsilon":    plot_contour(study, params=["lr", "epsilon_min"]),
        "parallel_coordinate":   plot_parallel_coordinate(study),
    }
    for name, fig in figures.items():
        path = os.path.join(PLOTS_DIR, f"{name}.html")
        fig.write_html(path)
        print(f"Saved → {path}")


if __name__ == "__main__":
    study = run_study()
    save_plots(study)
