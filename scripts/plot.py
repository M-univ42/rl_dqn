import os
import re
import csv
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # renders in PyCharm's built-in plot viewer
import matplotlib.pyplot as plt

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    steps, returns_smooth = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        last_step = -1
        for row in reader:
            step = int(row["env_step"])
            # the data in baseline csv is not perfectly sorted by env_step so the plot looks weird if we include rows where env_step goes backwards.
            # this is a quick fix to skip those rows, but ideally we should fix the data in the baselineCSV
            if step <= last_step:
                continue
            last_step = step
            steps.append(step)
            returns_smooth.append(float(row["Episode_Return_smooth"]))
    return np.array(steps), np.array(returns_smooth)

# group csv's by experiment name and seed number, so we can plot the mean and std across seeds for each experiment
def group_csvs(csv_files):
    groups = {}
    seed_pattern = re.compile(r"^(.+)_seed\d+\.csv$")
    for f in sorted(csv_files):
        m = seed_pattern.match(f)
        base = m.group(1) if m else f[:-4]
        groups.setdefault(base, []).append(f)
    return groups


def plot_learning_curves(title="Learning Curves", save_name="learning_curves.png"):
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    groups = group_csvs(csv_files)

    fig, ax = plt.subplots(figsize=(10, 5))

    for label, files in groups.items():
        runs = []
        for f in files:
            try:
                steps, smooth = load_csv(f)
                runs.append((steps, smooth))
            except FileNotFoundError:
                print(f"[skip] {f} not found")
                continue

        if not runs:
            continue

        if len(runs) == 1:
            # single run: just plot the smoothed line
            ax.plot(runs[0][0], runs[0][1], label=label, linewidth=2)
        else:
            # multiple seeds: interpolate to shared step grid, plot mean ± std
            # this gives us the nice shaded error region.
            max_step = min(r[0][-1] for r in runs)  # use shortest run's endpoint
            grid = np.linspace(0, max_step, 1000)
            interpolated = np.array([np.interp(grid, r[0], r[1]) for r in runs])
            mean = interpolated.mean(axis=0)
            std  = interpolated.std(axis=0)
            line, = ax.plot(grid, mean, label=label, linewidth=2)
            ax.fill_between(grid, mean - std, mean + std, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Environment Step")
    ax.set_ylabel("Episode Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    save_path = os.path.join(PLOTS_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_learning_curves(title="DQN Learning Curves", save_name="learning_curves.png")
