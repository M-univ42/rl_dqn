import os
import csv
import matplotlib
matplotlib.use("TkAgg")  # renders in PyCharm's built-in plot viewer
import matplotlib.pyplot as plt

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    steps, returns, returns_smooth = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        last_step = -1
        for row in reader:
            step = int(row["env_step"])
            # the data in baseline csv is not perfectly sorted by env_step so the plot looks weird if we include rows where env_step goes backwards.
            # this is a quick fix to skip those rows, but ideally we should fix the data in the baselineCSV
            if step <= last_step:  # skip rows where env_step goes backwards
                continue
            last_step = step
            steps.append(step)
            returns.append(float(row["Episode_Return"]))
            returns_smooth.append(float(row["Episode_Return_smooth"]))
    return steps, returns, returns_smooth


def plot_learning_curves(configs: dict, title="Learning Curves", save_name="learning_curves.png"):
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, filename in configs.items():
        try:
            steps, returns, returns_smooth = load_csv(filename)
        except FileNotFoundError:
            print(f"[skip] {filename} not found — run experiments.py first")
            continue

        # raw returns as faint background
        # ax.plot(steps, returns, alpha=0.2)
        # smoothed on top, with label
        ax.plot(steps, returns_smooth, label=label, linewidth=2)

    ax.set_xlabel("Environment Step")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    save_path = os.path.join(PLOTS_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    # stript the .csv extension from the filename, that will be the label in the plot
    configs = {f[:-4]: f for f in sorted(csv_files)}
    plot_learning_curves(configs, title="DQN Learning Curves", save_name="learning_curves.png")
