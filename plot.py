"""Plot training curves for a run.

Usage:
    python plot.py                      # newest run in results/
    python plot.py results/<run_name>   # a specific run
Saves into the run folder:
    step_curves.png   per-step metrics from step_history.json
    epoch_curves.png  train/test loss and accuracy from history.json
Panels whose metric is missing from the run are skipped.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "#e6e5e0"
INK = "#55554f"

# Per-step panels: (title, [(key, label, color), ...], draw zero line)
STEP_PANELS = [
    ("Training loss (batch MSE)", [("loss", None, BLUE)], False),
    ("Batch gradient norm", [("batch_grad_norm", None, BLUE)], False),
    ("cos(batch gradient, sample gradients)",
     [("cos_batch_samples_mean", "mean", BLUE), ("cos_batch_samples_min", "min", ORANGE)], True),
    ("Fraction of samples at an obtuse angle to the batch gradient",
     [("frac_batch_samples_obtuse", None, BLUE)], False),
    ("cos(v, batch gradient)", [("cos_v_batch_grad", None, BLUE)], True),
    ("cos(v, sample gradients)",
     [("cos_v_samples_mean", "mean", BLUE), ("cos_v_samples_min", "min", ORANGE)], True),
]


def pick_run_dir(arg):
    if arg is not None:
        return Path(arg)
    runs = [d for d in (ROOT / "results").iterdir() if (d / "step_history.json").exists()]
    if not runs:
        sys.exit("no runs with step_history.json in results/")
    return max(runs, key=lambda d: (d / "step_history.json").stat().st_mtime)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color="#1f1f1d")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)


def draw_series(ax, x, rows, series, zero_line):
    for key, label, color in series:
        ax.plot(x, [r[key] for r in rows], color=color, linewidth=1.2, label=label)
    if zero_line:
        ax.axhline(0, color=INK, linewidth=0.8)
    if any(label for _, label, _ in series):
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)


def plot_steps(run_dir):
    with open(run_dir / "step_history.json") as f:
        hist = json.load(f)
    panels = [p for p in STEP_PANELS if all(k in hist[0] for k, _, _ in p[1])]
    step = [h["step"] for h in hist]

    ncols = 2
    nrows = (len(panels) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.8 * nrows), sharex=True, squeeze=False)
    for ax, (title, series, zero) in zip(axes.flat, panels):
        draw_series(ax, step, hist, series, zero)
        style(ax, title)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("step", color=INK, fontsize=9)

    fig.suptitle(run_dir.name, x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout()
    out = run_dir / "step_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_epochs(run_dir):
    path = run_dir / "history.json"
    if not path.exists():
        return
    with open(path) as f:
        hist = json.load(f)
    epoch = [h["epoch"] for h in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2))
    draw_series(ax1, epoch, hist, [("train_loss", "train", BLUE), ("test_loss", "test", ORANGE)], False)
    style(ax1, "Loss (MSE)")
    draw_series(ax2, epoch, hist, [("train_acc", "train", BLUE), ("test_acc", "test", ORANGE)], False)
    style(ax2, "Accuracy")
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch", color=INK, fontsize=9)

    fig.suptitle(run_dir.name, x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout()
    out = run_dir / "epoch_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    run_dir = pick_run_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    plot_steps(run_dir)
    plot_epochs(run_dir)


if __name__ == "__main__":
    main()
