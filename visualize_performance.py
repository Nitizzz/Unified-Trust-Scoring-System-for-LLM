"""
Unified Trust Metric System — Model Performance Visualization
==============================================================
Generates a multi-panel performance dashboard from training.log and results.txt.
Saves the output as 'performance_graphs.png'.

Usage:
    python visualize_performance.py
"""

import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

# ── Colour Palette ────────────────────────────────────────────────────
BG        = "#0f1117"
PANEL_BG  = "#1a1d28"
ACCENT_1  = "#6c63ff"   # violet
ACCENT_2  = "#00d2ff"   # cyan
ACCENT_3  = "#ff6b6b"   # coral
ACCENT_4  = "#2ecc71"   # green
GRID_CLR  = "#2a2d3a"
TEXT_CLR  = "#e0e0e0"
MUTED     = "#7f8c9b"

# ── Parsing helpers ───────────────────────────────────────────────────

def _clean(line: str) -> str:
    """Strip ANSI escapes and carriage-return artefacts."""
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
    if '\r' in line:
        line = line.split('\r')[-1]
    return line.strip()


def parse_training_log(path: str = "training.log"):
    """Return epoch numbers and losses from the log file."""
    epochs, losses = [], []
    for enc in ("utf-16", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                for raw in f:
                    line = _clean(raw)
                    m = re.search(r"Epoch\s+(\d+)/\d+\s*-\s*Loss:\s*([\d.]+)", line)
                    if m:
                        epochs.append(int(m.group(1)))
                        losses.append(float(m.group(2)))
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    return epochs, losses


def parse_results(path: str = "results.txt"):
    """Return a dict with P, R, F1 for code & summary, correlation, and confusion matrix."""
    data = {}
    cm_lines = []
    reading_cm = False

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"Code Hallucination:\s*P=([\d.]+),\s*R=([\d.]+),\s*F1=([\d.]+)", line)
            if m:
                data["code"] = {"P": float(m.group(1)), "R": float(m.group(2)), "F1": float(m.group(3))}
            m = re.match(r"Summary Hallucination:\s*P=([\d.]+),\s*R=([\d.]+),\s*F1=([\d.]+)", line)
            if m:
                data["summary"] = {"P": float(m.group(1)), "R": float(m.group(2)), "F1": float(m.group(3))}
            m = re.match(r"Trust Score Correlation:\s*([\d.]+)", line)
            if m:
                data["correlation"] = float(m.group(1))
            if "Confusion Matrix" in line:
                reading_cm = True
                continue
            if reading_cm:
                nums = re.findall(r"\d+", line)
                if nums:
                    cm_lines.append([int(n) for n in nums])
                if len(cm_lines) == 2:
                    reading_cm = False

    if cm_lines:
        data["cm"] = np.array(cm_lines)
    return data


# ── Drawing helpers ───────────────────────────────────────────────────

def _style_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_CLR, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linestyle="--", linewidth=0.5, alpha=0.5)


def plot_loss_curve(ax, epochs, losses):
    """Panel 1: Training loss over epochs."""
    _style_ax(ax, "Training Loss Curve", "Epoch", "Loss")
    ax.fill_between(epochs, losses, alpha=0.15, color=ACCENT_1)
    ax.plot(epochs, losses, color=ACCENT_1, linewidth=2.5, marker="o",
            markersize=6, markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    for e, l in zip(epochs, losses):
        ax.annotate(f"{l:.3f}", (e, l), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7.5, color=TEXT_CLR, alpha=0.85)
    ax.set_xticks(epochs)


def plot_classification_bars(ax, results):
    """Panel 2: Grouped bar chart for P / R / F1."""
    _style_ax(ax, "Classification Metrics", "", "Score")
    metrics = ["Precision", "Recall", "F1-Score"]
    code_vals = [results["code"]["P"], results["code"]["R"], results["code"]["F1"]]
    summ_vals = [results["summary"]["P"], results["summary"]["R"], results["summary"]["F1"]]

    x = np.arange(len(metrics))
    w = 0.32
    bars1 = ax.bar(x - w/2, code_vals, w, label="Code Hallucination",
                   color=ACCENT_2, edgecolor="white", linewidth=0.6, zorder=3)
    bars2 = ax.bar(x + w/2, summ_vals, w, label="Summary Hallucination",
                   color=ACCENT_3, edgecolor="white", linewidth=0.6, zorder=3)

    for bar_group in (bars1, bars2):
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9, color=TEXT_CLR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color=MUTED)
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper left", fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_CLR,
              labelcolor=TEXT_CLR, framealpha=0.9)


def plot_confusion_matrix(ax, cm):
    """Panel 3: Heatmap of confusion matrix."""
    _style_ax(ax, "Code Hallucination — Confusion Matrix")
    ax.grid(False)
    labels = ["Not Halluc.", "Halluc."]

    cmap = plt.cm.get_cmap("YlOrRd")
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color=MUTED, fontsize=9)
    ax.set_yticklabels(labels, color=MUTED, fontsize=9)
    ax.set_xlabel("Predicted", color=MUTED, fontsize=10)
    ax.set_ylabel("Actual", color=MUTED, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "#1a1d28")


def plot_trust_gauge(ax, correlation):
    """Panel 4: Radial gauge showing trust score correlation."""
    _style_ax(ax, "Trust Score Correlation (Spearman)")
    ax.grid(False)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw arc background
    theta_bg = np.linspace(np.pi, 0, 200)
    r = 1.0
    ax.plot(r * np.cos(theta_bg), r * np.sin(theta_bg), color=GRID_CLR, linewidth=14,
            solid_capstyle="round", zorder=1)

    # Draw filled arc proportional to correlation
    frac = max(0, min(correlation, 1.0))
    theta_fill = np.linspace(np.pi, np.pi - frac * np.pi, 200)
    gradient_color = ACCENT_4 if frac >= 0.6 else (ACCENT_2 if frac >= 0.4 else ACCENT_3)
    ax.plot(r * np.cos(theta_fill), r * np.sin(theta_fill), color=gradient_color,
            linewidth=14, solid_capstyle="round", zorder=2)

    # Needle
    needle_angle = np.pi - frac * np.pi
    ax.annotate("", xy=(0.85 * np.cos(needle_angle), 0.85 * np.sin(needle_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=TEXT_CLR, lw=2))

    # Centre dot
    ax.plot(0, 0, "o", color=TEXT_CLR, markersize=8, zorder=5)

    # Score label
    ax.text(0, 0.4, f"{correlation:.3f}", ha="center", va="center",
            fontsize=28, fontweight="bold", color=TEXT_CLR)
    ax.text(0, 0.15, "Spearman ρ", ha="center", va="center",
            fontsize=10, color=MUTED)

    # Scale labels
    for val, angle in [(0.0, np.pi), (0.25, 3*np.pi/4), (0.5, np.pi/2),
                       (0.75, np.pi/4), (1.0, 0)]:
        lx = 1.18 * np.cos(angle)
        ly = 1.18 * np.sin(angle)
        ax.text(lx, ly, f"{val}", ha="center", va="center", fontsize=8, color=MUTED)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Parsing training.log ...")
    epochs, losses = parse_training_log()
    if not epochs:
        print("  ⚠  No epoch data found in training.log — skipping loss panel.")

    print("Parsing results.txt ...")
    results = parse_results()
    if not results:
        print("  ✗  Could not parse results.txt — aborting.")
        sys.exit(1)

    # ── Build figure ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle("Unified Trust Metric System — Model Performance Dashboard",
                 color=TEXT_CLR, fontsize=17, fontweight="bold", y=0.97)

    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30,
                  left=0.07, right=0.95, top=0.90, bottom=0.07)

    # Panel 1 — Loss curve
    ax1 = fig.add_subplot(gs[0, 0])
    if epochs:
        plot_loss_curve(ax1, epochs, losses)
    else:
        _style_ax(ax1, "Training Loss Curve")
        ax1.text(0.5, 0.5, "No Training Data", transform=ax1.transAxes,
                 ha="center", va="center", fontsize=14, color=MUTED)

    # Panel 2 — Classification bars
    ax2 = fig.add_subplot(gs[0, 1])
    plot_classification_bars(ax2, results)

    # Panel 3 — Confusion matrix
    ax3 = fig.add_subplot(gs[1, 0])
    if "cm" in results:
        plot_confusion_matrix(ax3, results["cm"])
    else:
        _style_ax(ax3, "Confusion Matrix")
        ax3.text(0.5, 0.5, "No CM Data", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=14, color=MUTED)

    # Panel 4 — Trust score gauge
    ax4 = fig.add_subplot(gs[1, 1])
    plot_trust_gauge(ax4, results.get("correlation", 0.0))

    # Save
    out = "performance_graphs.png"
    fig.savefig(out, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"\n✓  Saved dashboard → {out}")


if __name__ == "__main__":
    main()
