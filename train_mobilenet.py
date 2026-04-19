"""
train_mobilenet.py
------------------
Trains ONLY TransferMobileNet on the Belgium Traffic Sign Dataset
and saves all evaluation graphs to the  outputs/  folder.

Graphs produced
---------------
1. train_val_loss.png          — Training & validation loss over epochs
2. val_metrics.png             — Val accuracy + macro F1 over epochs
3. confusion_matrix.png        — Normalised confusion matrix (heat-map)
4. per_class_f1.png            — Per-class F1-score bar chart          [presentation]
5. top_confused_pairs.png      — Most-confused class pairs             [presentation]
6. confidence_histogram.png    — Correct vs wrong prediction confidence [presentation]

Run
---
    python train_mobilenet.py
"""

import os
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score,
    recall_score, classification_report,
)

from dataset import get_dataloaders
from model import TransferMobileNet

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_EPOCHS        = 15
PHASE2_START      = 5       # epoch index (0-based) to unfreeze backbone blocks
BATCH_SIZE        = 32
IMG_SIZE          = 224
AUGMENTED         = True
LR_PHASE1         = 1e-3
LR_PHASE2         = 1e-4
OUTPUT_DIR        = Path("outputs")
CHECKPOINT_PATH   = OUTPUT_DIR / "best_TransferMobileNet.pth"

# Load class name mapping if available
BASE_DIR          = Path(__file__).resolve().parent
MAPPING_PATH      = BASE_DIR / "class_mapping.json"

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_class_name_map():
    if MAPPING_PATH.exists():
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def short_name(class_id_str, name_map, max_len=14):
    """Return a short display label for a class."""
    name = name_map.get(class_id_str, class_id_str)
    return name if len(name) <= max_len else name[:max_len - 1] + "…"

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — Training & Validation Loss
# ══════════════════════════════════════════════════════════════════════════════
def plot_loss(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, history["train_loss"], color="steelblue",  lw=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   color="tomato",     lw=2, label="Val Loss", linestyle="--")

    # Shade the overfitting region where val loss > train loss by a threshold
    train = np.array(history["train_loss"])
    val   = np.array(history["val_loss"])
    ax.fill_between(epochs, train, val,
                    where=(val > train + 0.05),
                    alpha=0.12, color="tomato", label="Overfitting gap")

    # Mark phase-2 start
    ax.axvline(PHASE2_START + 1, color="green", lw=1.4, linestyle=":",
               label=f"Phase 2 starts (epoch {PHASE2_START + 1})")

    # Mark best val epoch
    best_ep = int(np.argmin(history["val_loss"])) + 1
    ax.axvline(best_ep, color="gold", lw=1.4, linestyle="-.",
               label=f"Best val loss (epoch {best_ep})")

    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("TransferMobileNet — Training & Validation Loss")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUTPUT_DIR / "train_val_loss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — Validation Accuracy + Macro F1
# ══════════════════════════════════════════════════════════════════════════════
def plot_val_metrics(history):
    epochs = range(1, len(history["val_acc"]) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_acc = "steelblue"
    color_f1  = "darkorange"

    ax1.plot(epochs, history["val_acc"], color=color_acc, lw=2, marker="o",
             markersize=4, label="Val Accuracy (%)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)

    # Best accuracy marker
    best_ep  = int(np.argmax(history["val_acc"])) + 1
    best_acc = max(history["val_acc"])
    ax1.annotate(f"Best {best_acc:.1f}%",
                 xy=(best_ep, best_acc),
                 xytext=(best_ep + 0.5, best_acc - 3),
                 fontsize=9, color=color_acc,
                 arrowprops=dict(arrowstyle="->", color=color_acc))

    # F1 on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_f1"], color=color_f1, lw=2, marker="s",
             markersize=4, linestyle="--", label="Val Macro F1 (%)")
    ax2.set_ylabel("Macro F1 (%)", color=color_f1)
    ax2.tick_params(axis="y", labelcolor=color_f1)

    # Phase-2 marker
    ax1.axvline(PHASE2_START + 1, color="green", lw=1.4, linestyle=":",
                label=f"Phase 2 (epoch {PHASE2_START + 1})")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

    ax1.set_title("TransferMobileNet — Validation Metrics per Epoch")
    fig.tight_layout()
    out = OUTPUT_DIR / "val_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 3 — Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(true_labels, pred_labels, class_names, name_map):
    cm   = confusion_matrix(true_labels, pred_labels, normalize="true")
    n    = len(class_names)
    tick_labels = [short_name(c, name_map, max_len=12) for c in class_names]

    fig_size = max(10, n // 2)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(cm, ax=ax, cmap="Blues",
                xticklabels=tick_labels, yticklabels=tick_labels,
                linewidths=0.3, linecolor="#dddddd",
                annot=(n <= 20), fmt=".2f",
                cbar_kws={"shrink": 0.75, "label": "Proportion"})

    ax.set_title("TransferMobileNet — Normalised Confusion Matrix", pad=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=90, labelsize=max(6, 10 - n // 8))
    ax.tick_params(axis="y", rotation=0,  labelsize=max(6, 10 - n // 8))
    fig.tight_layout()
    out = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 4 — Per-Class F1 Bar Chart  [PRESENTATION]
# ══════════════════════════════════════════════════════════════════════════════
def plot_per_class_f1(true_labels, pred_labels, class_names, name_map):
    f1_per_class = f1_score(true_labels, pred_labels, average=None,
                            labels=list(range(len(class_names))),
                            zero_division=0)

    # Sort from worst to best
    order  = np.argsort(f1_per_class)
    sorted_names  = [short_name(class_names[i], name_map, max_len=16) for i in order]
    sorted_f1     = f1_per_class[order]
    colors = ["#e74c3c" if v < 0.5 else "#f39c12" if v < 0.75 else "#27ae60"
              for v in sorted_f1]

    fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.35)))
    bars = ax.barh(range(len(sorted_f1)), sorted_f1, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Score (sorted worst → best)", pad=12)
    ax.axvline(0.75, color="black", lw=1, linestyle="--", alpha=0.5, label="0.75 target")

    # Value labels on bars
    for bar, val in zip(bars, sorted_f1):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

    # Legend
    patches = [
        mpatches.Patch(color="#e74c3c", label="F1 < 0.50 (needs work)"),
        mpatches.Patch(color="#f39c12", label="0.50 ≤ F1 < 0.75"),
        mpatches.Patch(color="#27ae60", label="F1 ≥ 0.75 (good)"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    fig.tight_layout()
    out = OUTPUT_DIR / "per_class_f1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 5 — Top Confused Pairs  [PRESENTATION]
# ══════════════════════════════════════════════════════════════════════════════
def plot_top_confused_pairs(true_labels, pred_labels, class_names, name_map, top_n=10):
    cm = confusion_matrix(true_labels, pred_labels)
    np.fill_diagonal(cm, 0)   # zero out correct predictions

    # Flatten and find top off-diagonal entries
    flat    = cm.flatten()
    indices = np.argsort(flat)[::-1][:top_n]
    rows    = indices // len(class_names)
    cols    = indices  % len(class_names)
    counts  = flat[indices]

    pair_labels = [
        f"{short_name(class_names[r], name_map, 14)}\n→ {short_name(class_names[c], name_map, 14)}"
        for r, c in zip(rows, cols)
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = plt.cm.Reds(np.linspace(0.4, 0.85, top_n))[::-1]
    bars = ax.bar(range(top_n), counts, color=bar_colors, edgecolor="none", width=0.65)

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(int(cnt)), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(top_n))
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel("Number of misclassifications")
    ax.set_title(f"Top {top_n} Most-Confused Class Pairs  (True → Predicted)", pad=12)
    fig.tight_layout()
    out = OUTPUT_DIR / "top_confused_pairs.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 6 — Confidence Histogram  [PRESENTATION]
# ══════════════════════════════════════════════════════════════════════════════
def plot_confidence_histogram(true_labels, pred_labels, confidences):
    correct_conf = [c for t, p, c in zip(true_labels, pred_labels, confidences) if t == p]
    wrong_conf   = [c for t, p, c in zip(true_labels, pred_labels, confidences) if t != p]

    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(correct_conf, bins=bins, alpha=0.65, color="steelblue",
            label=f"Correct ({len(correct_conf):,})", edgecolor="white", lw=0.4)
    ax.hist(wrong_conf,   bins=bins, alpha=0.65, color="tomato",
            label=f"Wrong ({len(wrong_conf):,})",   edgecolor="white", lw=0.4)

    ax.axvline(0.5, color="black", lw=1.2, linestyle="--", alpha=0.6, label="0.5 threshold")
    ax.set_xlabel("Predicted Confidence (softmax probability)")
    ax.set_ylabel("Number of samples")
    ax.set_title("Confidence Distribution — Correct vs Wrong Predictions", pad=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = OUTPUT_DIR / "confidence_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name_map  = load_class_name_map()
    print(f"Device : {device}")
    print(f"Output : {OUTPUT_DIR.resolve()}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data …")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=BATCH_SIZE, img_size=IMG_SIZE, augmented=AUGMENTED
    )
    num_classes = len(class_names)
    print(f"  Classes : {num_classes}")
    print(f"  Train   : {len(train_loader.dataset)} samples")
    print(f"  Val     : {len(val_loader.dataset)} samples")
    print(f"  Test    : {len(test_loader.dataset)} samples\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = TransferMobileNet(num_classes=num_classes, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1
    )

    # ── History buffers ───────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "val_f1":     [],
    }
    best_val_acc = 0.0

    print("=" * 55)
    print(f"  Training TransferMobileNet  ({NUM_EPOCHS} epochs)")
    print("=" * 55)

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # ── Phase 2: unfreeze backbone ─────────────────────────────────────
        if epoch == PHASE2_START:
            model.unfreeze_last_n_blocks(n=4)
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE2
            )
            print(f"\n  [Phase 2] Backbone unfrozen at epoch {epoch+1}, LR → {LR_PHASE2}")

        # ── Training step ─────────────────────────────────────────────────
        model.train()
        running_loss = correct_train = total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss  += loss.item()
            _, preds       = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train   += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc  = 100 * correct_train / total_train

        # ── Validation step ───────────────────────────────────────────────
        model.eval()
        val_loss_sum = correct_val = total_val = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss_sum += loss.item()
                _, preds      = torch.max(outputs, 1)
                correct_val  += (preds == labels).sum().item()
                total_val    += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss_sum / len(val_loader)
        val_acc  = 100 * correct_val / total_val
        val_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100

        # ── Log ──────────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch+1:02d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
              f"val_f1={val_f1:.1f}%  ({elapsed:.0f}s)")

        # ── Checkpoint ───────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"    ✓ New best saved ({best_val_acc:.2f}%)")

    # ── Final test evaluation ──────────────────────────────────────────────
    print(f"\nLoading best checkpoint for final test evaluation …")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    test_preds, test_labels, test_confs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)
            confs  = probs.max(dim=1).values

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())
            test_confs.extend(confs.cpu().numpy())

    test_acc = 100 * sum(p == t for p, t in zip(test_preds, test_labels)) / len(test_labels)
    test_f1  = f1_score(test_labels, test_preds, average="macro", zero_division=0) * 100
    test_prec = precision_score(test_labels, test_preds, average="macro", zero_division=0) * 100
    test_rec  = recall_score(test_labels, test_preds, average="macro", zero_division=0) * 100

    print(f"\n{'='*45}")
    print(f"  TEST RESULTS")
    print(f"{'='*45}")
    print(f"  Accuracy  : {test_acc:.2f}%")
    print(f"  Macro F1  : {test_f1:.2f}%")
    print(f"  Precision : {test_prec:.2f}%")
    print(f"  Recall    : {test_rec:.2f}%")
    print(f"{'='*45}\n")

    # ── Generate all graphs ────────────────────────────────────────────────
    print("Generating graphs …")
    plot_loss(history)
    plot_val_metrics(history)
    plot_confusion_matrix(test_labels, test_preds, class_names, name_map)
    plot_per_class_f1(test_labels, test_preds, class_names, name_map)
    plot_top_confused_pairs(test_labels, test_preds, class_names, name_map)
    plot_confidence_histogram(test_labels, test_preds, test_confs)

    print(f"\nAll graphs saved to:  {OUTPUT_DIR.resolve()}/")
    print("Done ✓")
