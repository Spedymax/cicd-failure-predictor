"""Reproducible EDA over the engineered features parquet.

Produces a small set of PNGs under data/processed/eda/ that are referenced
from chapter 4 of the diploma. Re-run after data changes::

    python -m notebooks.eda_run \\
        --features ../data/processed/features.parquet \\
        --out ../data/processed/eda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features.transformers import feature_columns

logger = logging.getLogger(__name__)

PALETTE = {
    "success": "#22c55e",
    "oom_killed": "#ef4444",
    "test_timeout": "#f59e0b",
    "dependency_error": "#3b82f6",
    "docker_build_failed": "#a855f7",
    "network_error": "#06b6d4",
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["failure_class"].fillna("success").value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [PALETTE.get(k, "#888") for k in counts.index]
    counts.plot.barh(ax=ax, color=colors)
    ax.set_xlabel("Number of runs")
    ax.set_title("Class balance in engineered dataset")
    ax.invert_yaxis()
    for i, v in enumerate(counts.values):
        ax.text(v + counts.max() * 0.005, i, f"{v}", va="center", fontsize=9)
    _save(fig, out_dir, "01_class_balance.png")


def plot_class_conditional_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("peak_memory_mb", "Peak memory (MB)", "log"),
        ("duration_seconds", "Duration (s)", "log"),
        ("estimated_final_image_size_mb", "Image size (MB)", "log"),
        ("feat_lines_changed_log", "log(1 + total lines)", "linear"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    classes = df["failure_class"].fillna("success").unique()
    for ax, (col, label, scale) in zip(axes.ravel(), metrics, strict=False):
        for cls in classes:
            mask = df["failure_class"].fillna("success") == cls
            values = df.loc[mask, col].dropna()
            if values.empty:
                continue
            ax.hist(
                values,
                bins=40,
                alpha=0.5,
                label=cls,
                color=PALETTE.get(cls, "#888"),
                density=True,
            )
        ax.set_xscale(scale)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Per-class feature distributions")
    fig.tight_layout()
    _save(fig, out_dir, "02_class_conditional_distributions.png")


def plot_feature_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    feats = feature_columns(df)
    corr = df[feats].corr().fillna(0)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=6)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Engineered feature correlation matrix")
    fig.tight_layout()
    _save(fig, out_dir, "03_feature_correlation.png")


def plot_class_means_table(df: pd.DataFrame, out_dir: Path) -> None:
    feats = [
        "feat_lines_changed_log",
        "feat_files_changed_log",
        "feat_has_dependency_change_int",
        "feat_has_dockerfile_change_int",
        "feat_new_deps_count",
        "feat_image_growth_ratio",
        "feat_author_success_rate",
        "feat_project_failure_rate",
    ]
    grouped = df.groupby(df["failure_class"].fillna("success"))[feats].median().round(2)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    table = ax.table(
        cellText=grouped.values,
        rowLabels=grouped.index,
        colLabels=[c.replace("feat_", "") for c in grouped.columns],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title("Median engineered features per class", pad=10)
    _save(fig, out_dir, "04_class_medians_table.png")


def plot_real_vs_synthetic(df: pd.DataFrame, out_dir: Path) -> None:
    if "dataset_source" not in df.columns or df["dataset_source"].nunique() < 2:
        logger.info("skipping real-vs-synthetic plot (single source)")
        return
    metrics = ["feat_lines_changed_log", "feat_files_changed_log"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 4))
    for ax, col in zip(axes, metrics, strict=False):
        for src, color in zip(("real", "synthetic"), ("#1d4ed8", "#9333ea"), strict=False):
            values = df.loc[df["dataset_source"] == src, col].dropna()
            if values.empty:
                continue
            ax.hist(values, bins=40, alpha=0.5, label=src, color=color, density=True)
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        ax.legend()
    fig.suptitle("Real vs synthetic feature distributions")
    fig.tight_layout()
    _save(fig, out_dir, "05_real_vs_synthetic.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    df = pd.read_parquet(args.features)
    logger.info("loaded %d rows × %d cols", len(df), len(df.columns))

    out_dir = Path(args.out)
    plot_class_balance(df, out_dir)
    plot_class_conditional_distributions(df, out_dir)
    plot_feature_correlation(df, out_dir)
    plot_class_means_table(df, out_dir)
    plot_real_vs_synthetic(df, out_dir)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
