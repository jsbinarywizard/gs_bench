#!/usr/bin/env python3
"""Plot results of gs_halo_bench (MPI vs RCCL halo exchange).

Usage:
    python plot_results.py gs_halo_results.csv
    python plot_results.py run1.csv run2.csv -o comparison.png
    python plot_results.py gs_halo_results.csv --show

Generates three separate figures per scaling mode (weak / strong):
    1. time per step vs memory size
    2. throughput (cell updates per second) vs memory size
    3. speedup relative to the MPI variant (t_mpi / t_variant)

Memory size is calculated as:
    size^2 * 4 * 8 bytes

Several rank counts in the data are shown as different line styles.
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

STYLE = {
    "mpi": ("#1f77b4", "o"),
    "rccl-1comm": ("#d62728", "s"),
    "rccl-2comm": ("#2ca02c", "^"),
}

LINESTYLES = ["-", "--", ":", "-."]

# Memory unit conversion
KIB = 1024
MIB = 1024**2
GIB = 1024**3


def load(paths):
    frames = [pd.read_csv(p, skipinitialspace=True) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    # A repeated measurement of the same case: keep the latest one
    df = df.drop_duplicates(
        subset=["variant", "scaling", "ranks", "size"],
        keep="last"
    )

    failed = df[df["check"] != "PASS"]
    if not failed.empty:
        print("WARNING: correctness check failed for:")
        print(
            failed[
                ["variant", "scaling", "ranks", "size", "check"]
            ].to_string(index=False)
        )

    return df


def add_memory_size(df):
    """Calculate memory size in bytes, KiB, MiB and GiB."""
    df = df.copy()

    df["memory_bytes"] = df["size"]**2 * 4 * 8
    df["memory_kib"] = df["memory_bytes"] / KIB
    df["memory_mib"] = df["memory_bytes"] / MIB
    df["memory_gib"] = df["memory_bytes"] / GIB

    return df


def format_memory_axis(ax, memory_bytes):
    """Display memory size using KiB, MiB and GiB."""

    ticks = []
    labels = []

    for value in sorted(memory_bytes.unique()):
        if value < MIB:
            ticks.append(value / KIB)
            labels.append(f"{value / KIB:g} KiB")
        elif value < GIB:
            ticks.append(value / MIB)
            labels.append(f"{value / MIB:g} MiB")
        else:
            ticks.append(value / GIB)
            labels.append(f"{value / GIB:g} GiB")

    # Use bytes as the underlying x-axis scale.
    ax.set_xscale("log", base=2)
    ax.set_xticks(memory_bytes.unique())
    ax.set_xticklabels(labels, rotation=30, ha="right")

    # Use a secondary axis is unnecessary: labels directly show the
    # corresponding human-readable memory size.
    ax.set_xlabel("Memory size (KiB / MiB / GiB)")


def plot_scaling(df, scaling, stem, plt):

    df = add_memory_size(df)

    ranks_list = sorted(df["ranks"].unique())
    linestyle = {
        r: LINESTYLES[i % len(LINESTYLES)]
        for i, r in enumerate(ranks_list)
    }

    xlabel = "Memory size (KiB / MiB / GiB)"

    # ---------------------------------------------------------
    # 1. Time per step
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))

    for (variant, ranks), g in df.groupby(["variant", "ranks"]):
        g = g.sort_values("memory_bytes")

        color, marker = STYLE.get(variant, (None, "x"))
        label = (
            variant
            if len(ranks_list) == 1
            else f"{variant}, {ranks} ranks"
        )

        ax.plot(
            g["memory_bytes"],
            g["sec_per_step"],
            color=color,
            marker=marker,
            linestyle=linestyle[ranks],
            label=label
        )

    ax.set_title(f"Time per step ({scaling} scaling)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("seconds / step")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    format_memory_axis(ax, df["memory_bytes"])

    fig.tight_layout()
    out = stem.with_name(f"{stem.stem}_time{stem.suffix}")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # ---------------------------------------------------------
    # 2. Throughput
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))

    for (variant, ranks), g in df.groupby(["variant", "ranks"]):
        g = g.sort_values("memory_bytes")

        color, marker = STYLE.get(variant, (None, "x"))
        label = (
            variant
            if len(ranks_list) == 1
            else f"{variant}, {ranks} ranks"
        )

        ax.plot(
            g["memory_bytes"],
            g["updates_per_s"],
            color=color,
            marker=marker,
            linestyle=linestyle[ranks],
            label=label
        )

    ax.set_title(f"Throughput ({scaling} scaling)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cell updates / s")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    format_memory_axis(ax, df["memory_bytes"])

    fig.tight_layout()
    out = stem.with_name(f"{stem.stem}_throughput{stem.suffix}")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # ---------------------------------------------------------
    # 3. Speedup relative to MPI
    # ---------------------------------------------------------

    base = (
        df[df["variant"] == "mpi"][
            ["ranks", "size", "sec_per_step"]
        ]
        .rename(columns={"sec_per_step": "t_mpi"})
    )

    sp = df[df["variant"] != "mpi"].merge(
        base,
        on=["ranks", "size"]
    )

    sp["speedup"] = sp["t_mpi"] / sp["sec_per_step"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for (variant, ranks), g in sp.groupby(["variant", "ranks"]):
        g = g.sort_values("memory_bytes")

        color, marker = STYLE.get(variant, (None, "x"))
        label = (
            variant
            if len(ranks_list) == 1
            else f"{variant}, {ranks} ranks"
        )

        ax.plot(
            g["memory_bytes"],
            g["speedup"],
            color=color,
            marker=marker,
            linestyle=linestyle[ranks],
            label=label
        )

    ax.axhline(1.0, color="black", linewidth=0.8)

    if sp.empty:
        ax.text(
            0.5, 0.5,
            "no MPI baseline in data",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

    ax.set_title(f"Speedup vs MPI ({scaling} scaling)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("t_MPI / t_variant")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    format_memory_axis(ax, df["memory_bytes"])

    fig.tight_layout()
    out = stem.with_name(f"{stem.stem}_speedup{stem.suffix}")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        "csv",
        nargs="+",
        help="result file(s) written by gs_halo_bench"
    )

    ap.add_argument(
        "-o",
        "--output",
        help="output image prefix"
    )

    ap.add_argument(
        "--show",
        action="store_true",
        help="open an interactive window"
    )

    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    df = load(args.csv)

    if df.empty:
        raise SystemExit("no data found")

    stem = (
        Path(args.output)
        if args.output
        else Path(args.csv[0]).with_suffix(".png")
    )

    scalings = sorted(df["scaling"].unique())

    for scaling in scalings:
        scaling_stem = (
            stem
            if len(scalings) == 1
            else stem.with_name(f"{stem.stem}_{scaling}{stem.suffix}")
        )

        plot_scaling(
            df[df["scaling"] == scaling],
            scaling,
            scaling_stem,
            plt
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()