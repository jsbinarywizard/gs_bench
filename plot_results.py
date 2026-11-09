#!/usr/bin/env python3
"""Plot results of gs_halo_bench (MPI vs RCCL halo exchange).

Usage:
    python plot_results.py gs_halo_results.csv                 # -> gs_halo_results_weak.png ...
    python plot_results.py run1.csv run2.csv -o comparison.png
    python plot_results.py gs_halo_results.csv --show

One figure per scaling mode (weak / strong) with three panels:
    1. time per step vs size
    2. throughput (cell updates per second) vs size
    3. speedup relative to the MPI variant (t_mpi / t_variant)
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


def load(paths):
    frames = [pd.read_csv(p, skipinitialspace=True) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    # a repeated measurement of the same case: keep the latest one
    df = df.drop_duplicates(subset=["variant", "scaling", "ranks", "size"], keep="last")
    failed = df[df["check"] != "PASS"]
    if not failed.empty:
        print("WARNING: correctness check failed for:")
        print(failed[["variant", "scaling", "ranks", "size", "check"]].to_string(index=False))
    return df


def plot_scaling(df, scaling, out_path, plt):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ranks_list = sorted(df["ranks"].unique())
    linestyle = {r: LINESTYLES[i % len(LINESTYLES)] for i, r in enumerate(ranks_list)}

    for (variant, ranks), g in df.groupby(["variant", "ranks"]):
        g = g.sort_values("size")
        color, marker = STYLE.get(variant, (None, "x"))
        label = variant if len(ranks_list) == 1 else f"{variant}, {ranks} ranks"
        kw = dict(color=color, marker=marker, linestyle=linestyle[ranks], label=label)
        axes[0].plot(g["size"], g["sec_per_step"], **kw)
        axes[1].plot(g["size"], g["updates_per_s"], **kw)

    # speedup relative to MPI
    base = (df[df["variant"] == "mpi"][["ranks", "size", "sec_per_step"]]
            .rename(columns={"sec_per_step": "t_mpi"}))
    sp = df[df["variant"] != "mpi"].merge(base, on=["ranks", "size"])
    sp["speedup"] = sp["t_mpi"] / sp["sec_per_step"]
    for (variant, ranks), g in sp.groupby(["variant", "ranks"]):
        g = g.sort_values("size")
        color, marker = STYLE.get(variant, (None, "x"))
        label = variant if len(ranks_list) == 1 else f"{variant}, {ranks} ranks"
        axes[2].plot(g["size"], g["speedup"], color=color, marker=marker,
                     linestyle=linestyle[ranks], label=label)
    axes[2].axhline(1.0, color="black", linewidth=0.8)
    if sp.empty:
        axes[2].text(0.5, 0.5, "no MPI baseline in data", ha="center", va="center",
                     transform=axes[2].transAxes)

    xlabel = ("local size n  (n x n cells per rank)" if scaling == "weak"
              else "global size n  (n x n cells in total)")
    titles = ["Time per step", "Throughput", "Speedup vs MPI"]
    ylabels = ["seconds / step", "cell updates / s", "t_MPI / t_variant"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(f"{title} ({scaling} scaling)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="result file(s) written by gs_halo_bench")
    ap.add_argument("-o", "--output", help="output image (a _weak/_strong suffix is added "
                                           "when both scalings are present)")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load(args.csv)
    if df.empty:
        raise SystemExit("no data found")

    stem = Path(args.output) if args.output else Path(args.csv[0]).with_suffix(".png")
    scalings = sorted(df["scaling"].unique())
    for scaling in scalings:
        out = stem if len(scalings) == 1 else stem.with_name(f"{stem.stem}_{scaling}{stem.suffix}")
        plot_scaling(df[df["scaling"] == scaling], scaling, out, plt)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()