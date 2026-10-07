"""Chart how well the Betfair SP prices the winner (the owner, 7 Oct).

Reads the output of research/queries/done/bsp_calibration.py (kept as reports/bsp_calibration.txt) and draws:
the three R-squared readings and the favourites' strike rate as tiles; the win rate against the BSP's chance in
20 equal-count groups (log scales, the line where they would be equal); and winners against the BSP's expected
winners by the BSP's chance, with 95% intervals. Both windows (10 and 5 years) on each.

    python scripts/bsp_calibration_chart.py [--txt reports/bsp_calibration.txt] [--out reports/bsp_calibration.png]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# the reference palette (light): surface, ink, chrome; series slots 1 and 2
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BORDER = "#e1e0d9", "#c3c2b7", (11 / 255, 11 / 255, 11 / 255, 0.10)
SERIES = {"10": "#2a78d6", "5": "#eb6834"}
NAMES = {"10": "10 years (Apr 2016 - Mar 2026)", "5": "5 years (Apr 2021 - Mar 2026)"}
PX = 72 / 96                     # one CSS pixel in points
BANDS = ["<1%", "1-2%", "2-3%", "3-5%", "5-7.5%", "7.5-10%", "10-15%", "15-20%", "20-25%", "25-30%", "30-40%",
         "40-50%", "50-60%", "60-70%", "70-80%", "80%+"]
BAND_BSP = ["100+", "50-100", "33-50", "20-33", "13-20", "10-13", "6.7-10", "5-6.7", "4-5", "3.3-4", "2.5-3.3",
            "2-2.5", "1.7-2", "1.4-1.7", "1.25-1.4", "<1.25"]


def parse(text: str) -> dict:
    """The query's printout: per window the measures, the fixed-band table and the 20 bins; the season table."""
    out: dict = {"10": {}, "5": {}}
    win, block = None, None
    for line in text.splitlines():
        if line.startswith("==== "):
            m = re.match(r"==== (\d+) years", line)
            win, block = (m.group(1) if m else None), ("measures" if m else "season")
            if win:
                out[win] = {"measures": {}, "bands": [], "bins": []}
            continue
        if line.startswith("-- calibration"):
            block = "bands"
            continue
        if line.startswith("-- ") or not line.strip() or line.startswith("("):
            if line.startswith("-- "):
                block = "other"
            continue
        if line.startswith("BIN|"):
            _, w, chance, won, n = line.split("|")
            out[w]["bins"].append((float(chance), float(won), int(n)))
            continue
        parts = line.split()
        if win and block == "measures" and len(parts) == 2:
            out[win]["measures"][parts[0]] = float(parts[1])
        elif win and block == "bands" and re.match(r"^\d", parts[0]) and len(parts) == 7:
            n, chance, _, _, _, ae = int(parts[1]), *map(float, parts[2:])
            out[win]["bands"].append((parts[0], n, chance, ae))
        elif block == "season" and re.match(r"^\d{4}-\d{2}$", parts[0]):
            out.setdefault("seasons", []).append((parts[0], *map(float, parts[1:])))
    return out


def ae_interval(n: int, chance: float, ae: float) -> tuple[float, float]:
    """95% interval of winners / expected: the win rate's binomial error, relative to it."""
    w = ae * chance
    rel = 1.96 * np.sqrt((1 - w) / (n * w))
    return ae * (1 - rel), ae * (1 + rel)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1 * PX)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8.5, width=1 * PX, length=3)
    ax.grid(True, color=GRID, linewidth=1 * PX, linestyle="-")
    ax.set_axisbelow(True)


def tile(fig, x, top, w, h, value: str, label: str, sub: str) -> None:
    """A stat tile: the figure, what it is, and its 5-year and by-season readings (all in figure fractions)."""
    fig.patches.append(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.006",
                                      transform=fig.transFigure, facecolor=SURFACE, edgecolor=BORDER,
                                      linewidth=1 * PX))
    pad = 0.016
    fig.text(x + pad, top - 0.008, value, fontsize=20, color=INK, va="top", fontweight="bold")
    fig.text(x + pad, top - 0.037, label, fontsize=8, color=INK2, va="top", linespacing=1.35)
    fig.text(x + pad, top - h + 0.007, sub, fontsize=7.5, color=MUTED, va="bottom", linespacing=1.35)


def pct(x: float) -> str:
    """A share to one decimal, halves rounded up (0.3445 -> 34.5%)."""
    return f"{x * 100 + 1e-9:.1f}%"


def chart(data: dict, out: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    H = 15.2                                                     # inches; positions below are from the top
    fig = plt.figure(figsize=(9, H), dpi=150, facecolor=SURFACE)

    def y(inches: float) -> float:
        return 1 - inches / H

    m10, m5 = data["10"]["measures"], data["5"]["measures"]
    runners, races = int(m10["runners"]), int(m10["races"])
    fig.text(0.06, y(0.35), "The Betfair SP is a calibrated win probability", fontsize=16, color=INK,
             fontweight="bold", va="top")
    fig.text(0.06, y(0.72), f"GB and IE races, April 2016 to March 2026: {runners:,} runners in {races:,} races. "
             "Each runner's chance is 1/BSP,\nscaled so the race sums to 100% (the book at the BSP averages "
             f"{m10['book'] * 100:.1f}%). The holdout from 1 April 2026 is not read.", fontsize=9, color=INK2,
             va="top", linespacing=1.4)

    seasons = data.get("seasons", [])
    lo_b = min(s[4] for s in seasons)
    lo_r, hi_r = min(s[5] for s in seasons), max(s[5] for s in seasons)
    lo_m, hi_m = min(s[6] for s in seasons), max(s[6] for s in seasons)
    tiles = [
        (f"{m10['r2_bins']:.4f}", "R² across 20 price groups:\nwin rate on BSP chance",
         f"5 years {m5['r2_bins']:.4f}\nevery season {lo_b:.3f}+"),
        (f"{m10['r2_efron']:.3f}", "R² per runner: the share\nof won-or-lost explained",
         f"5 years {m5['r2_efron']:.3f}\nseasons {lo_r:.3f}-{hi_r:.3f}"),
        (f"{m10['r2_mcfadden']:.3f}", "pseudo-R² per race\n(McFadden)",
         f"5 years {m5['r2_mcfadden']:.3f}\nseasons {lo_m:.3f}-{hi_m:.3f}"),
        (pct(m10["fav_win"]), f"BSP favourites won;\npriced at {pct(m10['fav_chance'])}",
         f"5 years {pct(m5['fav_win'])} won,\npriced at {pct(m5['fav_chance'])}"),
    ]
    x0, gap = 0.06, 0.012
    w = (0.88 - 3 * gap) / 4
    for i, (v, lab, sub) in enumerate(tiles):
        tile(fig, x0 + i * (w + gap), y(1.45), w, 1.5 / H, v, lab, sub)

    # -- the win rate against the BSP's chance, 20 equal-count groups
    side = 6.0
    ax = fig.add_axes([(9 - side) / 2 / 9 + 0.03, y(3.75 + side), side / 9, side / H])
    style(ax)
    lim = (0.001, 0.62)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(lim, lim, color=MUTED, linewidth=1 * PX, zorder=1, label="win rate = the BSP's chance")
    for key, size in (("10", 12), ("5", 8)):
        b = np.array(data[key]["bins"])
        ax.plot(b[:, 0], b[:, 1], linestyle="none", marker="o", markersize=size * PX,
                markerfacecolor=SERIES[key], markeredgecolor=SURFACE, markeredgewidth=2 * PX, zorder=3,
                label=NAMES[key])
    ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t * 100:g}%\n{1 / t:,.0f}" for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t * 100:g}%" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("The BSP's chance (and the BSP)", fontsize=9, color=INK2, labelpad=6)
    ax.set_ylabel("Share that won", fontsize=9, color=INK2)
    fig.text(0.06, y(3.3), "Win rate against the BSP's chance: 20 groups of equal size (log scales)",
             fontsize=10.5, color=INK, va="top")
    b10 = data["10"]["bins"][0]
    ax.annotate(f"the longest-priced 5%: BSP's chance {b10[0] * 100:.2f}%,\nwon {b10[1] * 100:.2f}%",
                xy=(b10[0], b10[1]), xytext=(0.0034, 0.00125), fontsize=8, color=INK2, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1 * PX, shrinkA=2, shrinkB=6))
    top = data["10"]["bins"][-1]
    ax.annotate(f"the shortest 5%: BSP's chance {top[0] * 100:.1f}%,\nwon {top[1] * 100:.1f}%",
                xy=(top[0], top[1]), xytext=(0.58, 0.075), fontsize=8, color=INK2, va="center", ha="right",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1 * PX, shrinkA=2, shrinkB=6))
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5, handletextpad=0.5, borderaxespad=0.6)
    for t in leg.get_texts():
        t.set_color(INK2)

    # -- winners against the BSP's expected winners, fixed bands of the chance
    bx = fig.add_axes([0.10, y(14.05), 0.84, 2.85 / H])
    style(bx)
    bx.axhline(1.0, color=MUTED, linewidth=1 * PX, zorder=1)
    xs = np.arange(len(BANDS))
    for key, dx in (("10", -0.17), ("5", 0.17)):
        rows = data[key]["bands"]
        ae = np.array([r[3] for r in rows])
        lo, hi = np.array([ae_interval(r[1], r[2], r[3]) for r in rows]).T
        bx.vlines(xs + dx, lo, hi, color=SERIES[key], linewidth=2 * PX, zorder=2)
        bx.plot(xs + dx, ae, linestyle="none", marker="o", markersize=8 * PX, markerfacecolor=SERIES[key],
                markeredgecolor=SURFACE, markeredgewidth=2 * PX, zorder=3, label=NAMES[key])
    bx.set_xticks(xs)
    bx.set_xticklabels([f"{a}\n{b}" for a, b in zip(BANDS, BAND_BSP)], fontsize=7.2)
    bx.set_xlim(-0.6, len(BANDS) - 0.4)
    bx.set_ylim(0.76, 1.12)
    bx.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10])
    bx.set_yticklabels(["0.80", "0.85", "0.90", "0.95", "1.00", "1.05", "1.10"])
    bx.grid(False, axis="x")
    bx.set_xlabel("The BSP's chance (top) and the BSP (below)", fontsize=9, color=INK2, labelpad=6)
    bx.set_ylabel("Winners ÷ expected", fontsize=9, color=INK2)
    fig.text(0.06, y(10.85), "Winners against the BSP's expected winners, by the BSP's chance "
             "(lines: 95% intervals)", fontsize=10.5, color=INK, va="top")
    r10, r5 = data["10"]["bands"][0], data["5"]["bands"][0]
    bx.annotate(f"BSP over 100: {r10[3]:.2f} (10 years), {r5[3]:.2f} (5 years),\nthe only clear miss",
                xy=(0.17, r5[3] - 0.012), xytext=(0.9, 0.80), fontsize=8, color=INK2, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1 * PX, shrinkA=2, shrinkB=4))
    leg = bx.legend(loc="upper right", frameon=False, fontsize=8, ncol=2, handletextpad=0.3, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.text(0.06, y(14.95), "Source: horse_racing.db race_results (Betfair win SP), research/queries/done/"
             "bsp_calibration.py; dead heats and races missing a BSP left out.", fontsize=7.5, color=MUTED)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--txt", default="reports/bsp_calibration.txt")
    ap.add_argument("--out", default="reports/bsp_calibration.png")
    a = ap.parse_args()
    data = parse(Path(a.txt).read_text())
    chart(data, Path(a.out))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
