# plot_helpers.py
import os
import matplotlib.pyplot as plt
import logging

# Create a charts directory if not already present
def ensure_chart_dir(dir_name="charts"):
    """
    Ensure the chart directory exists; return the path.
    """
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

# Plot reset points scatter chart
def plot_reset_points(symbol: str, greek_res: dict, today, chart_dir="charts"):
    """
    Plot reset points (price vs significance) for a given symbol.
    greek_res should include 'reset_points' as a list of dicts with keys 'price' and 'significance'.
    """
    rps = greek_res.get("reset_points", [])
    if not rps:
        return

    # Extract prices and significance values (handle numeric or string with '%')
    prices = []
    sigs = []
    for pt in rps:
        try:
            prices.append(float(pt.get("price", 0)))
        except Exception:
            continue
        # significance might be a float or string
        sig_val = pt.get("significance", 0)
        if isinstance(sig_val, str):
            sig_val = sig_val.strip().rstrip('%')
        try:
            sigs.append(float(sig_val))
        except Exception:
            sigs.append(0.0)

    plt.figure()
    plt.scatter(prices, sigs)
    plt.title(f"{symbol} Reset Points ({today})")
    plt.xlabel("Price")
    plt.ylabel("Significance (%)")

    out_file = os.path.join(chart_dir, f"{symbol}_reset_points_{today}.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    logging.info("Saved Reset Points chart to %s", os.path.abspath(out_file))

# Plot energy levels bar chart
def plot_energy_levels(symbol: str, greek_res: dict, today, chart_dir="charts"):
    """
    Plot energy levels (price vs strength) for a given symbol.
    greek_res should include 'energy_levels' as a list of dicts with keys 'price', 'strength', and 'direction'.
    """
    els = greek_res.get("energy_levels", [])
    if not els:
        return

    prices = []
    strengths = []
    directions = []
    for lvl in els:
        try:
            prices.append(float(lvl.get("price", 0)))
        except Exception:
            continue
        str_val = lvl.get("strength", 0)
        if isinstance(str_val, str):
            str_val = str_val.strip().rstrip('%')
        try:
            strengths.append(float(str_val))
        except Exception:
            strengths.append(0.0)
        directions.append(lvl.get("direction", "").lower())

    colors = ["green" if d == "support" else "red" for d in directions]

    plt.figure()
    plt.bar(prices, strengths, color=colors)
    plt.title(f"{symbol} Energy Levels ({today})")
    plt.xlabel("Price")
    plt.ylabel("Strength (%)")

    out_file = os.path.join(chart_dir, f"{symbol}_energy_levels_{today}.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    logging.info("Saved Energy Levels chart to %s", os.path.abspath(out_file))
