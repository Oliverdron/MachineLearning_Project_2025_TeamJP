import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# create logger for this module
logger = logging.getLogger(__name__)
plt.set_loglevel("warning")  # suppress matplotlib debug/info logs

# Colors for plotting
COLOR_ZERO = "#234C6A"   # ClaimNb = 0
COLOR_POS  = "#0C2B4E"   # ClaimNb > 0


def plot_distributions(
    train: pd.DataFrame,
    figures_dir: str,
    claim_col: str = "ClaimNb",
) -> None:
    """
        Main plotting function to generate and save all plots.

        Args:
            train (pd.DataFrame): Training dataset
            test (pd.DataFrame): Testing dataset
            figures_dir (str): Directory to save the figures
            claim_col (str): Name of the claim column in the training dataset

        Returns:
            None
    """
    # Create output directories
    base = Path(figures_dir)
    cat_dir = base / "categorical"
    num_dir = base / "numerical"

    # Make sure directories exist or create them
    cat_dir.mkdir(parents=True, exist_ok=True)
    num_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting started | cat_dir=%s | num_dir=%s", cat_dir, num_dir)

    # Categorical plots require claim_col
    plot_categorical_distributions(train=train, out_dir=cat_dir, claim_col=claim_col)

    # Numerical plots require claim_col for some plots
    plot_numerical_distributions(train=train, out_dir=num_dir, claim_col=claim_col)

    logger.info("Plotting finished")


def plot_categorical_distributions(train: pd.DataFrame, out_dir: Path, claim_col: str = "ClaimNb") -> None:
    """
        Saves all categorical distribution plots (train only, split by claim status).

        Args:
            train (pd.DataFrame): Training dataset
            out_dir (Path): Directory to save the figures
            claim_col (str): Name of the claim column in the training dataset

        Returns:
            None
    """
    # Check claim_col presence
    if claim_col not in train.columns:
        logger.warning("Categorical plots skipped | missing claim_col=%s in train", claim_col)
        raise ValueError(f"Missing claim_col={claim_col} in train DataFrame")

    # VehBrand – Top 15 horizontal stacked
    _plot_topn_stacked_barh(
        df=train,
        group_col="VehBrand",
        claim_col=claim_col,
        top_n=15,
        title="VehBrand: Top 15 brands (stacked by claim status)",
        xlabel="Number of policies",
        out_path=out_dir / "vehbrand_top15_stacked_barh.png",
    )

    # Region – Top 20 horizontal stacked
    _plot_topn_stacked_barh(
        df=train,
        group_col="Region",
        claim_col=claim_col,
        top_n=20,
        title="Region: Top 20 regions (stacked by claim status)",
        xlabel="Number of policies",
        out_path=out_dir / "region_top20_stacked_barh.png",
    )

    # Area – stacked vertical A–F
    _plot_stacked_bar(
        df=train,
        group_col="Area",
        claim_col=claim_col,
        title="Area: counts by claim status",
        xlabel="Area",
        ylabel="Number of policies",
        out_path=out_dir / "area_stacked_bar.png",
        sort_index=True,
        rotate_xticks=0,
        annotate_newline=False,
    )

    # VehGas – stacked vertical Diesel/Regular
    _plot_stacked_bar(
        df=train,
        group_col="VehGas",
        claim_col=claim_col,
        title="VehGas: counts by claim status",
        xlabel="Fuel type (VehGas)",
        ylabel="Number of policies",
        out_path=out_dir / "vehgas_stacked_bar.png",
        sort_index=True,
        rotate_xticks=0,
        annotate_newline=False,
    )

    # VehPower – stacked vertical (treat numeric as categorical)
    _plot_stacked_bar(
        df=train,
        group_col="VehPower",
        claim_col=claim_col,
        title="VehPower: counts by claim status",
        xlabel="VehPower (category)",
        ylabel="Number of policies",
        out_path=out_dir / "vehpower_stacked_bar.png",
        sort_index=True,
        rotate_xticks=90,  # typically many categories
        annotate_newline=True,
    )


def _build_claim_summary(df: pd.DataFrame, group_col: str, claim_col: str) -> pd.DataFrame:
    """
        Builds a summary DataFrame with total counts, zero claim counts, positive claim counts,
        and claim rates for each category in the specified grouping column.

        Args:
            df (pd.DataFrame): The input DataFrame
            group_col (str): The column to group by
            claim_col (str): The claim number column

        Returns:
            pd.DataFrame: Summary DataFrame with total, zero, positive counts and claim rates
    """
    # grouping key as string for consistent grouping
    g = df[group_col].astype("string")

    # sanity: ensure claim number is numeric and if not, convert to NaN then 0
    claims = pd.to_numeric(df[claim_col], errors="coerce").fillna(0).astype(int)

    # group by partitions the rows into groups, then we can aggregate
    total = claims.groupby(g).size()
    # use a mask to count zero claims, where True=1 and False=0 and sum over groups
    zero = (claims == 0).groupby(g).sum()
    # use a mask to count positive claims, where True=1 and False=0 and sum over groups
    pos = (claims > 0).groupby(g).sum()

    # build summary DataFrame with a fallback of filling missing groups with 0
    # convert counts to int
    summary = pd.DataFrame({"total": total, "zero": zero, "pos": pos}).fillna(0).astype(int)
    # compute claim rate percentage
    summary["claim_rate"] = summary["pos"] / summary["total"].replace(0, np.nan)

    # drop NA group label if present
    summary = summary.drop(index=[pd.NA], errors="ignore")
    return summary


def _plot_topn_stacked_barh(
    df: pd.DataFrame,
    group_col: str,
    claim_col: str,
    top_n: int,
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    """
        Plots a horizontal stacked bar chart for the top N categories in the specified grouping column.
        
        Args:
            df (pd.DataFrame): The input DataFrame
            group_col (str): The column to group by
            claim_col (str): The claim number column
            top_n (int): Number of top categories to plot
            title (str): Title of the plot
            xlabel (str): Label for the x-axis
            out_path (Path): Path to save the output plot

        Returns:
            None
    """
    # Sanity check
    if group_col not in df.columns:
        logger.warning("Plot skipped | missing col=%s", group_col)
        return

    # get the column dependent summary
    summary = _build_claim_summary(df, group_col, claim_col)
    # check for empty summary
    if summary.empty:
        logger.warning("Plot skipped | empty summary | group_col=%s", group_col)
        return

    # if there are multiple groups in a column, pick top N by total count
    summary = summary.sort_values("total", ascending=False).head(top_n)

    # figure size based on number of bars
    fig_h = max(4.0, 0.35 * len(summary) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    # set y positions for each bar
    ypos = np.arange(len(summary))

    # plot zero and non-zero claims as horizontal bars on top of each other
    ax.barh(ypos, summary["zero"], color=COLOR_ZERO, label="ClaimNb = 0")
    ax.barh(ypos, summary["pos"], left=summary["zero"], color=COLOR_POS, label="ClaimNb > 0")

    # set y ticks and labels
    ax.set_yticks(ypos)
    ax.set_yticklabels(summary.index.astype(str))
    ax.invert_yaxis()

    # add labels and title
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()

    # calculate a horizontal offset, that defines the location of the text label to the right of the bar
    x_offset = summary["total"].max() * 0.005

    # iterate over each group, the index gives the bar index, which is the y's position
    for i in range(len(summary)):
        # fetch the total policies for the given group
        tot = summary["total"].iloc[i]
        # fetch claimrate if possible, otherwise display 0 instead of crashing
        rate = float(summary["claim_rate"].iloc[i]) if pd.notna(summary["claim_rate"].iloc[i]) else 0.0
        # fetch zero and positive counts to compute the end of the bar, then add offset for the text position
        z = summary["zero"].iloc[i]
        p = summary["pos"].iloc[i]
        x_end = z + p
        ax.text(
            x_end + x_offset,
            i,
            f"{tot:,} ({rate * 100:.1f}% claims)",
            ha="left",
            va="center",
            fontsize=8,
            color="black",
        )

    # avoid cutting off labels and save figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    
    # log saved plot info
    logger.info("Saved plot | %s", out_path)


def _plot_stacked_bar(
    df: pd.DataFrame,
    group_col: str,
    claim_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    sort_index: bool = True,
    rotate_xticks: int = 0,
    annotate_newline: bool = False,
) -> None:
    """
        Plots a vertical stacked bar chart for the specified grouping column.
        
        Args:
            df (pd.DataFrame): The input DataFrame
            group_col (str): The column to group by
            claim_col (str): The claim number column
            title (str): Title of the plot
            xlabel (str): Label for the x-axis
            ylabel (str): Label for the y-axis
            out_path (Path): Path to save the output plot
            sort_index (bool): Whether to sort the index of the summary DataFrame
            rotate_xticks (int): Rotation angle for x-tick labels
            annotate_newline (bool): Whether to use newline in annotations
            
        Returns:
            None
    """
    # Sanity check
    if group_col not in df.columns:
        logger.warning("Plot skipped | missing col=%s", group_col)
        return

    # get the column dependent summary
    summary = _build_claim_summary(df, group_col, claim_col)
    # check for empty summary
    if summary.empty:
        logger.warning("Plot skipped | empty summary | group_col=%s", group_col)
        return

    # Sorting logic: VehPower needs numeric ordering; others can use label ordering
    try:
        # Only sort if necessary
        if sort_index:
            # Special case for VehPower numeric sorting
            if group_col == "VehPower":
                # Force numeric sort even if index is stringy (e.g. "10" vs "2")
                idx_num = pd.to_numeric(summary.index.astype(str), errors="coerce")
                order = np.argsort(idx_num.fillna(1e18).to_numpy())  # non-numeric -> pushed to end
                summary = summary.iloc[order]
                logger.debug("Sorted summary by numeric index | group_col=%s", group_col)
            else:
                # Default: lexicographic label sort (Area A–F, VehGas Diesel/Regular, etc.)
                summary = summary.sort_index()
                logger.debug("Sorted summary by label index | group_col=%s", group_col)
    except Exception:
        logger.exception("Failed to sort summary | group_col=%s | sort_index=%s", group_col, sort_index)

    # figure size based on number of bars
    fig_w = max(7.0, 0.45 * len(summary))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    # x position for each bar to draw
    xpos = np.arange(len(summary))

    # plot zero and non-zero claims as bars on top of each other
    ax.bar(xpos, summary["zero"], color=COLOR_ZERO, label="ClaimNb = 0")
    ax.bar(xpos, summary["pos"], bottom=summary["zero"], color=COLOR_POS, label="ClaimNb > 0")

    # set ticks for the given x positions
    ax.set_xticks(xpos)
    # set labels for each bar (rotate labels if requested for readability)
    ax.set_xticklabels(summary.index.astype(str), rotation=rotate_xticks)
    # set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    # calculate a vertical offset, that defines the location of the text label above the bar
    y_offset = summary["total"].max() * 0.01
    # iterate over each group, the index gives the bar index, which is the x's position
    for i in range(len(summary)):
        # fetch the total policies for the given group
        tot = int(summary["total"].iloc[i])
        # fetch claimrate if possible, otherwise display 0 instead of crashing
        rate = float(summary["claim_rate"].iloc[i]) if pd.notna(summary["claim_rate"].iloc[i]) else 0.0
        # build label with or without newline
        label = f"{tot:,}\n({rate * 100:.1f}% claims)" if annotate_newline else f"{tot:,} ({rate * 100:.1f}% claims)"
        # place text label above the bar
        ax.text(
            xpos[i],
            tot + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=7 if annotate_newline else 8,
            color="black",
        )
    
    # avoid cutting off labels and save figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    # log saved plot info
    logger.info("Saved plot | %s", out_path)


def plot_numerical_distributions(train: pd.DataFrame, out_dir: Path, claim_col: str = "ClaimNb") -> None:
    """
        Saves all numerical distribution plots (train only).

        Args:
            train (pd.DataFrame): Training dataset
            out_dir (Path): Directory to save the figures
            claim_col (str): Name of target column

        Returns:
            None
    """
    logger.info("Numerical plotting started | out_dir=%s", out_dir)

    # 1) Density - hist(counts) + KDE + IQR
    _plot_hist_kde_iqr(
        df=train,
        col="Density",
        out_path=out_dir / "density_hist_kde_iqr.png",
        title="Density: Histogram (counts) + KDE + IQR",
        xlabel="Density",
        xlim=None,
        xs_range=None,
        clip_iqr=None,
        focus_right=None,
    )

    # 2) Exposure - hist(counts) + KDE + IQR (0-1 focus)
    _plot_hist_kde_iqr(
        df=train,
        col="Exposure",
        out_path=out_dir / "exposure_hist_kde_iqr_0_1.png",
        title="Exposure: Histogram + KDE + IQR (0-1 focus)",
        xlabel="Exposure",
        xlim=(0.0, 1.0),
        xs_range=(0.0, 1.0),
        clip_iqr=(0.0, 1.0),
        focus_right=None,
    )

    # 3) ClaimNb - binned bar + log y + meaning labels (train only)
    _plot_claimnb_binned_log(
        df=train,
        claim_col=claim_col,
        out_path=out_dir / "claimnb_binned_log.png",
    )

    # 4) VehAge - binned stacked (zero vs pos) + counts + %
    _plot_binned_stacked_claims(
        df=train,
        value_col="VehAge",
        claim_col=claim_col,
        bins=[0, 3, 6, 11, 21],
        xtick_labels=[
            "0-2\n(brand-new)",
            "3-5\n(fairly new)",
            "6-10\n(average)",
            "11-20\n(older car)",
        ],
        title="VehAge: counts by claim status",
        xlabel="Vehicle age bin (years)",
        out_path=out_dir / "vehage_binned_stacked.png",
    )

    # 5) DrivAge - binned stacked (zero vs pos) + counts + %
    _plot_drivage_binned_stacked(
        df=train,
        claim_col=claim_col,
        out_path=out_dir / "drivage_binned_stacked.png",
    )

    # 6) BonusMalus - hist(counts) + KDE + IQR (focus on main mass)
    _plot_hist_kde_iqr(
        df=train,
        col="BonusMalus",
        out_path=out_dir / "bonusmalus_hist_kde_iqr.png",
        title="BonusMalus: Histogram (counts) + KDE + IQR",
        xlabel="BonusMalus",
        xlim=None,
        xs_range=None,
        clip_iqr=None,
        focus_right=1.2,  # xlim_right = iqr_upper * 1.2 (clipped to x.max)
    )

    logger.info("Numerical plotting finished")


def _plot_hist_kde_iqr(
    df: pd.DataFrame,
    col: str,
    out_path: Path,
    title: str,
    xlabel: str,
    xlim: tuple[float, float] | None,
    xs_range: tuple[float, float] | None,
    clip_iqr: tuple[float, float] | None,
    focus_right: float | None,
    bins: int = 50,
) -> None:
    """
        Histogram (counts) + KDE + IQR fences plot for a numerical column.

        Args:
            df (pd.DataFrame): Input DataFrame
            col (str): Column name to plot
            out_path (Path): Path to save the output plot
            title (str): Title of the plot
            xlabel (str): Label for the x-axis
            xlim (tuple[float, float] | None): x-axis limits (min, max) or None for auto
            xs_range (tuple[float, float] | None): x-range for KDE evaluation or None for auto
            clip_iqr (tuple[float, float] | None): (min, max) to clip IQR fences or None for no clipping
            focus_right (float | None): If set, xlim right = min(x.max, iqr_upper * focus_right)
            bins (int): Number of histogram bins

        Returns:
            None
    """
    # Sanity check
    if col not in df.columns:
        logger.warning("Plot skipped | missing col=%s", col)
        return

    # Make sure data is numeric
    x = pd.to_numeric(df[col], errors="coerce").dropna().astype(float).values
    # If no valid numeric data, skip plot
    if x.size == 0:
        logger.warning("Plot skipped | empty data after numeric coercion | col=%s", col)
        return

    # Initialize figure
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    # Histogram
    _, bin_edges, _ = ax.hist(x, bins=bins, color=COLOR_POS, alpha=0.7)
    # Make sure we have at least 2 bins to compute bin width
    if len(bin_edges) < 2:
        logger.warning("Plot skipped | insufficient bins | col=%s", col)
        plt.close(fig)
        return
    # Compute bin width from edges
    bin_width = float(bin_edges[1] - bin_edges[0])

    # KDE scaled to counts: kde(x) * N * bin_width
    try:
        from scipy.stats import gaussian_kde
        if x.size >= 2 and np.std(x) > 0:
            # KDE (kernel density estimate) is a probability density whose area integrates to 1
            # which is not on the same scale as histogram counts, so we scale it
            kde = gaussian_kde(x)
            # define range for KDE evaluation based on user input, otherwise use data min/max
            xs_min = xs_range[0] if xs_range else float(np.min(x))
            xs_max = xs_range[1] if xs_range else float(np.max(x))
            # create evenly spaced values for KDE evaluation with 500 points
            xs = np.linspace(xs_min, xs_max, 500)
            # kde(xs) gives density values, scale to counts by multiplying by N * bin_width
            ax.plot(xs, kde(xs) * len(x) * bin_width)
        else:
            logger.warning("KDE skipped | insufficient variance/samples | col=%s | n=%d", col, x.size)
    except Exception:
        logger.exception("KDE failed | col=%s (hist + IQR still saved)", col)

    # IQR fences
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower = float(q1 - 1.5 * iqr)
    upper = float(q3 + 1.5 * iqr)

    # clip IQR fences if requested (Exposure wants [0,1])
    if clip_iqr is not None:
        lower = max(lower, float(clip_iqr[0]))
        upper = min(upper, float(clip_iqr[1]))

    # draw dashed lines for IQR fences
    ax.axvline(lower, linestyle="--", color="black")
    ax.axvline(upper, linestyle="--", color="black")

    # two modes for x-limit:
    if focus_right is not None:
        # focus_right: ignores far right tail beyond iqr_upper * focus_right
        right = min(float(np.max(x)), upper * float(focus_right))
        # set xlim from min to computed right
        ax.set_xlim(float(np.min(x)), right)
    elif xlim is not None:
        # user-specified xlim
        ax.set_xlim(xlim[0], xlim[1])

    # labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    # avoid cutting off labels and save figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    logger.info("Saved plot | %s", out_path)


def _plot_claimnb_binned_log(df: pd.DataFrame, claim_col: str, out_path: Path) -> None:
    """
        ClaimNb binned bar plot with log y-axis and meaningful x-tick labels.
        Bins:
            0 (profit)
            1 (reasonable)
            2-4 (rare)
            5+ (suspicious fraud)

        Args:
            df (pd.DataFrame): Input DataFrame
            claim_col (str): Claim number column
            out_path (Path): Path to save the output plot

        Returns:
            None
    """
    # Sanity check
    if claim_col not in df.columns:
        logger.warning("Plot skipped | missing claim_col=%s", claim_col)
        return

    # Make sure claim numbers are numeric
    c = pd.to_numeric(df[claim_col], errors="coerce").fillna(0).astype(int).values
    if c.size == 0:
        logger.warning("Plot skipped | empty claims | claim_col=%s", claim_col)
        return

    # Define edges of bins
    bins = [0, 1, 2, 5, int(c.max()) + 1]
    # Bin the claim numbers, left-inclusive and use c.max()+1 to include max value, -1 to get 0-based bin index
    idx = np.digitize(c, bins, right=False) - 1
    # Force indices to be within [0, 3]: 0=0, 1=1, 2=2-4, 3=5+
    idx = np.clip(idx, 0, 3)
    # Count occurrences in each bin
    counts = np.bincount(idx, minlength=4)

    # Define x-tick labels that are meaningful
    xtick_labels = [
        "0\n(profit)",
        "1\n(reasonable)",
        "2-4\n(rare)",
        "5+\n(suspicious fraud)",
    ]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(range(4), counts, color=COLOR_POS)

    # Scale y-axis so 0 claims don't dominate
    ax.set_yscale("log")

    # white labels on each bar
    for bar, count in zip(bars, counts):
        ax.text(
            # center of the bar
            bar.get_x() + bar.get_width() / 2,
            # top of the bar
            bar.get_height(),
            f"{int(count):,}",
            ha="center",
            va="bottom",
            color="white",
            fontsize=9,
        )

    # labels and title
    ax.set_xticks(range(4))
    ax.set_xticklabels(xtick_labels)
    ax.set_title("ClaimNb: Binned counts (log scale)")
    ax.set_ylabel("Number of policies (log scale)")

    # avoid cutting off labels and save figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    logger.info("Saved plot | %s", out_path)


def _plot_binned_stacked_claims(
    df: pd.DataFrame,
    value_col: str,
    claim_col: str,
    bins: list[float],
    xtick_labels: list[str],
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    """
        Binned stacked bar plot for a numerical column showing counts of zero vs positive claims.

        Args:
            df (pd.DataFrame): Input DataFrame
            value_col (str): Numerical column to bin
            claim_col (str): Claim number column
            bins (list[float]): List of bin edges
            xtick_labels (list[str]): Labels for each bin
            title (str): Title of the plot
            xlabel (str): Label for the x-axis
            out_path (Path): Path to save the output plot

        Returns:
            None
    """
    # Sanity check
    if value_col not in df.columns:
        logger.warning("Plot skipped | missing col=%s", value_col)
        return
    
    # Sanity check
    if claim_col not in df.columns:
        logger.warning("Plot skipped | missing claim_col=%s", claim_col)
        return

    # Make sure data is numeric
    values = pd.to_numeric(df[value_col], errors="coerce").astype(float).values
    claims = pd.to_numeric(df[claim_col], errors="coerce").fillna(0).astype(int).values

    # Bin assignment
    idx = np.digitize(values, bins, right=False) - 1
    # Clip indices to valid range
    n_bins = len(xtick_labels)
    idx = np.clip(idx, 0, n_bins - 1)

    # Initialize counts (zero, positive, total) for each bin
    zero = np.zeros(n_bins, dtype=int)
    pos = np.zeros(n_bins, dtype=int)
    total = np.zeros(n_bins, dtype=int)

    # Count occurrences in each bin
    for b in range(n_bins):
        # After assigning each value to a bin, create a mask that selects only values in bin b
        mask = idx == b
        # Count zero and positive claims
        z = int(np.sum(claims[mask] == 0))
        p = int(np.sum(claims[mask] > 0))
        # Store counts
        zero[b] = z
        pos[b] = p
        total[b] = z + p

    # Compute percentages for annotations
    pct_zero = np.divide(zero, total, out=np.zeros_like(zero, dtype=float), where=total > 0)
    pct_pos = np.divide(pos, total, out=np.zeros_like(pos, dtype=float), where=total > 0)

    # Initialize figure
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    xpos = np.arange(n_bins)

    # Plot stacked bars
    ax.bar(xpos, zero, color=COLOR_ZERO, label="ClaimNb = 0")
    ax.bar(xpos, pos, bottom=zero, color=COLOR_POS, label="ClaimNb > 0")

    # Set ticks and labels
    ax.set_xticks(xpos)
    ax.set_xticklabels(xtick_labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of policies")
    ax.legend()

    # Vertical offset for text annotations, so they don't overlap with bars
    y_offset = total.max() * 0.01 if total.max() > 0 else 1.0

    # Annotations for each bin
    for b in range(n_bins):
        # Zero claims annotation inside the bar
        ax.text(
            xpos[b],
            zero[b] / 2 if zero[b] > 0 else 0.0,
            f"{zero[b]:,}\n({pct_zero[b] * 100:.1f}%)",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

        # Positive claims annotation above the bar
        ax.text(
            xpos[b],
            total[b] + y_offset,
            f"{pos[b]:,}\n({pct_pos[b] * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    # Avoid cutting off labels and save figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    logger.info("Saved plot | %s", out_path)


def _plot_drivage_binned_stacked(df: pd.DataFrame, claim_col: str, out_path: Path) -> None:
    """
        DrivAge binned stacked bar plot showing counts of zero vs positive claims.
        Bins:
            18-25
            26-35
            36-45
            46-60
            60+

        Args:
            df (pd.DataFrame): Input DataFrame
            claim_col (str): Claim number column
            out_path (Path): Path to save the output plot

        Returns:
            None
    """
    # Sanity check
    if "DrivAge" not in df.columns:
        logger.warning("Plot skipped | missing col=DrivAge")
        return
    
    # Sanity check
    if claim_col not in df.columns:
        logger.warning("Plot skipped | missing claim_col=%s", claim_col)
        return

    # Make sure data is numeric
    age = pd.to_numeric(df["DrivAge"], errors="coerce").astype(float).values
    claims = pd.to_numeric(df[claim_col], errors="coerce").fillna(0).astype(int).values

    # Fetch the maximum age ingoring NaNs; if all NaN, default to 61.0
    max_age = float(np.nanmax(age)) if np.isfinite(np.nanmax(age)) else 61.0
    # Ensure last edge is at least 62.0 to cover 60+, if max_age is greater, extend by 1 so that max_age is included
    last_edge = max(62.0, max_age + 1.0)

    # Define bins and labels
    bins = [18, 26, 36, 46, 61, last_edge]
    xtick_labels = ["18-25", "26-35", "36-45", "46-60", "60+"]

    # Bin assignment, left-inclusive, -1 to get 0-based bin index
    idx = np.digitize(age, bins, right=False) - 1
    # Clip indices to valid range
    n_bins = len(xtick_labels)
    idx = np.clip(idx, 0, n_bins - 1)

    # Initialize counts (zero, positive, total) for each bin
    zero = np.zeros(n_bins, dtype=int)
    pos = np.zeros(n_bins, dtype=int)
    total = np.zeros(n_bins, dtype=int)

    # Count occurrences in each bin
    for b in range(n_bins):
        # After assigning each value to a bin, create a mask that selects only values in bin b
        mask = idx == b
        # Count zero and positive claims
        z = int(np.sum(claims[mask] == 0))
        p = int(np.sum(claims[mask] > 0))
        # Store counts
        zero[b] = z
        pos[b] = p
        total[b] = z + p

    # Compute percentages for annotations
    pct_zero = np.divide(zero, total, out=np.zeros_like(zero, dtype=float), where=total > 0)
    pct_pos = np.divide(pos, total, out=np.zeros_like(pos, dtype=float), where=total > 0)

    # Initialize figure
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    xpos = np.arange(n_bins)

    # Plot stacked bars
    ax.bar(xpos, zero, color=COLOR_ZERO, label="ClaimNb = 0")
    ax.bar(xpos, pos, bottom=zero, color=COLOR_POS, label="ClaimNb > 0")

    # Set ticks and labels
    ax.set_xticks(xpos)
    ax.set_xticklabels(xtick_labels)
    ax.set_title("DrivAge: counts by claim status")
    ax.set_xlabel("Driver age bin (years)")
    ax.set_ylabel("Number of policies")
    ax.legend()

    # Vertical offset for text annotations, so they don't overlap with bars
    y_offset = total.max() * 0.01 if total.max() > 0 else 1.0

    # Annotations for each bin
    for b in range(n_bins):
        # Zero claims annotation inside the bar
        ax.text(
            xpos[b],
            zero[b] / 2 if zero[b] > 0 else 0.0,
            f"{zero[b]:,}\n({pct_zero[b] * 100:.1f}%)",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

        # Positive claims annotation above the bar
        ax.text(
            xpos[b],
            total[b] + y_offset,
            f"{pos[b]:,}\n({pct_pos[b] * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    # Avoid cutting off labels and save figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    logger.info("Saved plot | %s", out_path)
