import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# create logger for this module
logger = logging.getLogger(__name__)

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

    # Numerical plots to be implemented


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
