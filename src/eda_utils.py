import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ── Color Palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "primary": "#1d3557",
    "secondary": "#457b9d",
    "accent": "#e63946",
    "light": "#f1faee",
    "highlight": "#a8dadc"
}

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "axes.edgecolor": "#dddddd",
    "axes.linewidth": 0.8,
})

sns.set_palette([
    PALETTE["primary"],
    PALETTE["secondary"],
    PALETTE["highlight"],
    PALETTE["accent"]
])


# ── 1. Portfolio Summary ──────────────────────────────────────────────────────
def portfolio_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a high-level portfolio metrics summary table.
    """

    required_cols = ["TotalPremium", "TotalClaims"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print(f"Missing columns: {missing}")
        return pd.DataFrame()

    total_premiums = df["TotalPremium"].sum()
    total_claims = df["TotalClaims"].sum()

    if total_premiums == 0:
        print("Cannot compute loss ratio because TotalPremium is zero.")
        return pd.DataFrame()

    loss_ratio = total_claims / total_premiums
    claim_rate = (df["TotalClaims"] > 0).mean() * 100
    avg_claim = df[df["TotalClaims"] > 0]["TotalClaims"].mean()

    summary = pd.DataFrame({
        "Metric": [
            "Total Premiums Collected",
            "Total Claims Paid",
            "Overall Loss Ratio",
            "Claim Frequency (Rate %)",
            "Average Cost per Claim"
        ],
        "Value": [
            f"{total_premiums:,.2f}",
            f"{total_claims:,.2f}",
            f"{loss_ratio:.4f}",
            f"{claim_rate:.2f}%",
            f"{avg_claim:,.2f}"
        ]
    })

    return summary


# ── 2. Data Summarization ─────────────────────────────────────────────────────
def summarize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for all numerical columns.
    """

    if df.empty:
        print("DataFrame is empty.")
        return pd.DataFrame()

    summary = df.describe(include=[np.number]).T
    summary["skewness"] = df.select_dtypes(include=[np.number]).skew()
    summary["kurtosis"] = df.select_dtypes(include=[np.number]).kurt()

    return summary


def check_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of column data types and sample values.
    """

    if df.empty:
        print("DataFrame is empty.")
        return pd.DataFrame()

    return pd.DataFrame({
        "dtype": df.dtypes,
        "sample_value": df.iloc[0],
        "nunique": df.nunique()
    })


# ── 3. Data Quality ───────────────────────────────────────────────────────────
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a sorted report of missing values per column.
    """

    if df.empty:
        print("DataFrame is empty.")
        return pd.DataFrame()

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    report = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": (missing / len(df) * 100).round(2)
    }).sort_values("missing_pct", ascending=False)

    return report


# ── 4. Univariate Analysis ────────────────────────────────────────────────────
def plot_numerical_distributions(
    df: pd.DataFrame,
    columns: list,
    ncols: int = 3
) -> None:
    """
    Plot histograms for numerical columns.
    """

    if not columns:
        print("No numerical columns provided.")
        return

    nrows = -(-len(columns) // ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows)
    )

    axes = axes.flatten()

    for i, col in enumerate(columns):

        if col not in df.columns:
            print(f"{col} column not found.")
            continue

        axes[i].hist(
            df[col].dropna(),
            bins=40,
            color=PALETTE["primary"],
            edgecolor="white",
            alpha=0.85
        )

        axes[i].set_title(
            col,
            fontsize=11,
            fontweight="bold",
            color=PALETTE["primary"]
        )

        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Numerical Feature Distributions",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["primary"],
        y=1.01
    )

    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(
    df: pd.DataFrame,
    columns: list,
    ncols: int = 2
) -> None:
    """
    Plot bar charts for categorical columns.
    """

    if not columns:
        print("No categorical columns provided.")
        return

    nrows = -(-len(columns) // ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8 * ncols, 4 * nrows)
    )

    axes = axes.flatten()

    for i, col in enumerate(columns):

        if col not in df.columns:
            print(f"{col} column not found.")
            continue

        counts = df[col].value_counts().head(10)

        axes[i].barh(
            counts.index.astype(str),
            counts.values,
            color=PALETTE["secondary"],
            edgecolor="white"
        )

        axes[i].yaxis.set_tick_params(pad=5)

        axes[i].set_title(
            col,
            fontsize=11,
            fontweight="bold",
            color=PALETTE["primary"]
        )

        axes[i].set_xlabel("Count")
        axes[i].invert_yaxis()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Categorical Feature Distributions",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["primary"],
        y=1.01
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.30)
    plt.show()


# ── 5. Bivariate / Multivariate Analysis ─────────────────────────────────────
def plot_correlation_matrix(df: pd.DataFrame, columns: list) -> None:
    """
    Plot a heatmap of the correlation matrix.
    """

    missing = [col for col in columns if col not in df.columns]

    if missing:
        print(f"Missing columns: {missing}")
        return

    corr = df[columns].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )

    ax.set_title(
        "Correlation Matrix",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["primary"]
    )

    plt.tight_layout()
    plt.show()


def plot_premium_vs_claims(
    df: pd.DataFrame,
    hue_col: str = None
) -> None:
    """
    Scatter plot of TotalPremium vs TotalClaims.
    """

    required_cols = ["TotalPremium", "TotalClaims"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print(f"Missing columns: {missing}")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    if hue_col and hue_col in df.columns:

        categories = df[hue_col].dropna().unique()[:8]

        colors = [
            PALETTE["primary"],
            PALETTE["secondary"],
            PALETTE["accent"],
            PALETTE["highlight"],
            "#2a9d8f",
            "#e9c46a",
            "#f4a261",
            "#264653"
        ]

        for cat, color in zip(categories, colors):

            subset = df[df[hue_col] == cat]

            ax.scatter(
                subset["TotalPremium"],
                subset["TotalClaims"],
                label=str(cat),
                alpha=0.5,
                s=15,
                color=color
            )

        ax.legend(
            title=hue_col,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=8
        )

    else:
        ax.scatter(
            df["TotalPremium"],
            df["TotalClaims"],
            alpha=0.4,
            s=15,
            color=PALETTE["primary"]
        )

    ax.set_xlabel("Total Premium", fontsize=11)

    ax.set_ylabel("Total Claims", fontsize=11)

    ax.set_title(
        "Total Premium vs Total Claims",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["primary"]
    )

    plt.tight_layout()
    plt.show()


def plot_premium_vs_claims_by_postalcode(df: pd.DataFrame) -> None:
    """
    Scatter plot grouped by PostalCode.
    """

    required_cols = [
        "PostalCode",
        "TotalPremium",
        "TotalClaims"
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print(f"Missing columns: {missing}")
        return

    top_codes = df["PostalCode"].value_counts().head(8).index

    filtered = df[df["PostalCode"].isin(top_codes)].copy()

    filtered["PostalCode"] = filtered["PostalCode"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [
        PALETTE["primary"],
        PALETTE["secondary"],
        PALETTE["accent"],
        PALETTE["highlight"],
        "#2a9d8f",
        "#e9c46a",
        "#f4a261",
        "#264653"
    ]

    for code, color in zip(top_codes.astype(str), colors):

        subset = filtered[filtered["PostalCode"] == code]

        ax.scatter(
            subset["TotalPremium"],
            subset["TotalClaims"],
            label=code,
            alpha=0.5,
            s=15,
            color=color
        )

    ax.set_xlabel("Total Premium", fontsize=11)

    ax.set_ylabel("Total Claims", fontsize=11)

    ax.set_title(
        "Total Premium vs Total Claims by PostalCode",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["primary"]
    )

    ax.legend(
        title="PostalCode",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()


def plot_claim_rate_by_group(
    df: pd.DataFrame,
    group_col: str
) -> None:
    """
    Plot claim rate by a grouping column.
    """

    required_cols = [group_col, "TotalClaims"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print(f"Missing columns: {missing}")
        return

    claim_rate = (
        df.groupby(group_col)
        .apply(lambda x: (x["TotalClaims"] > 0).mean() * 100)
        .rename("ClaimRate")
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        claim_rate[group_col].astype(str),
        claim_rate["ClaimRate"],
        color=PALETTE["accent"],
        edgecolor="white"
    )

    ax.yaxis.set_tick_params(pad=5)

    ax.invert_yaxis()

    ax.set_xlabel("Claim Rate (%)", fontsize=11)

    ax.set_title(
        f"Claim Rate (%) by {group_col}",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["primary"]
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.25)
    plt.show()


# ── 6. Geographic Trends ──────────────────────────────────────────────────────
def plot_province_analysis(df: pd.DataFrame) -> None:
    """
    Compare average TotalPremium and TotalClaims by Province.
    """

    required_cols = [
        "Province",
        "TotalPremium",
        "TotalClaims"
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print(f"Missing columns: {missing}")
        return

    province_stats = (
        df.groupby("Province")[["TotalPremium", "TotalClaims"]]
        .mean()
        .sort_values("TotalPremium", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(province_stats))
    width = 0.35

    ax.bar(
        x - width / 2,
        province_stats["TotalPremium"],
        width,
        label="Avg Premium",
        color=PALETTE["primary"],
        edgecolor="white"
    )

    ax.bar(
        x + width / 2,
        province_stats["TotalClaims"],
        width,
        label="Avg Claims",
        color=PALETTE["accent"],
        edgecolor="white"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        province_stats.index,
        rotation=30,
        ha="right"
    )

    ax.set_title(
        "Average Premium & Claims by Province",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["primary"]
    )

    ax.set_ylabel("Amount")
    ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()


# ── 7. Outlier Detection ──────────────────────────────────────────────────────
def plot_boxplots(
    df: pd.DataFrame,
    columns: list,
    ncols: int = 3
) -> None:
    """
    Plot box plots for numerical columns.
    """

    if not columns:
        print("No columns provided.")
        return

    nrows = -(-len(columns) // ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows)
    )

    axes = axes.flatten()

    for i, col in enumerate(columns):

        if col not in df.columns:
            print(f"{col} column not found.")
            continue

        axes[i].boxplot(
            df[col].dropna(),
            patch_artist=True,
            boxprops=dict(
                facecolor=PALETTE["primary"],
                color="white"
            ),
            medianprops=dict(
                color="white",
                linewidth=2
            ),
            whiskerprops=dict(
                color=PALETTE["primary"]
            ),
            capprops=dict(
                color=PALETTE["primary"]
            ),
            flierprops=dict(
                marker="o",
                color=PALETTE["accent"],
                alpha=0.4,
                markersize=3
            )
        )

        axes[i].set_title(
            col,
            fontsize=11,
            fontweight="bold",
            color=PALETTE["primary"]
        )

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Outlier Detection — Box Plots",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["primary"],
        y=1.01
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.15)
    plt.show()


# ── 8. Loss Ratio ─────────────────────────────────────────────────────────────
def compute_loss_ratio(
    df: pd.DataFrame,
    group_col: str = None
):
    """
    Compute Loss Ratio = TotalClaims / TotalPremium.
    """

    required_cols = ["TotalClaims", "TotalPremium"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print(f"Missing columns: {missing}")
        return None

    if group_col:

        if group_col not in df.columns:
            print(f"{group_col} column not found.")
            return None

        return (
            df.groupby(group_col)
            .apply(
                lambda x:
                x["TotalClaims"].sum() /
                x["TotalPremium"].sum()
            )
            .rename("LossRatio")
            .sort_values(ascending=False)
        )

    return (
        df["TotalClaims"].sum() /
        df["TotalPremium"].sum()
    )


# ── 9. Vehicle Make Analysis ──────────────────────────────────────────────────
def plot_vehicle_makes_by_claims(
    df: pd.DataFrame,
    top_n: int = 10
) -> None:
    """
    Plot vehicle makes by average claim amount.
    """

    required_cols = ["make", "TotalClaims"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print(f"Missing columns: {missing}")
        return

    make_claims = (
        df.groupby("make")["TotalClaims"]
        .mean()
        .sort_values(ascending=False)
    )

    highest = make_claims.head(top_n)

    lowest = (
        make_claims.tail(top_n)
        .sort_values(ascending=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(
        highest.index,
        highest.values,
        color=PALETTE["accent"],
        edgecolor="white"
    )

    axes[0].invert_yaxis()

    axes[0].set_title(
        f"Top {top_n} Makes — Highest Avg Claims",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["primary"]
    )

    axes[0].set_xlabel("Average TotalClaims")

    axes[1].barh(
        lowest.index,
        lowest.values.clip(min=0.001),
        color=PALETTE["secondary"],
        edgecolor="white"
    )

    axes[1].invert_yaxis()

    axes[1].set_title(
        f"Top {top_n} Makes — Lowest Avg Claims",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["primary"]
    )

    axes[1].set_xlabel("Average TotalClaims")

    for idx, val in enumerate(lowest.values):

        axes[1].text(
            0.0005,
            idx,
            f"{val:.4f}",
            va="center",
            fontsize=8,
            color=PALETTE["primary"]
        )

    plt.suptitle(
        "Vehicle Makes by Average Claim Amount",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["primary"]
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, wspace=0.4)
    plt.show()
