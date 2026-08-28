# ================================================================
# HIT140 Foundation of Data Science
# FIFA World Cup 2026 - Objective 1
# Student Part: Analytic Task 1 + Analytic Task 2
#
# Task 1: Passing performance - defenders vs midfielders
# Task 2: Shooting accuracy - forwards vs midfielders
#
# Primary approved data source: FBref 2026 FIFA World Cup pages.
# Cross-check source: FIFA World Cup 2026 official statistics.
# ================================================================

from __future__ import annotations

from pathlib import Path
from io import StringIO
import re
import time
import math
import warnings

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
SAMPLE_N = 30
ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95

# Pre-specified inclusion rules.
# These rules are defined before viewing the inferential results.
TASK1_MIN_MINUTES = 90
TASK1_MIN_PASSES = 20

TASK2_MIN_MINUTES = 90
TASK2_MIN_SHOTS = 3

URLS = {
    "standard": [
        "https://fbref.com/en/comps/1/2026/stats/2026-World-Cup-Stats",
        "https://fbref.com/en/comps/1/stats/World-Cup-Stats",
    ],
    "passing": [
        "https://fbref.com/en/comps/1/2026/passing/2026-World-Cup-Stats",
        "https://fbref.com/en/comps/1/passing/World-Cup-Stats",
    ],
    "shooting": [
        "https://fbref.com/en/comps/1/2026/shooting/2026-World-Cup-Stats",
        "https://fbref.com/en/comps/1/shooting/World-Cup-Stats",
    ],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
}


# ================================================================
# 1. DATA ACQUISITION AND WRANGLING HELPERS
# ================================================================

def _norm(text: str) -> str:
    """Normalise a column name to a predictable snake_case label."""
    text = str(text).strip()
    text = re.sub(r"^Unnamed:.*?_level_\d+$", "", text)
    text = text.replace("%", " pct ")
    text = text.replace("#", " num ")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten FBref MultiIndex table headers while retaining the top-level
    category where needed (for example Total_Cmp and Standard_Sh).
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        new_cols = []
        for col in out.columns:
            parts = []
            for part in col:
                p = str(part).strip()
                if p and not p.startswith("Unnamed:"):
                    parts.append(p)
            if not parts:
                new_cols.append("column")
            elif len(parts) == 1:
                new_cols.append(_norm(parts[0]))
            else:
                new_cols.append(_norm("_".join(parts)))
        out.columns = new_cols
    else:
        out.columns = [_norm(c) for c in out.columns]

    # Make duplicate names unique, without silently dropping data.
    seen = {}
    unique = []
    for c in out.columns:
        count = seen.get(c, 0)
        if count == 0:
            unique.append(c)
        else:
            unique.append(f"{c}_{count+1}")
        seen[c] = count + 1
    out.columns = unique
    return out


def _read_all_tables_from_html(html: str) -> list[pd.DataFrame]:
    """
    FBref sometimes stores tables directly in HTML and sometimes inside
    HTML comments. Read both locations.
    """
    tables = []
    try:
        tables.extend(pd.read_html(StringIO(html)))
    except ValueError:
        pass

    soup = BeautifulSoup(html, "lxml")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        if "<table" in comment:
            try:
                tables.extend(pd.read_html(StringIO(comment)))
            except ValueError:
                pass
    return tables


def _is_player_table(df: pd.DataFrame, kind: str) -> bool:
    """Identify the player table, not the squad/opponent table."""
    x = flatten_columns(df)
    cols = set(x.columns)

    has_player = "player" in cols
    has_squad = "squad" in cols
    has_pos = "pos" in cols or "position" in cols

    if not (has_player and has_squad and has_pos):
        return False

    if kind == "standard":
        return any(c in cols for c in ["playing_time_min", "min", "playing_time_90s", "90s"])
    if kind == "passing":
        return any("cmp" in c for c in cols) and any("att" in c for c in cols)
    if kind == "shooting":
        return any(c.endswith("_sh") or c == "sh" for c in cols) and any("sot" in c for c in cols)
    return False


def fetch_fbref_player_table(kind: str, pause_seconds: float = 2.5) -> pd.DataFrame:
    """
    Download a 2026 World Cup FBref page and return its player-level table.
    Tries the year-specific URL first and the current-competition URL second.

    If the website blocks automated access, the analysis can instead use
    manually downloaded CSVs placed in data/:
        data/fbref_standard_2026.csv
        data/fbref_passing_2026.csv
        data/fbref_shooting_2026.csv
    """
    local_path = DATA_DIR / f"fbref_{kind}_2026.csv"
    if local_path.exists():
        print(f"Using local file: {local_path}")
        return pd.read_csv(local_path)

    last_error = None
    for url in URLS[kind]:
        try:
            print(f"Downloading {kind}: {url}")
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            tables = _read_all_tables_from_html(response.text)

            for table in tables:
                if _is_player_table(table, kind):
                    result = flatten_columns(table)
                    result.to_csv(local_path, index=False)
                    print(f"Saved {kind} player table -> {local_path}")
                    time.sleep(pause_seconds)
                    return result

            last_error = RuntimeError(
                f"Page loaded but no player-level {kind} table was identified."
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not obtain the FBref {kind} table.\n"
        f"Last error: {last_error}\n\n"
        f"Fallback: open the approved FBref 2026 World Cup {kind} page, "
        f"use its table export/CSV option, and save it as:\n{local_path}\n"
        f"Then rerun the notebook. No analysis code needs to change."
    )


def find_col(df: pd.DataFrame, exact=(), contains=(), endswith=()) -> str | None:
    """Find one likely column from a normalised dataframe."""
    cols = list(df.columns)
    for name in exact:
        if name in cols:
            return name
    for token in contains:
        for c in cols:
            if token in c:
                return c
    for token in endswith:
        for c in cols:
            if c.endswith(token):
                return c
    return None


def clean_fbref_player_table(df: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated header rows and obvious non-player rows."""
    x = flatten_columns(df)
    player_col = find_col(x, exact=["player"])
    if player_col is None:
        raise ValueError("Player column could not be found.")

    x = x[x[player_col].notna()].copy()
    x[player_col] = x[player_col].astype(str).str.strip()
    x = x[
        ~x[player_col].isin(["Player", "player", ""])
        & ~x[player_col].str.contains("Squad Total", case=False, na=False)
    ]
    return x.reset_index(drop=True)


def numeric(series: pd.Series) -> pd.Series:
    """Convert FBref strings such as '89.3%' to numeric values."""
    return pd.to_numeric(
        series.astype(str)
              .str.replace("%", "", regex=False)
              .str.replace(",", "", regex=False)
              .str.replace("+", "", regex=False)
              .replace({"nan": np.nan, "": np.nan}),
        errors="coerce",
    )


def primary_position(pos) -> str | float:
    """
    Resolve dual-position labels consistently.
    FBref may use labels such as DF,MF. The first listed position is treated
    as the player's primary tournament position.
    """
    if pd.isna(pos):
        return np.nan
    p = str(pos).strip().upper()
    first = re.split(r"[,/\-]", p)[0].strip()

    mapping = {
        "DF": "DF", "DEF": "DF", "DEFENDER": "DF",
        "MF": "MF", "MID": "MF", "MIDFIELDER": "MF",
        "FW": "FW", "FWD": "FW", "FORWARD": "FW",
        "GK": "GK", "GOALKEEPER": "GK",
    }
    return mapping.get(first, first)


def select_core_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Create a small, clearly named table for each source."""
    x = clean_fbref_player_table(df)

    player = find_col(x, exact=["player"])
    squad = find_col(x, exact=["squad"])
    pos = find_col(x, exact=["pos", "position"])

    if kind == "standard":
        minutes = find_col(
            x,
            exact=["playing_time_min", "min"],
            contains=["playing_time_min"],
        )
        nineties = find_col(
            x,
            exact=["playing_time_90s", "90s"],
            contains=["playing_time_90s"],
        )
        cols = [c for c in [player, squad, pos, minutes, nineties] if c]
        y = x[cols].copy()
        rename = {player: "player", squad: "squad", pos: "pos_standard"}
        if minutes:
            rename[minutes] = "minutes"
        if nineties:
            rename[nineties] = "nineties_standard"
        y = y.rename(columns=rename)

        if "minutes" not in y.columns and "nineties_standard" in y.columns:
            y["minutes"] = numeric(y["nineties_standard"]) * 90
        elif "minutes" in y.columns:
            y["minutes"] = numeric(y["minutes"])

        return y.drop_duplicates(["player", "squad"])

    if kind == "passing":
        cmp_col = find_col(
            x,
            exact=["total_cmp", "cmp"],
            contains=["total_cmp"],
        )
        att_col = find_col(
            x,
            exact=["total_att", "att"],
            contains=["total_att"],
        )
        pct_col = find_col(
            x,
            exact=["total_cmp_pct", "cmp_pct"],
            contains=["total_cmp_pct"],
        )
        nineties = find_col(
            x,
            exact=["90s", "playing_time_90s"],
            contains=["playing_time_90s"],
        )
        if cmp_col is None or att_col is None:
            raise ValueError(
                "Could not identify total completed/attempted pass columns.\n"
                f"Available columns:\n{list(x.columns)}"
            )

        cols = [c for c in [player, squad, pos, nineties, cmp_col, att_col, pct_col] if c]
        y = x[cols].copy()
        rename = {
            player: "player", squad: "squad", pos: "pos_passing",
            cmp_col: "passes_completed", att_col: "passes_attempted",
        }
        if nineties:
            rename[nineties] = "nineties_passing"
        if pct_col:
            rename[pct_col] = "published_pass_completion_pct"
        y = y.rename(columns=rename)

        for c in ["passes_completed", "passes_attempted",
                  "nineties_passing", "published_pass_completion_pct"]:
            if c in y.columns:
                y[c] = numeric(y[c])
        return y.drop_duplicates(["player", "squad"])

    if kind == "shooting":
        sh_col = find_col(
            x,
            exact=["standard_sh", "sh"],
            contains=["standard_sh"],
            endswith=["_sh"],
        )
        sot_col = find_col(
            x,
            exact=["standard_sot", "sot"],
            contains=["standard_sot"],
            endswith=["_sot"],
        )
        pct_col = find_col(
            x,
            exact=["standard_sot_pct", "sot_pct"],
            contains=["standard_sot_pct"],
        )
        nineties = find_col(
            x,
            exact=["90s", "playing_time_90s"],
            contains=["playing_time_90s"],
        )
        if sh_col is None or sot_col is None:
            raise ValueError(
                "Could not identify total shots/shots-on-target columns.\n"
                f"Available columns:\n{list(x.columns)}"
            )

        cols = [c for c in [player, squad, pos, nineties, sh_col, sot_col, pct_col] if c]
        y = x[cols].copy()
        rename = {
            player: "player", squad: "squad", pos: "pos_shooting",
            sh_col: "shots", sot_col: "shots_on_target",
        }
        if nineties:
            rename[nineties] = "nineties_shooting"
        if pct_col:
            rename[pct_col] = "published_shot_on_target_pct"
        y = y.rename(columns=rename)

        for c in ["shots", "shots_on_target", "nineties_shooting",
                  "published_shot_on_target_pct"]:
            if c in y.columns:
                y[c] = numeric(y[c])
        return y.drop_duplicates(["player", "squad"])

    raise ValueError(f"Unknown kind: {kind}")


def acquire_and_prepare_sources():
    """Acquire all approved source tables and prepare merge-ready data."""
    raw_standard = fetch_fbref_player_table("standard")
    raw_passing = fetch_fbref_player_table("passing")
    raw_shooting = fetch_fbref_player_table("shooting")

    standard = select_core_columns(raw_standard, "standard")
    passing = select_core_columns(raw_passing, "passing")
    shooting = select_core_columns(raw_shooting, "shooting")

    standard.to_csv(DATA_DIR / "standard_clean.csv", index=False)
    passing.to_csv(DATA_DIR / "passing_clean.csv", index=False)
    shooting.to_csv(DATA_DIR / "shooting_clean.csv", index=False)

    print("\nClean source sizes:")
    print("Standard:", standard.shape)
    print("Passing:", passing.shape)
    print("Shooting:", shooting.shape)

    return standard, passing, shooting


# ================================================================
# 2. STATISTICAL HELPERS - MATCH THE WEEK 2-4 LEARNING MATERIAL
# ================================================================

def descriptive_statistics(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Week 2 descriptive statistics for each comparison group."""
    rows = []
    for group, part in df.groupby(group_col):
        s = part[value_col].dropna().astype(float)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        rows.append({
            "group": group,
            "n": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "standard_deviation": s.std(ddof=1),
            "variance": s.var(ddof=1),
            "minimum": s.min(),
            "Q1": q1,
            "Q3": q3,
            "IQR": q3 - q1,
            "maximum": s.max(),
            "range": s.max() - s.min(),
        })
    return pd.DataFrame(rows)


def mean_ci_t(series: pd.Series, confidence=0.95) -> dict:
    """
    95% confidence interval for a population mean when population sigma
    is unknown: xbar +/- t* (s/sqrt(n)).
    """
    s = pd.Series(series).dropna().astype(float)
    n = len(s)
    if n < 2:
        raise ValueError("At least two observations are required for a CI.")

    mean = s.mean()
    sd = s.std(ddof=1)
    se = sd / math.sqrt(n)
    df = n - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    margin = t_critical * se

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "standard_error": se,
        "df": df,
        "t_critical": t_critical,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
        "confidence": confidence,
    }


def two_sample_t_test_course_method(x: pd.Series, y: pd.Series) -> dict:
    """
    Independent two-sample t-test using the method shown in the HIT140
    Week 4 material:

        t* = (xbar1 - xbar2) / sqrt(s1^2/n1 + s2^2/n2)

    For degrees of freedom, use the course's conservative approach:
        df = smaller of (n1 - 1) and (n2 - 1).

    A two-sided alternative is used:
        H0: mu1 = mu2
        Ha: mu1 != mu2
    """
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)

    n1, n2 = len(x), len(y)
    mean1, mean2 = x.mean(), y.mean()
    s1, s2 = x.std(ddof=1), y.std(ddof=1)

    se = math.sqrt((s1**2 / n1) + (s2**2 / n2))
    t_stat = (mean1 - mean2) / se
    df = min(n1 - 1, n2 - 1)
    p_value = 2 * stats.t.sf(abs(t_stat), df)

    return {
        "n1": n1, "n2": n2,
        "mean1": mean1, "mean2": mean2,
        "sd1": s1, "sd2": s2,
        "mean_difference": mean1 - mean2,
        "standard_error_difference": se,
        "t_statistic": t_stat,
        "df_conservative": df,
        "p_value_two_sided": p_value,
        "alpha": ALPHA,
        "decision": "Reject H0" if p_value <= ALPHA else "Fail to reject H0",
    }


def equal_stratified_random_sample(
    population: pd.DataFrame,
    group_col: str,
    groups: tuple[str, str],
    n_per_group: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Equal-allocation stratified random sampling.
    Within each position group, each eligible player has an equal chance
    of selection. Fixed random seeds make the sample reproducible.
    """
    samples = []
    for i, group in enumerate(groups):
        subset = population[population[group_col] == group].copy()
        if len(subset) < n_per_group:
            raise ValueError(
                f"Only {len(subset)} eligible observations exist for {group}, "
                f"but n={n_per_group} was requested. "
                f"Review the pre-specified inclusion threshold; do not alter "
                f"it merely to obtain a preferred p-value."
            )
        samples.append(
            subset.sample(n=n_per_group, replace=False, random_state=seed + i)
        )
    return pd.concat(samples, ignore_index=True)


def make_ci_table(sample: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    rows = []
    for group, part in sample.groupby(group_col):
        ci = mean_ci_t(part[value_col], CONFIDENCE_LEVEL)
        ci["group"] = group
        rows.append(ci)
    cols = [
        "group", "n", "mean", "sd", "standard_error", "df",
        "t_critical", "ci_lower", "ci_upper", "confidence",
    ]
    return pd.DataFrame(rows)[cols]


def save_boxplot(sample, group_col, value_col, group_order, ylabel, title, filename):
    data = [
        sample.loc[sample[group_col] == g, value_col].dropna().values
        for g in group_order
    ]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=group_order, showmeans=True)
    plt.ylabel(ylabel)
    plt.xlabel("Group")
    plt.title(title)
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.show()
    return path


def save_ci_plot(ci_table, ylabel, title, filename):
    x = np.arange(len(ci_table))
    means = ci_table["mean"].values
    lower = means - ci_table["ci_lower"].values
    upper = ci_table["ci_upper"].values - means

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x, means, yerr=np.vstack([lower, upper]),
        fmt="o", capsize=7
    )
    plt.xticks(x, ci_table["group"])
    plt.ylabel(ylabel)
    plt.xlabel("Group")
    plt.title(title)
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.show()
    return path


def result_sentence(test, group1, group2, metric):
    p = test["p_value_two_sided"]
    m1, m2 = test["mean1"], test["mean2"]
    if p <= ALPHA:
        return (
            f"The two-sample t-test found statistically significant evidence "
            f"of a difference in mean {metric} between {group1} and {group2} "
            f"(t={test['t_statistic']:.2f}, df={test['df_conservative']}, "
            f"p={p:.4f}). The sampled means were {m1:.2f}% and {m2:.2f}%, "
            f"respectively. Because this is an observational tournament dataset, "
            f"the result supports an association with playing position rather "
            f"than a causal effect of position."
        )
    return (
        f"The two-sample t-test did not find sufficient statistical evidence "
        f"of a difference in mean {metric} between {group1} and {group2} "
        f"(t={test['t_statistic']:.2f}, df={test['df_conservative']}, "
        f"p={p:.4f}). The sampled means were {m1:.2f}% and {m2:.2f}%, "
        f"respectively. Failing to reject H0 does not prove the two population "
        f"means are identical; it means this sample does not provide strong "
        f"enough evidence of a difference at alpha=0.05."
    )


# ================================================================
# 3. ANALYTIC TASK 1 - PASSING PERFORMANCE
# ================================================================

def run_task1(standard: pd.DataFrame, passing: pd.DataFrame):
    """
    Analytic Question:
    Among FIFA World Cup 2026 defenders and midfielders who played at least
    90 minutes and attempted at least 20 passes, is there a difference in
    average pass-completion rate?
    """
    print("\n" + "="*70)
    print("ANALYTIC TASK 1 - PASSING PERFORMANCE")
    print("="*70)

    # Week 5 wrangling: inner merge on shared player + squad keys.
    merged = standard.merge(
        passing,
        on=["player", "squad"],
        how="inner",
        validate="one_to_one",
    )

    pos_source = merged["pos_standard"].where(
        merged["pos_standard"].notna(), merged["pos_passing"]
    )
    merged["position"] = pos_source.apply(primary_position)

    # Week 5 missing-data handling: listwise deletion only for variables that
    # are indispensable to this analysis. Mean imputation is not suitable for
    # pass counts because it would manufacture football performance data.
    needed = ["minutes", "passes_completed", "passes_attempted", "position"]
    before = len(merged)
    merged = merged.dropna(subset=needed).copy()
    removed_missing = before - len(merged)

    merged = merged[merged["passes_attempted"] > 0].copy()

    # Feature construction.
    merged["pass_completion_rate"] = (
        merged["passes_completed"] / merged["passes_attempted"] * 100
    )

    # Data-quality rule: impossible percentages are removed.
    merged = merged[
        merged["pass_completion_rate"].between(0, 100, inclusive="both")
    ].copy()

    # Pre-defined sampling frame / population for inference.
    population = merged[
        merged["position"].isin(["DF", "MF"])
        & (merged["minutes"] >= TASK1_MIN_MINUTES)
        & (merged["passes_attempted"] >= TASK1_MIN_PASSES)
    ].copy()

    population = population.sort_values(["position", "squad", "player"])
    population.to_csv(DATA_DIR / "task1_population.csv", index=False)

    print(f"Rows removed for missing required values: {removed_missing}")
    print("\nEligible population counts:")
    print(population["position"].value_counts().sort_index())

    if len(population) == 0:
        raise ValueError("Task 1 population is empty. Inspect source column mapping.")

    # Quality assurance against FBref's displayed percentage, if available.
    if "published_pass_completion_pct" in population.columns:
        check = population.dropna(subset=["published_pass_completion_pct"]).copy()
        if len(check):
            max_diff = (
                check["pass_completion_rate"]
                - check["published_pass_completion_pct"]
            ).abs().max()
            print(
                f"Maximum difference between constructed and published pass "
                f"completion rate: {max_diff:.3f} percentage points "
                f"(small differences are expected from rounding)."
            )

    # Week 3: equal-allocation stratified random sample.
    sample = equal_stratified_random_sample(
        population, "position", ("DF", "MF"), SAMPLE_N, RANDOM_SEED
    )
    sample.to_csv(DATA_DIR / "task1_sample.csv", index=False)

    # Week 2: descriptive statistics.
    desc = descriptive_statistics(sample, "position", "pass_completion_rate")
    desc.to_csv(OUTPUT_DIR / "task1_descriptive_statistics.csv", index=False)
    print("\nDescriptive statistics:")
    print(desc.round(3).to_string(index=False))

    # Week 3/4: 95% t confidence intervals.
    ci = make_ci_table(sample, "position", "pass_completion_rate")
    ci.to_csv(OUTPUT_DIR / "task1_confidence_intervals.csv", index=False)
    print("\n95% confidence intervals:")
    print(ci[["group", "mean", "ci_lower", "ci_upper"]].round(3).to_string(index=False))

    # Week 4: State -> Plan -> Solve -> Conclude
    defenders = sample.loc[sample["position"] == "DF", "pass_completion_rate"]
    midfielders = sample.loc[sample["position"] == "MF", "pass_completion_rate"]

    test = two_sample_t_test_course_method(defenders, midfielders)
    test_df = pd.DataFrame([test])
    test_df.to_csv(OUTPUT_DIR / "task1_ttest.csv", index=False)

    print("\nSTATE")
    print(
        "Practical question: Is mean pass-completion rate different for "
        "eligible defenders and midfielders?"
    )
    print("\nPLAN")
    print("H0: mu_DF = mu_MF")
    print("Ha: mu_DF != mu_MF")
    print("alpha = 0.05; independent two-sample t-test.")
    print(
        "Sampling condition: an equal-size random sample was drawn separately "
        "from the defender and midfielder sampling frames."
    )
    print(
        "Large-sample condition: n=30 per group, consistent with the Week 3 "
        "Central Limit Theorem rule of thumb."
    )

    print("\nSOLVE")
    print(
        f"t={test['t_statistic']:.4f}, "
        f"df={test['df_conservative']}, "
        f"p={test['p_value_two_sided']:.6f}"
    )
    print("\nCONCLUDE")
    conclusion = result_sentence(
        test, "defenders", "midfielders", "pass-completion rate"
    )
    print(conclusion)

    save_boxplot(
        sample, "position", "pass_completion_rate", ["DF", "MF"],
        "Pass completion rate (%)",
        "Task 1: Pass completion rate by primary position",
        "task1_boxplot.png",
    )
    save_ci_plot(
        ci, "Pass completion rate (%)",
        "Task 1: Sample means with 95% confidence intervals",
        "task1_confidence_intervals.png",
    )

    return {
        "population": population,
        "sample": sample,
        "descriptive": desc,
        "ci": ci,
        "test": test,
        "conclusion": conclusion,
    }


# ================================================================
# 4. ANALYTIC TASK 2 - SHOOTING ACCURACY
# ================================================================

def run_task2(standard: pd.DataFrame, shooting: pd.DataFrame):
    """
    Analytic Question:
    Among FIFA World Cup 2026 forwards and midfielders who played at least
    90 minutes and attempted at least 3 shots, is there a difference in
    average shot-on-target rate?
    """
    print("\n" + "="*70)
    print("ANALYTIC TASK 2 - SHOOTING ACCURACY")
    print("="*70)

    # Week 5 wrangling: merge separate player and shooting tables.
    merged = standard.merge(
        shooting,
        on=["player", "squad"],
        how="inner",
        validate="one_to_one",
    )

    pos_source = merged["pos_standard"].where(
        merged["pos_standard"].notna(), merged["pos_shooting"]
    )
    merged["position"] = pos_source.apply(primary_position)

    needed = ["minutes", "shots", "shots_on_target", "position"]
    before = len(merged)
    merged = merged.dropna(subset=needed).copy()
    removed_missing = before - len(merged)

    # Feature construction.
    merged = merged[merged["shots"] > 0].copy()
    merged["shot_on_target_rate"] = (
        merged["shots_on_target"] / merged["shots"] * 100
    )
    merged = merged[
        merged["shot_on_target_rate"].between(0, 100, inclusive="both")
    ].copy()

    # Pre-defined eligible population.
    population = merged[
        merged["position"].isin(["FW", "MF"])
        & (merged["minutes"] >= TASK2_MIN_MINUTES)
        & (merged["shots"] >= TASK2_MIN_SHOTS)
    ].copy()

    population = population.sort_values(["position", "squad", "player"])
    population.to_csv(DATA_DIR / "task2_population.csv", index=False)

    print(f"Rows removed for missing required values: {removed_missing}")
    print("\nEligible population counts:")
    print(population["position"].value_counts().sort_index())

    if len(population) == 0:
        raise ValueError("Task 2 population is empty. Inspect source column mapping.")

    if "published_shot_on_target_pct" in population.columns:
        check = population.dropna(subset=["published_shot_on_target_pct"]).copy()
        if len(check):
            max_diff = (
                check["shot_on_target_rate"]
                - check["published_shot_on_target_pct"]
            ).abs().max()
            print(
                f"Maximum difference between constructed and published "
                f"shot-on-target rate: {max_diff:.3f} percentage points "
                f"(small differences are expected from rounding)."
            )

    sample = equal_stratified_random_sample(
        population, "position", ("FW", "MF"), SAMPLE_N, RANDOM_SEED + 100
    )
    sample.to_csv(DATA_DIR / "task2_sample.csv", index=False)

    desc = descriptive_statistics(sample, "position", "shot_on_target_rate")
    desc.to_csv(OUTPUT_DIR / "task2_descriptive_statistics.csv", index=False)
    print("\nDescriptive statistics:")
    print(desc.round(3).to_string(index=False))

    ci = make_ci_table(sample, "position", "shot_on_target_rate")
    ci.to_csv(OUTPUT_DIR / "task2_confidence_intervals.csv", index=False)
    print("\n95% confidence intervals:")
    print(ci[["group", "mean", "ci_lower", "ci_upper"]].round(3).to_string(index=False))

    forwards = sample.loc[sample["position"] == "FW", "shot_on_target_rate"]
    midfielders = sample.loc[sample["position"] == "MF", "shot_on_target_rate"]

    test = two_sample_t_test_course_method(forwards, midfielders)
    pd.DataFrame([test]).to_csv(OUTPUT_DIR / "task2_ttest.csv", index=False)

    print("\nSTATE")
    print(
        "Practical question: Is mean shot-on-target rate different for "
        "eligible forwards and midfielders?"
    )
    print("\nPLAN")
    print("H0: mu_FW = mu_MF")
    print("Ha: mu_FW != mu_MF")
    print("alpha = 0.05; independent two-sample t-test.")
    print(
        "Sampling condition: an equal-size random sample was drawn separately "
        "from the forward and midfielder sampling frames."
    )
    print(
        "Large-sample condition: n=30 per group, consistent with the Week 3 "
        "Central Limit Theorem rule of thumb."
    )

    print("\nSOLVE")
    print(
        f"t={test['t_statistic']:.4f}, "
        f"df={test['df_conservative']}, "
        f"p={test['p_value_two_sided']:.6f}"
    )
    print("\nCONCLUDE")
    conclusion = result_sentence(
        test, "forwards", "midfielders", "shot-on-target rate"
    )
    print(conclusion)

    save_boxplot(
        sample, "position", "shot_on_target_rate", ["FW", "MF"],
        "Shot-on-target rate (%)",
        "Task 2: Shot-on-target rate by primary position",
        "task2_boxplot.png",
    )
    save_ci_plot(
        ci, "Shot-on-target rate (%)",
        "Task 2: Sample means with 95% confidence intervals",
        "task2_confidence_intervals.png",
    )

    return {
        "population": population,
        "sample": sample,
        "descriptive": desc,
        "ci": ci,
        "test": test,
        "conclusion": conclusion,
    }


# ================================================================
# 5. SAVE SLIDE-READY SUMMARY
# ================================================================

def save_slide_summary(task1, task2):
    t1 = task1["test"]
    t2 = task2["test"]
    ci1 = task1["ci"].set_index("group")
    ci2 = task2["ci"].set_index("group")

    text = f"""
HIT140 - FIFA WORLD CUP 2026 - SLIDE-READY RESULTS
=================================================

ANALYTIC TASK 1 - PASSING PERFORMANCE
Question:
Among defenders and midfielders who played >= {TASK1_MIN_MINUTES} minutes
and attempted >= {TASK1_MIN_PASSES} passes, is mean pass-completion rate different?

Population / sampling frame:
All eligible 2026 World Cup DF and MF records in the cleaned FBref dataset.

Sampling:
Equal-allocation stratified random sample, n={SAMPLE_N} DF and n={SAMPLE_N} MF,
random seed={RANDOM_SEED}.

Feature:
pass_completion_rate = passes_completed / passes_attempted * 100

95% confidence intervals:
DF mean = {ci1.loc['DF','mean']:.2f}%
95% CI = [{ci1.loc['DF','ci_lower']:.2f}%, {ci1.loc['DF','ci_upper']:.2f}%]

MF mean = {ci1.loc['MF','mean']:.2f}%
95% CI = [{ci1.loc['MF','ci_lower']:.2f}%, {ci1.loc['MF','ci_upper']:.2f}%]

Hypotheses:
H0: mu_DF = mu_MF
Ha: mu_DF != mu_MF

Two-sample t-test:
t = {t1['t_statistic']:.3f}
df = {t1['df_conservative']}
p = {t1['p_value_two_sided']:.4f}
Decision = {t1['decision']}

Conclusion:
{task1['conclusion']}


ANALYTIC TASK 2 - SHOOTING ACCURACY
Question:
Among forwards and midfielders who played >= {TASK2_MIN_MINUTES} minutes
and attempted >= {TASK2_MIN_SHOTS} shots, is mean shot-on-target rate different?

Population / sampling frame:
All eligible 2026 World Cup FW and MF records in the cleaned FBref dataset.

Sampling:
Equal-allocation stratified random sample, n={SAMPLE_N} FW and n={SAMPLE_N} MF,
random seed={RANDOM_SEED + 100}.

Feature:
shot_on_target_rate = shots_on_target / shots * 100

95% confidence intervals:
FW mean = {ci2.loc['FW','mean']:.2f}%
95% CI = [{ci2.loc['FW','ci_lower']:.2f}%, {ci2.loc['FW','ci_upper']:.2f}%]

MF mean = {ci2.loc['MF','mean']:.2f}%
95% CI = [{ci2.loc['MF','ci_lower']:.2f}%, {ci2.loc['MF','ci_upper']:.2f}%]

Hypotheses:
H0: mu_FW = mu_MF
Ha: mu_FW != mu_MF

Two-sample t-test:
t = {t2['t_statistic']:.3f}
df = {t2['df_conservative']}
p = {t2['p_value_two_sided']:.4f}
Decision = {t2['decision']}

Conclusion:
{task2['conclusion']}


IMPORTANT LIMITATIONS TO STATE IN PRESENTATION
----------------------------------------------
1. The analyses use tournament observational data, so differences are
   associations and do not establish that player position causes performance.
2. Eligibility thresholds were pre-specified to reduce unstable percentages
   from very small denominators.
3. Dual-position players are classified by the first-listed (primary) FBref
   position, applied consistently before analysis.
4. Missing values required for a task are removed rather than mean-imputed,
   because imputation would manufacture football performance counts.
5. The random samples are reproducible through fixed random seeds.
6. Percentage variables are bounded; n=30 per group helps support inference
   about sample means through the Central Limit Theorem, but interpretation
   should remain cautious.
"""
    path = OUTPUT_DIR / "slide_ready_results.txt"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


# ================================================================
# 6. RUN EVERYTHING
# ================================================================

def main():
    print("HIT140 FIFA World Cup 2026 - Objective 1")
    print("Running the student's two analytic tasks...\n")

    standard, passing, shooting = acquire_and_prepare_sources()

    task1 = run_task1(standard, passing)
    task2 = run_task2(standard, shooting)

    summary_path = save_slide_summary(task1, task2)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("Datasets saved in:", DATA_DIR.resolve())
    print("Statistics/charts saved in:", OUTPUT_DIR.resolve())
    print("Slide-ready results:", summary_path.resolve())
    print(
        "\nBefore submission, read every section and make sure you can explain "
        "the cleaning rules, formulas, sample design, confidence intervals, "
        "t-statistics, p-values, and conclusions."
    )


if __name__ == "__main__":
    main()
