
"""
HIT140 Foundation of Data Science
FIFA World Cup 2026 - Objective 1
Student Part: Analytic Task 1 + Analytic Task 2

DATA SOURCE:
Official FIFA World Cup 2026 public API endpoints.

Task 1:
Among eligible defenders and midfielders, is there a difference in
average pass-completion rate?

Task 2:
Among eligible forwards and midfielders, is there a difference in
average shot-on-target rate?

This version does NOT use FBref, so it avoids FBref's 403 Forbidden error.

Install:
    py -m pip install pandas numpy scipy matplotlib requests

Run:
    py HIT140_Tasks_1_2_FIFA_Official.py
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
import matplotlib.pyplot as plt


# ================================================================
# CONFIGURATION
# ================================================================

SEASON_ID = "285023"        # FIFA World Cup 2026
COMPETITION_ID = "17"       # Men's FIFA World Cup
LANGUAGE = "en"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
SAMPLE_N = 30
ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95

# Pre-specified inclusion rules
TASK1_MIN_MINUTES = 90
TASK1_MIN_PASSES = 20

TASK2_MIN_MINUTES = 90
TASK2_MIN_SHOTS = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-AU,en;q=0.9",
}


# ================================================================
# DATA ACQUISITION - OFFICIAL FIFA
# ================================================================

def get_json(url, params=None, retries=4):
    """Download JSON with retries and a browser-like user agent."""
    last_error = None

    for attempt in range(retries):
        try:
            r = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=40,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_error = exc
            print(
                f"Request attempt {attempt + 1}/{retries} failed: {exc}"
            )
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"FIFA request failed after {retries} attempts.\n"
        f"URL: {url}\n"
        f"Last error: {last_error}"
    )


def description(value):
    """Extract a readable name from FIFA localized JSON fields."""
    if value is None:
        return np.nan

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        preferred = [
            "Description",
            "DescriptionLocalized",
            "ShortName",
            "Name",
        ]
        for key in preferred:
            if key in value:
                result = description(value[key])
                if pd.notna(result):
                    return result

        for v in value.values():
            result = description(v)
            if pd.notna(result):
                return result

    if isinstance(value, list):
        for item in value:
            result = description(item)
            if pd.notna(result):
                return result

    return np.nan


def normalise(text):
    """Convert FIFA statistic names to snake_case."""
    text = str(text).strip().lower()
    text = text.replace("%", " percent ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def fetch_teams():
    url = f"https://api.fifa.com/api/v3/competitions/teams/{SEASON_ID}"
    data = get_json(url, params={"language": LANGUAGE})

    results = data.get("Results", data if isinstance(data, list) else [])
    rows = []

    for item in results:
        team_id = (
            item.get("IdTeam")
            or item.get("idTeam")
            or item.get("Id")
        )

        team_name = (
            description(item.get("Name"))
            or description(item.get("TeamName"))
            or item.get("ShortClubName")
            or item.get("Abbreviation")
        )

        team_code = (
            item.get("Abbreviation")
            or item.get("ShortName")
        )

        if team_id is not None:
            rows.append({
                "team_id": str(team_id),
                "team": team_name,
                "team_code": team_code,
            })

    teams = pd.DataFrame(rows).drop_duplicates("team_id")

    if teams.empty:
        raise RuntimeError(
            "FIFA team endpoint returned no teams."
        )

    teams.to_csv(DATA_DIR / "fifa_2026_teams.csv", index=False)
    return teams


def fetch_squad(team_id, team_name, team_code):
    url = f"https://api.fifa.com/api/v3/teams/{team_id}/squad"

    data = get_json(
        url,
        params={
            "idCompetition": COMPETITION_ID,
            "idSeason": SEASON_ID,
            "language": LANGUAGE,
        },
    )

    players = data.get("Players", [])
    rows = []

    for p in players:
        player_id = (
            p.get("IdPlayer")
            or p.get("idPlayer")
            or p.get("Id")
        )

        player_name = (
            description(p.get("PlayerName"))
            or description(p.get("ShortName"))
            or description(p.get("Name"))
        )

        position = (
            description(p.get("PositionLocalized"))
            or description(p.get("RealPositionLocalized"))
            or description(p.get("Position"))
        )

        rows.append({
            "player_id": str(player_id) if player_id is not None else None,
            "player_name": player_name,
            "team_id": str(team_id),
            "team": team_name,
            "team_code": team_code,
            "position_raw": position,
        })

    return pd.DataFrame(rows)


def fetch_all_players(teams):
    frames = []

    for number, (_, row) in enumerate(teams.iterrows(), start=1):
        print(
            f"Downloading squad {number}/{len(teams)}: {row['team']}"
        )

        frame = fetch_squad(
            str(row["team_id"]),
            row["team"],
            row.get("team_code"),
        )

        if not frame.empty:
            frames.append(frame)

        time.sleep(0.15)

    if not frames:
        raise RuntimeError("No squad records were downloaded.")

    players = pd.concat(frames, ignore_index=True)
    players = (
        players
        .dropna(subset=["player_id"])
        .drop_duplicates("player_id")
    )

    players.to_csv(
        DATA_DIR / "fifa_2026_players.csv",
        index=False,
    )
    return players


def fetch_player_stats_long():
    """
    Download FIFA's tournament player-statistics payload.

    The endpoint returns a mapping of player IDs to lists of statistic
    records. The parser supports both list and dictionary record formats.
    """
    url = (
        f"https://fdh-api.fifa.com/v1/stats/"
        f"season/{SEASON_ID}/players.json"
    )

    data = get_json(url)
    rows = []

    if not isinstance(data, dict):
        raise RuntimeError(
            "Unexpected FIFA player-statistics response."
        )

    for player_id, records in data.items():
        if not isinstance(records, list):
            continue

        for record in records:
            stat_name = None
            stat_value = None

            if isinstance(record, list) and len(record) >= 2:
                stat_name = record[0]
                stat_value = record[1]

            elif isinstance(record, dict):
                stat_name = (
                    record.get("stat")
                    or record.get("Stat")
                    or record.get("name")
                    or record.get("Name")
                    or record.get("Statistic")
                )

                if "value" in record:
                    stat_value = record.get("value")
                elif "Value" in record:
                    stat_value = record.get("Value")
                else:
                    stat_value = record.get("StatisticValue")

            if stat_name is not None:
                rows.append({
                    "player_id": str(player_id),
                    "stat": str(stat_name),
                    "value": stat_value,
                })

    stats_long = pd.DataFrame(rows)

    if stats_long.empty:
        raise RuntimeError(
            "FIFA returned player data but no statistics were parsed."
        )

    stats_long["value"] = pd.to_numeric(
        stats_long["value"],
        errors="coerce",
    )

    stats_long.to_csv(
        DATA_DIR / "fifa_2026_player_stats_long.csv",
        index=False,
    )

    return stats_long


def stats_to_wide(stats_long):
    """
    Convert FIFA long statistics into one row per player.

    The maximum is used when the same tournament aggregate statistic
    appears more than once for a player.
    """
    clean = stats_long.dropna(
        subset=["value"]
    ).copy()

    clean["stat_clean"] = clean["stat"].apply(normalise)

    wide = (
        clean.groupby(
            ["player_id", "stat_clean"],
            as_index=False,
        )["value"]
        .max()
        .pivot(
            index="player_id",
            columns="stat_clean",
            values="value",
        )
        .reset_index()
    )

    wide.columns.name = None

    wide.to_csv(
        DATA_DIR / "fifa_2026_player_stats_wide.csv",
        index=False,
    )

    return wide


# ================================================================
# COLUMN / POSITION HELPERS
# ================================================================

def primary_position(value):
    if pd.isna(value):
        return np.nan

    p = str(value).strip().lower()

    if (
        "goal" in p
        or p in {"gk"}
    ):
        return "GK"

    if (
        "def" in p
        or p in {"df"}
    ):
        return "DF"

    if (
        "mid" in p
        or p in {"mf"}
    ):
        return "MF"

    if (
        "forw" in p
        or "attack" in p
        or p in {"fw", "fwd"}
    ):
        return "FW"

    return str(value).strip().upper()


def find_stat_column(df, candidate_names):
    """
    First try exact normalized names.
    Then allow contained matches.
    """
    columns = list(df.columns)

    for candidate in candidate_names:
        candidate = normalise(candidate)
        if candidate in columns:
            return candidate

    for candidate in candidate_names:
        candidate = normalise(candidate)

        for column in columns:
            if candidate in column:
                return column

    return None


def identify_required_stats(df):
    """
    Identify FIFA's relevant player-stat columns.
    If names have changed, the script prints every available statistic
    so the mapping can be adjusted transparently.
    """
    # FIFA's API commonly uses compact camelCase names such as
    # passesCompleted, attemptAtGoal, attemptAtGoalOnTarget and timePlayed.
    # Our normaliser lowercases these without inserting underscores, so both
    # compact and human-readable variants are listed explicitly.

    passes = find_stat_column(
        df,
        [
            "passes",
            "passesattempted",
            "passes attempted",
            "totalpasses",
            "total passes",
            "passattempts",
            "pass attempts",
        ],
    )

    passes_completed = find_stat_column(
        df,
        [
            "passescompleted",
            "passes completed",
            "completedpasses",
            "completed passes",
            "successfulpasses",
            "successful passes",
        ],
    )

    shots = find_stat_column(
        df,
        [
            "attemptatgoal",
            "attempts at goal",
            "attemptsatgoal",
            "attempts",
            "shots",
            "totalshots",
            "total attempts",
        ],
    )

    shots_on_target = find_stat_column(
        df,
        [
            "attemptatgoalontarget",
            "attemptsatgoalontarget",
            "attempts on target",
            "shotsontarget",
            "shots on target",
            "ontarget",
            "on target",
        ],
    )

    minutes = find_stat_column(
        df,
        [
            "timeplayed",
            "time played",
            "minutesplayed",
            "minutes played",
            "minutes",
        ],
    )

    mapping = {
        "passes_attempted": passes,
        "passes_completed": passes_completed,
        "shots": shots,
        "shots_on_target": shots_on_target,
        "minutes": minutes,
    }

    print("\nDetected FIFA statistic mapping:")
    for key, value in mapping.items():
        print(f"  {key}: {value}")

    missing = [
        key
        for key, value in mapping.items()
        if value is None
    ]

    if missing:
        print("\nAvailable FIFA statistic columns:")
        for col in sorted(df.columns):
            print(" ", col)

        raise RuntimeError(
            "\nCould not automatically identify these required statistics: "
            + ", ".join(missing)
            + "\nThe FIFA data was downloaded successfully. "
              "Only the statistic-name mapping needs adjustment."
        )

    return mapping


def acquire_fifa_dataset():
    print("\n1/4 Downloading official FIFA teams...")
    teams = fetch_teams()

    print("\n2/4 Downloading official FIFA squads and positions...")
    players = fetch_all_players(teams)

    print("\n3/4 Downloading official FIFA player statistics...")
    stats_long = fetch_player_stats_long()

    print("\n4/4 Reshaping FIFA statistics...")
    stats_wide = stats_to_wide(stats_long)

    mapping = identify_required_stats(stats_wide)

    merged = players.merge(
        stats_wide,
        on="player_id",
        how="left",
        validate="one_to_one",
    )

    merged["position"] = (
        merged["position_raw"]
        .apply(primary_position)
    )

    merged = merged.rename(
        columns={
            mapping["passes_attempted"]: "passes_attempted",
            mapping["passes_completed"]: "passes_completed",
            mapping["shots"]: "shots",
            mapping["shots_on_target"]: "shots_on_target",
            mapping["minutes"]: "minutes",
        }
    )

    for column in [
        "passes_attempted",
        "passes_completed",
        "shots",
        "shots_on_target",
        "minutes",
    ]:
        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        )

    merged.to_csv(
        DATA_DIR / "fifa_2026_analysis_master.csv",
        index=False,
    )

    return merged


# ================================================================
# STATISTICAL FUNCTIONS
# ================================================================

def descriptive_statistics(df, group_col, value_col):
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


def mean_ci_t(series, confidence=0.95):
    s = pd.Series(series).dropna().astype(float)

    n = len(s)
    mean = s.mean()
    sd = s.std(ddof=1)
    se = sd / math.sqrt(n)

    df = n - 1
    t_critical = stats.t.ppf(
        (1 + confidence) / 2,
        df,
    )

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
    }


def make_ci_table(sample, group_col, value_col):
    rows = []

    for group, part in sample.groupby(group_col):
        result = mean_ci_t(
            part[value_col],
            CONFIDENCE_LEVEL,
        )
        result["group"] = group
        rows.append(result)

    return pd.DataFrame(rows)[
        [
            "group",
            "n",
            "mean",
            "sd",
            "standard_error",
            "df",
            "t_critical",
            "ci_lower",
            "ci_upper",
        ]
    ]


def stratified_sample(
    population,
    group_col,
    group1,
    group2,
    n,
    seed,
):
    outputs = []

    for i, group in enumerate([group1, group2]):
        subset = population[
            population[group_col] == group
        ].copy()

        if len(subset) < n:
            raise RuntimeError(
                f"Task requires n={n} for {group}, "
                f"but only {len(subset)} eligible players exist.\n"
                f"Do NOT change the threshold just to change the p-value. "
                f"Review the sample-size plan with your lecturer or use "
                f"the full eligible subgroup if appropriate."
            )

        outputs.append(
            subset.sample(
                n=n,
                replace=False,
                random_state=seed + i,
            )
        )

    return pd.concat(
        outputs,
        ignore_index=True,
    )


def two_sample_t_test_course(x, y):
    """
    Independent two-sample t-test using the form taught in the unit.

    t = (xbar1 - xbar2) /
        sqrt(s1^2/n1 + s2^2/n2)

    Conservative df:
        min(n1 - 1, n2 - 1)
    """
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)

    n1, n2 = len(x), len(y)
    mean1, mean2 = x.mean(), y.mean()
    sd1, sd2 = x.std(ddof=1), y.std(ddof=1)

    se = math.sqrt(
        (sd1**2 / n1)
        + (sd2**2 / n2)
    )

    t_stat = (mean1 - mean2) / se

    df = min(
        n1 - 1,
        n2 - 1,
    )

    p_value = (
        2
        * stats.t.sf(
            abs(t_stat),
            df,
        )
    )

    return {
        "n1": n1,
        "n2": n2,
        "mean1": mean1,
        "mean2": mean2,
        "sd1": sd1,
        "sd2": sd2,
        "mean_difference": mean1 - mean2,
        "t_statistic": t_stat,
        "df": df,
        "p_value": p_value,
        "decision": (
            "Reject H0"
            if p_value <= ALPHA
            else "Fail to reject H0"
        ),
    }


def plot_boxplot(
    sample,
    group_col,
    value_col,
    groups,
    ylabel,
    title,
    filename,
):
    values = [
        sample.loc[
            sample[group_col] == group,
            value_col,
        ].dropna().values
        for group in groups
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        values,
        labels=groups,
        showmeans=True,
    )
    plt.xlabel("Position")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


def plot_ci(
    ci_table,
    ylabel,
    title,
    filename,
):
    x = np.arange(
        len(ci_table)
    )

    means = (
        ci_table["mean"]
        .to_numpy()
    )

    lower = (
        means
        - ci_table["ci_lower"].to_numpy()
    )

    upper = (
        ci_table["ci_upper"].to_numpy()
        - means
    )

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x,
        means,
        yerr=np.vstack(
            [lower, upper]
        ),
        fmt="o",
        capsize=7,
    )

    plt.xticks(
        x,
        ci_table["group"],
    )
    plt.xlabel("Position")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


# ================================================================
# TASK 1 - PASSING
# ================================================================

def run_task1(master):
    print("\n" + "=" * 68)
    print("TASK 1 - PASSING PERFORMANCE")
    print("=" * 68)

    data = master.dropna(
        subset=[
            "position",
            "minutes",
            "passes_attempted",
            "passes_completed",
        ]
    ).copy()

    data = data[
        data["passes_attempted"] > 0
    ].copy()

    data["pass_completion_rate"] = (
        data["passes_completed"]
        / data["passes_attempted"]
        * 100
    )

    data = data[
        data["pass_completion_rate"]
        .between(0, 100)
    ].copy()

    population = data[
        data["position"].isin(
            ["DF", "MF"]
        )
        & (
            data["minutes"]
            >= TASK1_MIN_MINUTES
        )
        & (
            data["passes_attempted"]
            >= TASK1_MIN_PASSES
        )
    ].copy()

    population.to_csv(
        DATA_DIR / "task1_population.csv",
        index=False,
    )

    print("\nEligible Task 1 population:")
    print(
        population["position"]
        .value_counts()
        .sort_index()
    )

    sample = stratified_sample(
        population,
        "position",
        "DF",
        "MF",
        SAMPLE_N,
        RANDOM_SEED,
    )

    sample.to_csv(
        DATA_DIR / "task1_sample.csv",
        index=False,
    )

    desc = descriptive_statistics(
        sample,
        "position",
        "pass_completion_rate",
    )

    desc.to_csv(
        OUTPUT_DIR
        / "task1_descriptive_statistics.csv",
        index=False,
    )

    ci = make_ci_table(
        sample,
        "position",
        "pass_completion_rate",
    )

    ci.to_csv(
        OUTPUT_DIR
        / "task1_confidence_intervals.csv",
        index=False,
    )

    df_values = sample.loc[
        sample["position"] == "DF",
        "pass_completion_rate",
    ]

    mf_values = sample.loc[
        sample["position"] == "MF",
        "pass_completion_rate",
    ]

    test = two_sample_t_test_course(
        df_values,
        mf_values,
    )

    pd.DataFrame([test]).to_csv(
        OUTPUT_DIR / "task1_ttest.csv",
        index=False,
    )

    plot_boxplot(
        sample,
        "position",
        "pass_completion_rate",
        ["DF", "MF"],
        "Pass completion rate (%)",
        "Task 1: Pass completion rate by position",
        "task1_boxplot.png",
    )

    plot_ci(
        ci,
        "Pass completion rate (%)",
        "Task 1: Mean pass completion with 95% CI",
        "task1_confidence_intervals.png",
    )

    print("\nDescriptive statistics:")
    print(
        desc.round(3)
        .to_string(index=False)
    )

    print("\n95% confidence intervals:")
    print(
        ci[
            [
                "group",
                "mean",
                "ci_lower",
                "ci_upper",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nSTATE")
    print(
        "Is average pass-completion rate different "
        "between eligible defenders and midfielders?"
    )

    print("\nPLAN")
    print("H0: mu_DF = mu_MF")
    print("Ha: mu_DF != mu_MF")
    print("alpha = 0.05")

    print("\nSOLVE")
    print(
        f"t = {test['t_statistic']:.4f}"
    )
    print(
        f"df = {test['df']}"
    )
    print(
        f"p = {test['p_value']:.6f}"
    )

    print("\nCONCLUDE")
    print(test["decision"])

    return {
        "population": population,
        "sample": sample,
        "desc": desc,
        "ci": ci,
        "test": test,
    }


# ================================================================
# TASK 2 - SHOOTING
# ================================================================

def run_task2(master):
    print("\n" + "=" * 68)
    print("TASK 2 - SHOOTING ACCURACY")
    print("=" * 68)

    data = master.dropna(
        subset=[
            "position",
            "minutes",
            "shots",
            "shots_on_target",
        ]
    ).copy()

    data = data[
        data["shots"] > 0
    ].copy()

    data["shot_on_target_rate"] = (
        data["shots_on_target"]
        / data["shots"]
        * 100
    )

    data = data[
        data["shot_on_target_rate"]
        .between(0, 100)
    ].copy()

    population = data[
        data["position"].isin(
            ["FW", "MF"]
        )
        & (
            data["minutes"]
            >= TASK2_MIN_MINUTES
        )
        & (
            data["shots"]
            >= TASK2_MIN_SHOTS
        )
    ].copy()

    population.to_csv(
        DATA_DIR / "task2_population.csv",
        index=False,
    )

    print("\nEligible Task 2 population:")
    print(
        population["position"]
        .value_counts()
        .sort_index()
    )

    sample = stratified_sample(
        population,
        "position",
        "FW",
        "MF",
        SAMPLE_N,
        RANDOM_SEED + 100,
    )

    sample.to_csv(
        DATA_DIR / "task2_sample.csv",
        index=False,
    )

    desc = descriptive_statistics(
        sample,
        "position",
        "shot_on_target_rate",
    )

    desc.to_csv(
        OUTPUT_DIR
        / "task2_descriptive_statistics.csv",
        index=False,
    )

    ci = make_ci_table(
        sample,
        "position",
        "shot_on_target_rate",
    )

    ci.to_csv(
        OUTPUT_DIR
        / "task2_confidence_intervals.csv",
        index=False,
    )

    fw_values = sample.loc[
        sample["position"] == "FW",
        "shot_on_target_rate",
    ]

    mf_values = sample.loc[
        sample["position"] == "MF",
        "shot_on_target_rate",
    ]

    test = two_sample_t_test_course(
        fw_values,
        mf_values,
    )

    pd.DataFrame([test]).to_csv(
        OUTPUT_DIR / "task2_ttest.csv",
        index=False,
    )

    plot_boxplot(
        sample,
        "position",
        "shot_on_target_rate",
        ["FW", "MF"],
        "Shot-on-target rate (%)",
        "Task 2: Shot-on-target rate by position",
        "task2_boxplot.png",
    )

    plot_ci(
        ci,
        "Shot-on-target rate (%)",
        "Task 2: Mean shot-on-target rate with 95% CI",
        "task2_confidence_intervals.png",
    )

    print("\nDescriptive statistics:")
    print(
        desc.round(3)
        .to_string(index=False)
    )

    print("\n95% confidence intervals:")
    print(
        ci[
            [
                "group",
                "mean",
                "ci_lower",
                "ci_upper",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nSTATE")
    print(
        "Is average shot-on-target rate different "
        "between eligible forwards and midfielders?"
    )

    print("\nPLAN")
    print("H0: mu_FW = mu_MF")
    print("Ha: mu_FW != mu_MF")
    print("alpha = 0.05")

    print("\nSOLVE")
    print(
        f"t = {test['t_statistic']:.4f}"
    )
    print(
        f"df = {test['df']}"
    )
    print(
        f"p = {test['p_value']:.6f}"
    )

    print("\nCONCLUDE")
    print(test["decision"])

    return {
        "population": population,
        "sample": sample,
        "desc": desc,
        "ci": ci,
        "test": test,
    }


# ================================================================
# SLIDE-READY SUMMARY
# ================================================================

def make_summary(task1, task2):
    t1_ci = (
        task1["ci"]
        .set_index("group")
    )

    t2_ci = (
        task2["ci"]
        .set_index("group")
    )

    t1 = task1["test"]
    t2 = task2["test"]

    if t1["p_value"] <= ALPHA:
        t1_conclusion = (
            "At the 5% significance level, the sample provides "
            "statistically significant evidence that mean pass-completion "
            "rate differs between eligible defenders and midfielders."
        )
    else:
        t1_conclusion = (
            "At the 5% significance level, the sample does not provide "
            "sufficient statistical evidence that mean pass-completion "
            "rate differs between eligible defenders and midfielders."
        )

    if t2["p_value"] <= ALPHA:
        t2_conclusion = (
            "At the 5% significance level, the sample provides "
            "statistically significant evidence that mean shot-on-target "
            "rate differs between eligible forwards and midfielders."
        )
    else:
        t2_conclusion = (
            "At the 5% significance level, the sample does not provide "
            "sufficient statistical evidence that mean shot-on-target "
            "rate differs between eligible forwards and midfielders."
        )

    summary = f"""
HIT140 FIFA WORLD CUP 2026
MY TWO ANALYTIC TASKS - FINAL NUMERICAL RESULTS
============================================================

DATA SOURCE
Official FIFA World Cup 2026 public statistics APIs.

TASK 1 - PASSING PERFORMANCE
------------------------------------------------------------
Question:
Among defenders and midfielders who played at least
{TASK1_MIN_MINUTES} minutes and attempted at least
{TASK1_MIN_PASSES} passes, is there a difference in
average pass-completion rate?

Feature:
pass_completion_rate =
passes_completed / passes_attempted * 100

Sampling:
n = {SAMPLE_N} defenders
n = {SAMPLE_N} midfielders
Fixed random seed = {RANDOM_SEED}

Defenders:
Mean = {t1_ci.loc['DF', 'mean']:.2f}%
95% CI = [{t1_ci.loc['DF', 'ci_lower']:.2f}%,
          {t1_ci.loc['DF', 'ci_upper']:.2f}%]

Midfielders:
Mean = {t1_ci.loc['MF', 'mean']:.2f}%
95% CI = [{t1_ci.loc['MF', 'ci_lower']:.2f}%,
          {t1_ci.loc['MF', 'ci_upper']:.2f}%]

H0: mu_DF = mu_MF
Ha: mu_DF != mu_MF

t = {t1['t_statistic']:.3f}
df = {t1['df']}
p = {t1['p_value']:.4f}

Decision:
{t1['decision']}

Interpretation:
{t1_conclusion}


TASK 2 - SHOOTING ACCURACY
------------------------------------------------------------
Question:
Among forwards and midfielders who played at least
{TASK2_MIN_MINUTES} minutes and attempted at least
{TASK2_MIN_SHOTS} shots, is there a difference in
average shot-on-target rate?

Feature:
shot_on_target_rate =
shots_on_target / shots * 100

Sampling:
n = {SAMPLE_N} forwards
n = {SAMPLE_N} midfielders
Fixed random seed = {RANDOM_SEED + 100}

Forwards:
Mean = {t2_ci.loc['FW', 'mean']:.2f}%
95% CI = [{t2_ci.loc['FW', 'ci_lower']:.2f}%,
          {t2_ci.loc['FW', 'ci_upper']:.2f}%]

Midfielders:
Mean = {t2_ci.loc['MF', 'mean']:.2f}%
95% CI = [{t2_ci.loc['MF', 'ci_lower']:.2f}%,
          {t2_ci.loc['MF', 'ci_upper']:.2f}%]

H0: mu_FW = mu_MF
Ha: mu_FW != mu_MF

t = {t2['t_statistic']:.3f}
df = {t2['df']}
p = {t2['p_value']:.4f}

Decision:
{t2['decision']}

Interpretation:
{t2_conclusion}


LIMITATIONS
------------------------------------------------------------
1. This is observational tournament data; statistical differences
   do not demonstrate causation.
2. Eligibility rules are used to avoid highly unstable rates from
   players with extremely little playing time or opportunity.
3. Position classification simplifies complex or hybrid player roles.
4. Sampling introduces sampling error, represented through the
   confidence intervals.
5. Results apply to the defined eligible World Cup 2026 populations,
   not automatically to football players in other competitions.
"""

    path = (
        OUTPUT_DIR
        / "slide_ready_results.txt"
    )

    path.write_text(
        summary.strip() + "\n",
        encoding="utf-8",
    )

    return path


# ================================================================
# MAIN
# ================================================================

def main():
    print(
        "HIT140 - FIFA WORLD CUP 2026 "
        "- OFFICIAL FIFA DATA VERSION"
    )

    print(
        "\nThis version bypasses FBref and uses "
        "the approved official FIFA data source."
    )

    master = acquire_fifa_dataset()

    print(
        f"\nMaster player dataset created: "
        f"{len(master)} rows"
    )

    task1 = run_task1(master)
    task2 = run_task2(master)

    summary_path = make_summary(
        task1,
        task2,
    )

    print("\n" + "=" * 68)
    print("ALL ANALYSIS COMPLETED")
    print("=" * 68)

    print(
        "\nData folder:",
        DATA_DIR.resolve(),
    )

    print(
        "Outputs folder:",
        OUTPUT_DIR.resolve(),
    )

    print(
        "Presentation-ready results:",
        summary_path.resolve(),
    )

    print(
        "\nUpload the generated data and output files "
        "to your GitHub repository after checking them."
    )


if __name__ == "__main__":
    main()
