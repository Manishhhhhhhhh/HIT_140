"""
HIT140 Foundation of Data Science
FIFA World Cup 2026 — Objective 1
Analytic Task 3: Yellow Card Discipline

Analytic question:
Among FIFA World Cup 2026 players with sufficient playing time,
does average yellow-card rate per 90 minutes differ between defenders
and forwards?

Data source:
data/fifa_2026_analysis_master.csv

This script:
- uses only FIFA World Cup 2026 data;
- filters players with valid minutes, position and yellow-card data;
- constructs yellow cards per 90 minutes;
- compares defenders and forwards;
- uses reproducible random sampling;
- calculates descriptive statistics;
- calculates 95% t-confidence intervals;
- runs an independent two-sample t-test;
- saves graphs and result files.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

# ================================================================
# SETTINGS
# ================================================================

DATA_FILE = Path("data/fifa_2026_analysis_master.csv")
OUTPUT_DIR = Path("CardData/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_MINUTES = 90
SAMPLE_N = 30
RANDOM_SEED = 73
ALPHA = 0.05
CONFIDENCE = 0.95

# ================================================================
# 1. LOAD AND PREPARE DATA
# ================================================================

df = pd.read_csv(DATA_FILE)

required = ["player_name", "team", "position", "minutes", "yellowcards"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        "Missing required columns from fifa_2026_analysis_master.csv: "
        + ", ".join(missing)
    )

data = df[required].copy()

data["minutes"] = pd.to_numeric(data["minutes"], errors="coerce")
data["yellowcards"] = pd.to_numeric(data["yellowcards"], errors="coerce")

data = data.dropna(
    subset=["player_name", "team", "position", "minutes", "yellowcards"]
).copy()

data = data[data["minutes"] >= MIN_MINUTES].copy()

# Keep only the two focal positions.
population = data[
    data["position"].isin(["DF", "FW"])
].copy()

# Construct yellow cards per 90 minutes.
population["yellow_cards_per_90"] = (
    population["yellowcards"] / (population["minutes"] / 90)
)

population = population.replace(
    [np.inf, -np.inf], np.nan
).dropna(subset=["yellow_cards_per_90"])

population.to_csv(
    OUTPUT_DIR / "task3_yellowcard_population.csv",
    index=False
)

print("Eligible population counts:")
print(population["position"].value_counts().sort_index())

# ================================================================
# 2. SAMPLING
# ================================================================

samples = []

for i, position in enumerate(["DF", "FW"]):
    group = population[
        population["position"] == position
    ].copy()

    if len(group) < SAMPLE_N:
        raise ValueError(
            f"Not enough eligible {position} players for n={SAMPLE_N}. "
            f"Available: {len(group)}"
        )

    sampled = group.sample(
        n=SAMPLE_N,
        replace=False,
        random_state=RANDOM_SEED + i
    )

    samples.append(sampled)

sample = pd.concat(samples, ignore_index=True)

sample.to_csv(
    OUTPUT_DIR / "task3_yellowcard_sample.csv",
    index=False
)

print("\nSample counts:")
print(sample["position"].value_counts().sort_index())

# ================================================================
# 3. DESCRIPTIVE STATISTICS
# ================================================================

def descriptive(values):
    s = pd.Series(values).dropna().astype(float)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)

    return {
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
    }

rows = []

for position, part in sample.groupby("position"):
    row = descriptive(part["yellow_cards_per_90"])
    row["group"] = position
    rows.append(row)

desc = pd.DataFrame(rows)[
    [
        "group", "n", "mean", "median",
        "standard_deviation", "variance",
        "minimum", "Q1", "Q3", "IQR",
        "maximum", "range"
    ]
]

print("\nDescriptive statistics:")
print(desc.round(4).to_string(index=False))

desc.to_csv(
    OUTPUT_DIR / "task3_yellowcard_descriptive_statistics.csv",
    index=False
)

# ================================================================
# 4. 95% t-CONFIDENCE INTERVALS
# ================================================================

def mean_t_ci(values, confidence=0.95):
    s = pd.Series(values).dropna().astype(float)

    n = len(s)
    mean = s.mean()
    sd = s.std(ddof=1)
    se = sd / math.sqrt(n)
    dfree = n - 1

    t_critical = st.t.ppf((1 + confidence) / 2, dfree)
    margin = t_critical * se

    return {
        "n": n,
        "mean": mean,
        "standard_deviation": sd,
        "standard_error": se,
        "df": dfree,
        "t_critical": t_critical,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
    }

ci_rows = []

for position, part in sample.groupby("position"):
    row = mean_t_ci(part["yellow_cards_per_90"], CONFIDENCE)
    row["group"] = position
    ci_rows.append(row)

ci = pd.DataFrame(ci_rows)[
    [
        "group", "n", "mean",
        "standard_deviation", "standard_error",
        "df", "t_critical", "ci_lower", "ci_upper"
    ]
]

print("\n95% confidence intervals:")
print(
    ci[
        ["group", "mean", "ci_lower", "ci_upper"]
    ].round(4).to_string(index=False)
)

ci.to_csv(
    OUTPUT_DIR / "task3_yellowcard_confidence_intervals.csv",
    index=False
)

# ================================================================
# 5. TWO-SAMPLE t-TEST
# ================================================================

defenders = sample.loc[
    sample["position"] == "DF",
    "yellow_cards_per_90"
]

forwards = sample.loc[
    sample["position"] == "FW",
    "yellow_cards_per_90"
]

print("\nSTATE")
print(
    "Among eligible FIFA World Cup 2026 players, "
    "is average yellow-card rate per 90 minutes different "
    "between defenders and forwards?"
)

print("\nPLAN")
print("H0: mu_DF = mu_FW")
print("Ha: mu_DF != mu_FW")
print("alpha = 0.05")

# Welch's independent two-sample t-test:
# equal population variances are not assumed.
t_stat, p_value = st.ttest_ind(
    defenders,
    forwards,
    equal_var=False,
    alternative="two-sided"
)

print("\nSOLVE")
print(f"t-statistic = {t_stat:.4f}")
print(f"p-value = {p_value:.6f}")

if p_value <= ALPHA:
    decision = "Reject H0"
    conclusion = (
        "There is statistically significant evidence at the 5% level "
        "that mean yellow-card rate per 90 minutes differs between "
        "eligible defenders and forwards."
    )
else:
    decision = "Fail to reject H0"
    conclusion = (
        "There is insufficient statistical evidence at the 5% level "
        "to conclude that mean yellow-card rate per 90 minutes differs "
        "between eligible defenders and forwards."
    )

print("\nCONCLUDE")
print(decision)
print(conclusion)

pd.DataFrame([{
    "t_statistic": t_stat,
    "p_value": p_value,
    "alpha": ALPHA,
    "decision": decision,
    "conclusion": conclusion
}]).to_csv(
    OUTPUT_DIR / "task3_yellowcard_ttest.csv",
    index=False
)

# ================================================================
# 6. VISUALISATIONS
# ================================================================

plt.figure(figsize=(8, 5))
plt.boxplot(
    [
        defenders.values,
        forwards.values
    ],
    tick_labels=["Defenders", "Forwards"],
    showmeans=True
)
plt.ylabel("Yellow cards per 90 minutes")
plt.xlabel("Position")
plt.title("Yellow-card rate by position — FIFA World Cup 2026")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "task3_yellowcard_boxplot.png",
    dpi=220,
    bbox_inches="tight"
)
plt.close()

ci_plot = ci.set_index("group").loc[["DF", "FW"]].reset_index()

x = np.arange(len(ci_plot))
means = ci_plot["mean"].to_numpy()
lower = means - ci_plot["ci_lower"].to_numpy()
upper = ci_plot["ci_upper"].to_numpy() - means

plt.figure(figsize=(8, 5))
plt.errorbar(
    x,
    means,
    yerr=np.vstack([lower, upper]),
    fmt="o",
    capsize=7
)
plt.xticks(x, ["Defenders", "Forwards"])
plt.ylabel("Mean yellow cards per 90")
plt.xlabel("Position")
plt.title("Mean yellow-card rate with 95% CI")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "task3_yellowcard_confidence_intervals.png",
    dpi=220,
    bbox_inches="tight"
)
plt.close()

# ================================================================
# 7. SLIDE-READY SUMMARY
# ================================================================

ci_index = ci.set_index("group")

mean_df = ci_index.loc["DF", "mean"]
low_df = ci_index.loc["DF", "ci_lower"]
high_df = ci_index.loc["DF", "ci_upper"]

mean_fw = ci_index.loc["FW", "mean"]
low_fw = ci_index.loc["FW", "ci_lower"]
high_fw = ci_index.loc["FW", "ci_upper"]

summary = f"""
TASK 3 — YELLOW CARD DISCIPLINE
===============================

Question:
Among FIFA World Cup 2026 players who played at least {MIN_MINUTES} minutes,
does average yellow-card rate per 90 minutes differ between defenders
and forwards?

Population:
All eligible FIFA World Cup 2026 defenders and forwards in the master dataset.

Unit of observation:
One player.

Feature:
Yellow Cards per 90 =
Yellow Cards / (Minutes / 90)

Sampling:
- n={SAMPLE_N} defenders
- n={SAMPLE_N} forwards
- reproducible random sampling

Defenders:
Mean = {mean_df:.4f}
95% CI = [{low_df:.4f}, {high_df:.4f}]

Forwards:
Mean = {mean_fw:.4f}
95% CI = [{low_fw:.4f}, {high_fw:.4f}]

H0: mu_DF = mu_FW
Ha: mu_DF != mu_FW

t = {t_stat:.4f}
p = {p_value:.6f}

Decision:
{decision}

Conclusion:
{conclusion}

Limitation:
The analysis uses observational tournament data. Position does not necessarily
cause disciplinary differences, and playing style, match context, refereeing,
and team tactics may also influence yellow-card rates.
"""

summary_path = (
    OUTPUT_DIR / "task3_yellowcard_slide_ready_results.txt"
)

summary_path.write_text(
    summary.strip() + "\n",
    encoding="utf-8"
)

print("\nALL TASK 3 YELLOW-CARD ANALYSIS COMPLETED")
print("Saved outputs to:", OUTPUT_DIR.resolve())
print("Slide-ready results:", summary_path.resolve())
