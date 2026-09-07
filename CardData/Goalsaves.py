"""
HIT140 Foundation of Data Science
FIFA World Cup 2026 — Objective 1
Analytic Task 4: Goalkeeping

Analytic question:
Among FIFA World Cup 2026 teams, is the average goalkeeper save percentage
different between teams that recorded at least one clean sheet and teams that
recorded no clean sheets?

This script accepts either:
1. the original malformed 96-row FIFA goalkeeper CSV, or
2. the already-cleaned 48-row FIFA goalkeeper CSV.

It automatically detects which version is present.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

DATA_FILE = Path("CardData/FIFA_Goalkeeper_2026.csv")
OUTPUT_DIR = Path("CardData/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 48
SAMPLE_N_PER_GROUP = 20
ALPHA = 0.05
CONFIDENCE = 0.95


# ================================================================
# 1. LOAD AND CLEAN
# ================================================================

raw = pd.read_csv(DATA_FILE)
print("Raw dataset shape:", raw.shape)

# If the file already contains a valid Team column and Save Percentage,
# it is the cleaned 48-team version and should NOT be halved again.
if (
    len(raw) == 48
    and "Save Percentage" in raw.columns
    and raw["Team"].notna().sum() == 48
):
    clean = raw.copy()
    print("Detected already-cleaned 48-team dataset.")

else:
    # Original raw FIFA file alternates:
    # statistics row, then team-name row.
    stat_rows = raw.iloc[::2].reset_index(drop=True)
    team_rows = raw.iloc[1::2].reset_index(drop=True)

    clean = stat_rows.copy()
    clean["Team"] = team_rows["Team"].values

    clean = clean.rename(columns={
        "Goalkeeper Actions Outside the Pentalty Area":
        "Goalkeeper Actions Outside the Penalty Area"
    })

    required = [
        "Team",
        "Clean Sheets",
        "Goals Conceded",
        "Goalkeeper Saves",
        "Goalkeeper Actions Inside the Penalty Area",
    ]

    clean = clean.dropna(subset=required).copy()

    clean["Shots on Target Faced (proxy)"] = (
        clean["Goalkeeper Saves"] + clean["Goals Conceded"]
    )

    clean = clean[
        clean["Shots on Target Faced (proxy)"] > 0
    ].copy()

    clean["Save Percentage"] = (
        clean["Goalkeeper Saves"]
        / clean["Shots on Target Faced (proxy)"]
        * 100
    )

    clean["Clean Sheet Group"] = np.where(
        clean["Clean Sheets"] > 0,
        "At least 1 clean sheet",
        "No clean sheet"
    )

# Ensure required constructed fields exist in either version.
if "Shots on Target Faced (proxy)" not in clean.columns:
    clean["Shots on Target Faced (proxy)"] = (
        clean["Goalkeeper Saves"] + clean["Goals Conceded"]
    )

if "Save Percentage" not in clean.columns:
    clean["Save Percentage"] = (
        clean["Goalkeeper Saves"]
        / clean["Shots on Target Faced (proxy)"]
        * 100
    )

if "Clean Sheet Group" not in clean.columns:
    clean["Clean Sheet Group"] = np.where(
        clean["Clean Sheets"] > 0,
        "At least 1 clean sheet",
        "No clean sheet"
    )

clean = clean.dropna(
    subset=[
        "Team",
        "Clean Sheets",
        "Goals Conceded",
        "Goalkeeper Saves",
        "Save Percentage",
        "Clean Sheet Group",
    ]
).copy()

print("\nClean dataset shape:", clean.shape)
print("\nPopulation counts:")
print(clean["Clean Sheet Group"].value_counts())

clean.to_csv(
    OUTPUT_DIR / "goalkeeper_clean_population.csv",
    index=False
)


# ================================================================
# 2. POPULATION AND SAMPLING
# ================================================================

groups = [
    "At least 1 clean sheet",
    "No clean sheet"
]

samples = []

for i, group in enumerate(groups):
    population_group = clean[
        clean["Clean Sheet Group"] == group
    ].copy()

    if len(population_group) < SAMPLE_N_PER_GROUP:
        raise ValueError(
            f"Not enough teams in '{group}' for n={SAMPLE_N_PER_GROUP}. "
            f"Available: {len(population_group)}"
        )

    samples.append(
        population_group.sample(
            n=SAMPLE_N_PER_GROUP,
            replace=False,
            random_state=RANDOM_SEED + i
        )
    )

sample = pd.concat(samples, ignore_index=True)

sample.to_csv(
    OUTPUT_DIR / "goalkeeper_sample.csv",
    index=False
)

print("\nSample counts:")
print(sample["Clean Sheet Group"].value_counts())


# ================================================================
# 3. DESCRIPTIVE STATISTICS
# ================================================================

def describe(values):
    values = pd.Series(values).dropna().astype(float)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    return {
        "n": len(values),
        "mean": values.mean(),
        "median": values.median(),
        "standard_deviation": values.std(ddof=1),
        "variance": values.var(ddof=1),
        "minimum": values.min(),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "maximum": values.max(),
        "range": values.max() - values.min(),
    }

rows = []

for group, part in sample.groupby("Clean Sheet Group"):
    row = describe(part["Save Percentage"])
    row["group"] = group
    rows.append(row)

descriptive = pd.DataFrame(rows)[
    [
        "group", "n", "mean", "median",
        "standard_deviation", "variance",
        "minimum", "Q1", "Q3", "IQR",
        "maximum", "range"
    ]
]

print("\nDescriptive statistics:")
print(descriptive.round(3).to_string(index=False))

descriptive.to_csv(
    OUTPUT_DIR / "goalkeeper_descriptive_statistics.csv",
    index=False
)


# ================================================================
# 4. 95% t-CONFIDENCE INTERVALS
# ================================================================

def mean_t_ci(values, confidence=0.95):
    values = pd.Series(values).dropna().astype(float)

    n = len(values)
    mean = values.mean()
    sd = values.std(ddof=1)
    se = sd / math.sqrt(n)

    df = n - 1
    t_critical = st.t.ppf((1 + confidence) / 2, df)
    margin = t_critical * se

    return {
        "n": n,
        "mean": mean,
        "standard_deviation": sd,
        "standard_error": se,
        "df": df,
        "t_critical": t_critical,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
    }

ci_rows = []

for group, part in sample.groupby("Clean Sheet Group"):
    row = mean_t_ci(part["Save Percentage"], CONFIDENCE)
    row["group"] = group
    ci_rows.append(row)

ci_table = pd.DataFrame(ci_rows)[
    [
        "group", "n", "mean",
        "standard_deviation", "standard_error",
        "df", "t_critical", "ci_lower", "ci_upper"
    ]
]

print("\n95% confidence intervals:")
print(
    ci_table[
        ["group", "mean", "ci_lower", "ci_upper"]
    ].round(3).to_string(index=False)
)

ci_table.to_csv(
    OUTPUT_DIR / "goalkeeper_confidence_intervals.csv",
    index=False
)


# ================================================================
# 5. INDEPENDENT TWO-SAMPLE t-TEST
# ================================================================

clean_sheet_values = sample.loc[
    sample["Clean Sheet Group"] == "At least 1 clean sheet",
    "Save Percentage"
]

no_clean_sheet_values = sample.loc[
    sample["Clean Sheet Group"] == "No clean sheet",
    "Save Percentage"
]

print("\nSTATE")
print(
    "Is average goalkeeper save percentage different between teams "
    "with at least one clean sheet and teams with no clean sheets?"
)

print("\nPLAN")
print("H0: mu_clean_sheet = mu_no_clean_sheet")
print("Ha: mu_clean_sheet != mu_no_clean_sheet")
print("alpha = 0.05")

t_stat, p_value = st.ttest_ind(
    clean_sheet_values,
    no_clean_sheet_values,
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
        "that mean goalkeeper save percentage differs between the two groups."
    )
else:
    decision = "Fail to reject H0"
    conclusion = (
        "There is insufficient statistical evidence at the 5% level "
        "to conclude that mean goalkeeper save percentage differs "
        "between the two groups."
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
    OUTPUT_DIR / "goalkeeper_ttest.csv",
    index=False
)


# ================================================================
# 6. VISUALISATIONS
# ================================================================

plot_data = [
    sample.loc[
        sample["Clean Sheet Group"] == group,
        "Save Percentage"
    ].values
    for group in groups
]

plt.figure(figsize=(8, 5))
plt.boxplot(
    plot_data,
    tick_labels=groups,
    showmeans=True
)
plt.ylabel("Save percentage (%)")
plt.xlabel("Clean-sheet group")
plt.title("Goalkeeper save percentage by clean-sheet group")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "goalkeeper_boxplot.png",
    dpi=220,
    bbox_inches="tight"
)
plt.close()

ci_plot = ci_table.set_index("group").loc[groups].reset_index()

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
plt.xticks(x, groups)
plt.ylabel("Mean save percentage (%)")
plt.xlabel("Clean-sheet group")
plt.title("Mean goalkeeper save percentage with 95% CI")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "goalkeeper_confidence_intervals.png",
    dpi=220,
    bbox_inches="tight"
)
plt.close()


# ================================================================
# 7. SLIDE-READY SUMMARY
# ================================================================

ci_index = ci_table.set_index("group")

mean_a = ci_index.loc["At least 1 clean sheet", "mean"]
low_a = ci_index.loc["At least 1 clean sheet", "ci_lower"]
high_a = ci_index.loc["At least 1 clean sheet", "ci_upper"]

mean_b = ci_index.loc["No clean sheet", "mean"]
low_b = ci_index.loc["No clean sheet", "ci_lower"]
high_b = ci_index.loc["No clean sheet", "ci_upper"]

summary = f"""
GOALKEEPER ANALYTIC TASK — FINAL RESULTS
========================================

Question:
Among FIFA World Cup 2026 teams, is average goalkeeper save percentage
different between teams that recorded at least one clean sheet and teams
that recorded no clean sheets?

Population:
All 48 FIFA World Cup 2026 teams represented in the cleaned goalkeeper dataset.

Unit of observation:
One national team.

Feature:
Save Percentage =
Goalkeeper Saves / (Goalkeeper Saves + Goals Conceded) * 100

Sampling:
Stratified random sample:
- n={SAMPLE_N_PER_GROUP} teams with at least one clean sheet
- n={SAMPLE_N_PER_GROUP} teams with no clean sheet

At least 1 clean sheet:
Mean = {mean_a:.2f}%
95% CI = [{low_a:.2f}%, {high_a:.2f}%]

No clean sheet:
Mean = {mean_b:.2f}%
95% CI = [{low_b:.2f}%, {high_b:.2f}%]

H0: mu_clean_sheet = mu_no_clean_sheet
Ha: mu_clean_sheet != mu_no_clean_sheet

t = {t_stat:.3f}
p = {p_value:.4f}

Decision:
{decision}

Conclusion:
{conclusion}

Limitation:
Save percentage is constructed from tournament totals. This is observational
data, so any detected difference should be interpreted as an association
rather than a causal effect.
"""

summary_path = (
    OUTPUT_DIR / "goalkeeper_slide_ready_results.txt"
)

summary_path.write_text(
    summary.strip() + "\n",
    encoding="utf-8"
)

print("\nALL GOALKEEPER ANALYSIS COMPLETED")
print("Saved outputs to:", OUTPUT_DIR.resolve())
print("Slide-ready results:", summary_path.resolve())
