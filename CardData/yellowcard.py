"""
HIT140 Foundation of Data Science
FIFA World Cup — Objective 1
Analytic Task 3: Yellow Card Discipline

Analytic question:
Is the average number of yellow cards per 90 minutes different between
teams in the FIFA World Cup 2022 and FIFA World Cup 2026?

This corrected version:
- uses yellow cards per 90 instead of raw tournament totals;
- uses reproducible random sampling;
- calculates descriptive statistics;
- uses t-based 95% confidence intervals;
- uses Welch's independent two-sample t-test;
- uses correct "fail to reject H0" wording;
- saves all outputs and plots.
"""

from pathlib import Path
import math
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt


# ================================================================
# SETTINGS
# ================================================================

DATA_2022 = Path("CardData/2022_yellowcards.csv")
DATA_2026 = Path("CardData/2026_yellowcards.csv")
OUTPUT_DIR = Path("CardData/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_N = 30
RANDOM_SEED_2022 = 32
RANDOM_SEED_2026 = 48
ALPHA = 0.05
CONFIDENCE = 0.95


# ================================================================
# 1. LOAD AND WRANGLE
# ================================================================

df_2022 = pd.read_csv(DATA_2022, header=1)
df_2026 = pd.read_csv(DATA_2026, header=1)

print("2022 raw shape:", df_2022.shape)
print("2026 raw shape:", df_2026.shape)

required = ["Squad", "90s", "CrdY"]

df_2022 = df_2022[required].copy()
df_2026 = df_2026[required].copy()

df_2022["Year"] = 2022
df_2026["Year"] = 2026

# Remove rows missing required variables.
df_2022 = df_2022.dropna(subset=required).copy()
df_2026 = df_2026.dropna(subset=required).copy()

# Prevent division by zero.
df_2022 = df_2022[df_2022["90s"] > 0].copy()
df_2026 = df_2026[df_2026["90s"] > 0].copy()

# Feature construction:
# CrdY = total yellow cards accumulated by the team.
# Dividing by 90s standardises the total for tournament playing time.
df_2022["Yellow Cards per 90"] = (
    df_2022["CrdY"] / df_2022["90s"]
)

df_2026["Yellow Cards per 90"] = (
    df_2026["CrdY"] / df_2026["90s"]
)

print("\nPopulation sizes:")
print("2022:", len(df_2022))
print("2026:", len(df_2026))

df_2022.to_csv(
    OUTPUT_DIR / "yellowcard_population_2022.csv",
    index=False
)
df_2026.to_csv(
    OUTPUT_DIR / "yellowcard_population_2026.csv",
    index=False
)


# ================================================================
# 2. SAMPLING
# ================================================================

if len(df_2022) < SAMPLE_N:
    raise ValueError(
        f"2022 population has only {len(df_2022)} teams; "
        f"cannot sample n={SAMPLE_N}."
    )

if len(df_2026) < SAMPLE_N:
    raise ValueError(
        f"2026 population has only {len(df_2026)} teams; "
        f"cannot sample n={SAMPLE_N}."
    )

sample_2022 = df_2022.sample(
    n=SAMPLE_N,
    replace=False,
    random_state=RANDOM_SEED_2022
)

sample_2026 = df_2026.sample(
    n=SAMPLE_N,
    replace=False,
    random_state=RANDOM_SEED_2026
)

sample_2022.to_csv(
    OUTPUT_DIR / "yellowcard_sample_2022.csv",
    index=False
)
sample_2026.to_csv(
    OUTPUT_DIR / "yellowcard_sample_2026.csv",
    index=False
)

print("\nSample sizes:")
print("2022:", len(sample_2022))
print("2026:", len(sample_2026))


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


desc_2022 = describe(sample_2022["Yellow Cards per 90"])
desc_2022["year"] = 2022

desc_2026 = describe(sample_2026["Yellow Cards per 90"])
desc_2026["year"] = 2026

descriptive = pd.DataFrame(
    [desc_2022, desc_2026]
)[
    [
        "year", "n", "mean", "median",
        "standard_deviation", "variance",
        "minimum", "Q1", "Q3", "IQR",
        "maximum", "range"
    ]
]

print("\nDescriptive statistics:")
print(descriptive.round(3).to_string(index=False))

descriptive.to_csv(
    OUTPUT_DIR / "yellowcard_descriptive_statistics.csv",
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


ci_2022 = mean_t_ci(
    sample_2022["Yellow Cards per 90"],
    CONFIDENCE
)
ci_2022["year"] = 2022

ci_2026 = mean_t_ci(
    sample_2026["Yellow Cards per 90"],
    CONFIDENCE
)
ci_2026["year"] = 2026

ci_table = pd.DataFrame(
    [ci_2022, ci_2026]
)[
    [
        "year", "n", "mean",
        "standard_deviation", "standard_error",
        "df", "t_critical", "ci_lower", "ci_upper"
    ]
]

print("\n95% confidence intervals:")
print(
    ci_table[
        ["year", "mean", "ci_lower", "ci_upper"]
    ].round(3).to_string(index=False)
)

ci_table.to_csv(
    OUTPUT_DIR / "yellowcard_confidence_intervals.csv",
    index=False
)


# ================================================================
# 5. INDEPENDENT TWO-SAMPLE t-TEST
# ================================================================

values_2022 = sample_2022["Yellow Cards per 90"]
values_2026 = sample_2026["Yellow Cards per 90"]

print("\nSTATE")
print(
    "Is average yellow cards per 90 minutes different between "
    "FIFA World Cup 2022 teams and FIFA World Cup 2026 teams?"
)

print("\nPLAN")
print("H0: mu_2022 = mu_2026")
print("Ha: mu_2022 != mu_2026")
print("alpha = 0.05")

# Welch's independent two-sample t-test is used because
# equal population variances are not assumed.
t_stat, p_value = st.ttest_ind(
    values_2022,
    values_2026,
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
        "that mean yellow cards per 90 minutes differs between "
        "World Cup 2022 and World Cup 2026 teams."
    )
else:
    decision = "Fail to reject H0"
    conclusion = (
        "There is insufficient statistical evidence at the 5% level "
        "to conclude that mean yellow cards per 90 minutes differs "
        "between World Cup 2022 and World Cup 2026 teams."
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
    OUTPUT_DIR / "yellowcard_ttest.csv",
    index=False
)


# ================================================================
# 6. VISUALISATIONS
# ================================================================

plt.figure(figsize=(8, 5))
plt.boxplot(
    [
        values_2022.values,
        values_2026.values
    ],
    tick_labels=["2022", "2026"],
    showmeans=True
)
plt.ylabel("Yellow cards per 90 minutes")
plt.xlabel("Tournament year")
plt.title("Yellow-card rate: World Cup 2022 vs 2026")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "yellowcard_boxplot.png",
    dpi=220,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(
    values_2022,
    bins=7,
    alpha=0.6,
    label="2022"
)
plt.hist(
    values_2026,
    bins=7,
    alpha=0.6,
    label="2026"
)
plt.xlabel("Yellow cards per 90 minutes")
plt.ylabel("Number of teams")
plt.title("Distribution of yellow-card rate")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "yellowcard_histogram.png",
    dpi=220,
    bbox_inches="tight"
)
plt.close()


# ================================================================
# 7. SLIDE-READY SUMMARY
# ================================================================

ci_index = ci_table.set_index("year")

mean_2022 = ci_index.loc[2022, "mean"]
low_2022 = ci_index.loc[2022, "ci_lower"]
high_2022 = ci_index.loc[2022, "ci_upper"]

mean_2026 = ci_index.loc[2026, "mean"]
low_2026 = ci_index.loc[2026, "ci_lower"]
high_2026 = ci_index.loc[2026, "ci_upper"]

summary = f"""
YELLOW-CARD ANALYTIC TASK — FINAL RESULTS
=========================================

Question:
Is average yellow cards per 90 minutes different between
FIFA World Cup 2022 teams and FIFA World Cup 2026 teams?

Population:
All teams represented in the supplied World Cup 2022 and 2026
discipline datasets.

Unit of observation:
One national team.

Feature:
Yellow Cards per 90 =
Total Yellow Cards / Team 90s

Sampling:
Simple random sample:
- n={SAMPLE_N} teams from 2022
- n={SAMPLE_N} teams from 2026

2022:
Mean = {mean_2022:.3f} cards per 90
95% CI = [{low_2022:.3f}, {high_2022:.3f}]

2026:
Mean = {mean_2026:.3f} cards per 90
95% CI = [{low_2026:.3f}, {high_2026:.3f}]

H0: mu_2022 = mu_2026
Ha: mu_2022 != mu_2026

t = {t_stat:.3f}
p = {p_value:.4f}

Decision:
{decision}

Conclusion:
{conclusion}

Limitation:
The analysis compares team-level tournament rates across two World Cups.
Changes in tournament format, teams, refereeing context and match situations
may also influence yellow-card rates. The result describes an association
between tournament year and the observed discipline rate, not causation.
"""

summary_path = (
    OUTPUT_DIR / "yellowcard_slide_ready_results.txt"
)

summary_path.write_text(
    summary.strip() + "\n",
    encoding="utf-8"
)

print("\nALL YELLOW-CARD ANALYSIS COMPLETED")
print("Saved outputs to:", OUTPUT_DIR.resolve())
print("Slide-ready results:", summary_path.resolve())
