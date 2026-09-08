# HIT140 Foundation of Data Science
## FIFA World Cup 2026 — Objective 1 Group Project

This repository contains the Python code, datasets, statistical outputs, and visualisations for the group's four Objective 1 analytic tasks for the HIT140 Foundation of Data Science assessment.

All data wrangling, sampling, descriptive statistics, confidence intervals, hypothesis tests, and visualisations are completed in Python.

---

## Final Analytic Tasks

### 1. Passing Performance

**Question**

Among FIFA World Cup 2026 defenders and midfielders who played at least 90 minutes and attempted at least 20 passes, is there a difference in average pass-completion rate?

**Constructed variable**

`pass_completion_rate = passes_completed / passes_attempted * 100`

**Groups compared**
- Defenders (DF)
- Midfielders (MF)

**Sampling**
- 30 defenders
- 30 midfielders
- Reproducible random sampling using a fixed random seed

**Statistical methods**
- Descriptive statistics
- 95% t-confidence intervals
- Independent two-sample t-test

**Final result**
- Defenders mean: **86.36%**
- Midfielders mean: **87.87%**
- `t = -0.966`
- `p = 0.342`
- Decision: **Fail to reject H0**

At the 5% significance level, the sample does not provide sufficient statistical evidence that average pass-completion rate differs between eligible defenders and midfielders.

---

### 2. Shooting Accuracy

**Question**

Among FIFA World Cup 2026 forwards and midfielders who played at least 90 minutes and attempted at least 3 shots, is there a difference in average shot-on-target rate?

**Constructed variable**

`shot_on_target_rate = shots_on_target / shots * 100`

**Groups compared**
- Forwards (FW)
- Midfielders (MF)

**Sampling**
- 30 forwards
- 30 midfielders
- Reproducible random sampling using a fixed random seed

**Statistical methods**
- Descriptive statistics
- 95% t-confidence intervals
- Independent two-sample t-test

**Final result**
- Forwards mean: **37.35%**
- Midfielders mean: **31.77%**
- `t = 0.968`
- `p = 0.341`
- Decision: **Fail to reject H0**

At the 5% significance level, the sample does not provide sufficient statistical evidence that average shot-on-target rate differs between eligible forwards and midfielders.

---

### 3. Yellow-Card Discipline

**Question**

Among FIFA World Cup players who played at least 90 minutes, does average yellow-card rate per 90 minutes differ between players from 2022 and 2026?

**Groups compared**
- 2022
- 2026

**Eligible population**
- 2022 32 teams
- 2026 48 teams

**Sampling**
- 30 teams 2022
- 30 teams 2026
- Reproducible random sampling using fixed random seeds : 32, 48 respectively

**Statistical methods**
- Descriptive statistics
- 95% t-confidence intervals
- Welch independent two-sample t-test

**Final result**
- 2022 mean: **1.8076 yellow cards per 90**
- 2026 mean: **1.3724 yellow cards per 90**
- `t = 2.5274`
- `p = 0.0146`
- Decision: **Reject H0**

At the 5% significance level, there is insufficient statistical evidence to conclude that average yellow-card rate per 90 minutes differs between eligible defenders and forwards.

---

### 4. Goalkeeping Performance

**Question**

Among FIFA World Cup 2026 teams, is average goalkeeper saves greater then the average goals per match?

**Groups compared**
- Goalkeeper Saves per team
- Average match goals 2026 : 2.96

**Eligible population**
- 48 teams


**Sampling**
- 30 teams 
- Reproducible random sampling using fixed random seed : 52

**Statistical methods**
- Descriptive statistics
- 95% t-confidence intervals
- Welch independent one-sample t-test

**Final result**
- Average goals saved per team: **3.1717**
- Average goals scored per match: **1.3176**
- `t = 0.8799`
- `p = 0.1931`
- Decision: **Failure to reject H0**

There isn't statistically significant evidence at the 5% level that mean goalkeeper saves is greater then average goals per match.


---

## Data Source

The analyses use FIFA World Cup 2026 data obtained from the approved FIFA source and datasets prepared for this assessment.

The main FIFA acquisition script downloads and processes:
- participating teams;
- squads and players;
- player positions;
- tournament player statistics.

The resulting player-level datasets are saved in the `data/` folder.

The goalkeeper analysis also uses the cleaned FIFA goalkeeper dataset stored in `CardData/FIFA_Goalkeeper_2026.csv`.

---

## Python Skills Applied

Across the four analytic tasks, the project demonstrates:

- analytic question formulation;
- data acquisition;
- data cleaning and wrangling;
- merging and reshaping data;
- missing-value handling;
- feature construction;
- population definition;
- unit-of-observation definition;
- inclusion and exclusion rules;
- reproducible random sampling;
- descriptive statistics;
- 95% confidence intervals using the t-distribution;
- independent two-sample t-tests;
- hypothesis formulation;
- p-value interpretation;
- State → Plan → Solve → Conclude;
- Python visualisation;
- interpretation of results and limitations.

---

## Repository Structure

```text
HIT_140/
│
├── HIT140_Tasks_1_2_FIFA_Final.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── fifa_2026_analysis_master.csv
│   ├── fifa_2026_player_stats_long.csv
│   ├── fifa_2026_player_stats_wide.csv
│   ├── fifa_2026_players.csv
│   ├── fifa_2026_teams.csv
│   ├── task1_population.csv
│   ├── task1_sample.csv
│   ├── task2_population.csv
│   └── task2_sample.csv
│
├── outputs/
│   ├── slide_ready_results.txt
│   ├── task1_boxplot.png
│   ├── task1_confidence_intervals.csv
│   ├── task1_confidence_intervals.png
│   ├── task1_descriptive_statistics.csv
│   ├── task1_ttest.csv
│   ├── task2_boxplot.png
│   ├── task2_confidence_intervals.csv
│   ├── task2_confidence_intervals.png
│   ├── task2_descriptive_statistics.csv
│   └── task2_ttest.csv
│
└── CardData/
    ├── FIFA_Goalkeeper_2026.csv
    ├── Goalsaves.py
    ├── Yellowcard_2026_Final.py
    └── outputs/
        ├── goalkeeper_boxplot.png
        ├── goalkeeper_clean_population.csv
        ├── goalkeeper_confidence_intervals.csv
        ├── goalkeeper_confidence_intervals.png
        ├── goalkeeper_descriptive_statistics.csv
        ├── goalkeeper_sample.csv
        ├── goalkeeper_slide_ready_results.txt
        ├── goalkeeper_ttest.csv
        ├── yellowcard_population.csv
        ├── yellowcard_sample.csv
        ├── yellowcard_descriptive_statistics.csv
        ├── yellowcard_confidence_intervals.csv
        ├── yellowcard_boxplot.png
        ├── yellowcard_confidence_intervals.png
        ├── yellowcard_ttest.csv
        └── yellowcard_slide_ready_results.txt
```

---

## How to Run

Open a terminal in the repository root.

### Install required packages

```bash
py -m pip install -r requirements.txt
```

### Run Passing and Shooting analyses

```bash
py .\HIT140_Tasks_1_2_FIFA_Final.py
```

### Run Yellow-Card analysis

```bash
py .\CardData\Yellowcard_2026_Final.py
```

### Run Goalkeeping analysis

```bash
py .\CardData\Goalsaves.py
```

Each script saves its calculated statistics and visualisations to the appropriate `outputs/` folder.

---

## Reproducibility

All reported statistical results are calculated in Python from the project datasets.

Fixed random seeds are used so that the selected samples can be reproduced when the same source data and code are used.

Results are not manually hard-coded into the analysis.

---

## Interpretation and Limitations

The analyses use observational tournament data.

Therefore:
- statistically significant results indicate evidence of association, not causation;
- non-significant results do not prove that population means are identical;
- player position, playing style, team tactics, refereeing context, match situation, and tournament structure may influence the observed measures;
- sampling introduces uncertainty, which is represented through the confidence intervals.

---

## Collaboration and Documentation

The group documents its creative process, decisions, progress, reviews, and individual contributions in the lecturer-created Microsoft Teams space, as required by the assessment instructions.

GitHub is used to store and version the Python code, datasets, and analysis outputs.

---

## Academic Integrity and AI Use

Generative AI was used as a support tool for activities such as:
- brainstorming analytic questions;
- structuring Python workflows;
- explaining statistical concepts;
- debugging code;
- reviewing methodology;
- preparing presentation structure and wording.

All final statistical results were calculated from the project datasets using Python.

The group must be able to explain the code and statistical decisions used in the project.

AI use is acknowledged separately in the required signed CDU AI Usage Declaration Form.
