# HIT140 Foundation of Data Science
## FIFA World Cup 2026 — Objective 1
### Student Contribution: Analytic Tasks 1 and 2

This repository contains the completed Python analysis for my two assigned Objective 1 analytic tasks in the HIT140 Foundation of Data Science group project.

## Analytic Task 1 — Passing Performance

**Question:**  
Among FIFA World Cup 2026 defenders and midfielders who played at least 90 minutes and attempted at least 20 passes, is there a difference in their average pass-completion rate?

**Feature constructed:**

`pass_completion_rate = passes_completed / passes_attempted * 100`

**Groups compared:**
- Defenders (DF)
- Midfielders (MF)

**Sampling:**
- 30 defenders
- 30 midfielders
- Reproducible random sampling using a fixed random seed

**Inferential method:**
- 95% confidence intervals
- Independent two-sample t-test

## Analytic Task 2 — Shooting Accuracy

**Question:**  
Among FIFA World Cup 2026 forwards and midfielders who played at least 90 minutes and attempted at least 3 shots, is there a difference in their average shot-on-target rate?

**Feature constructed:**

`shot_on_target_rate = shots_on_target / shots * 100`

**Groups compared:**
- Forwards (FW)
- Midfielders (MF)

**Sampling:**
- 30 forwards
- 30 midfielders
- Reproducible random sampling using a fixed random seed

**Inferential method:**
- 95% confidence intervals
- Independent two-sample t-test

## Data Source

The analysis uses FIFA World Cup 2026 data obtained from the official FIFA public data source.

The Python script automatically downloads:
- participating teams;
- squad and player information;
- player positions;
- tournament player statistics.

The downloaded and processed datasets are saved in the `data/` folder.

## Python Skills Used

The analysis applies skills from Weeks 1–5 of HIT140, including:
- analytic question formulation;
- population and sample definition;
- data acquisition;
- data wrangling with `pandas`;
- joining and merging datasets;
- handling missing values;
- feature construction;
- random sampling;
- descriptive statistics;
- 95% confidence intervals;
- hypothesis testing;
- independent two-sample t-tests;
- Python visualisation.

## Repository Structure

```text
HIT_140/
│
├── HIT140_Tasks_1_2_FIFA_Final.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── fifa_2026_teams.csv
│   ├── fifa_2026_players.csv
│   ├── fifa_2026_player_stats_long.csv
│   ├── fifa_2026_player_stats_wide.csv
│   ├── fifa_2026_analysis_master.csv
│   ├── task1_population.csv
│   ├── task1_sample.csv
│   ├── task2_population.csv
│   └── task2_sample.csv
│
└── outputs/
    ├── task1_descriptive_statistics.csv
    ├── task1_confidence_intervals.csv
    ├── task1_ttest.csv
    ├── task1_boxplot.png
    ├── task1_confidence_intervals.png
    ├── task2_descriptive_statistics.csv
    ├── task2_confidence_intervals.csv
    ├── task2_ttest.csv
    ├── task2_boxplot.png
    ├── task2_confidence_intervals.png
    └── slide_ready_results.txt
```

## How to Run

1. Keep the repository folder structure unchanged.

2. Open a terminal in the project folder.

3. Install the required Python packages:

```bash
py -m pip install -r requirements.txt
```

4. Run the final analysis script:

```bash
py HIT140_Tasks_1_2_FIFA_Final.py
```

5. The script will automatically:
   - obtain FIFA World Cup 2026 data;
   - prepare and clean the datasets;
   - create the Task 1 and Task 2 eligible populations;
   - select reproducible random samples;
   - calculate descriptive statistics;
   - calculate 95% confidence intervals;
   - perform independent two-sample t-tests;
   - generate visualisations;
   - save all datasets and results.

6. Open:

```text
outputs/slide_ready_results.txt
```

to view the final numerical results used in the presentation.

## Reproducibility

All statistical results are calculated directly in Python from the downloaded FIFA World Cup 2026 data.

The numerical results are not manually entered into the analysis.

Fixed random seeds are used so that the selected samples can be reproduced when the script is rerun with the same source data.

## Task 1 Final Result

- Defenders mean pass-completion rate: **86.36%**
- Midfielders mean pass-completion rate: **87.87%**
- Two-sample t-test: **t = -0.966**
- **p = 0.342**
- Decision: **Fail to reject H0**

At the 5% significance level, the sample does not provide sufficient statistical evidence that the average pass-completion rate differs between eligible defenders and midfielders.

## Task 2 Final Result

- Forwards mean shot-on-target rate: **37.35%**
- Midfielders mean shot-on-target rate: **31.77%**
- Two-sample t-test: **t = 0.968**
- **p = 0.341**
- Decision: **Fail to reject H0**

At the 5% significance level, the sample does not provide sufficient statistical evidence that the average shot-on-target rate differs between eligible forwards and midfielders.

## Limitations

- The analysis uses observational tournament data, so results should not be interpreted as causal.
- Eligibility thresholds were used to reduce unstable percentages from very small numbers of passes or shots.
- Hybrid player roles were simplified using a consistent primary-position rule.
- Sampling introduces sampling error, which is reflected in the confidence intervals.
- Results apply to the defined FIFA World Cup 2026 eligible populations.

## Academic Integrity and AI Use

Generative AI was used as a support tool for brainstorming analytic questions, structuring the Python workflow, explaining statistical concepts, debugging code, and helping prepare the presentation structure.

All final numerical results were calculated from the FIFA World Cup 2026 data using Python.

AI usage is acknowledged separately in the required CDU AI Usage Declaration Form.
