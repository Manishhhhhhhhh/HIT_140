# HIT140 Foundation of Data Science
## FIFA World Cup 2026 — Student Part (Analytic Tasks 1 and 2)

This folder contains the complete working material for the student's two Objective 1 analytic tasks.

### Final task questions

**Task 1 — Passing / Distribution**

> Among FIFA World Cup 2026 defenders and midfielders who played at least 90 minutes and attempted at least 20 passes, is there a difference in average pass-completion rate?

**Task 2 — Shooting / Attacking**

> Among FIFA World Cup 2026 forwards and midfielders who played at least 90 minutes and attempted at least 3 shots, is there a difference in average shot-on-target rate?

These are distinct from the teammate's yellow-card and goalkeeping focal points.

## What is included

- `HIT140_Tasks_1_2_Complete.ipynb` — main Jupyter notebook, with written methodology and all Python analysis.
- `HIT140_Tasks_1_2_Full_Analysis.py` — full script version.
- `requirements.txt` — Python packages.
- `PRESENTATION_NOTES.md` — slide structure and speaking guidance for the student's two tasks.
- `AI_USAGE_DRAFT.md` — draft reflective wording to adapt for the compulsory AI declaration.
- `TEAMS_COLLABORATION_CHECKLIST.md` — evidence checklist for the 20% collaboration/documentation criterion.
- `data/` — source, cleaned, population and sampled CSVs will be saved here.
- `outputs/` — statistics, t-tests, graphs and slide-ready results will be saved here.

## How to run

1. Keep the folder structure unchanged.
2. Open a terminal in this folder.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Start Jupyter:

   `jupyter notebook`

5. Open `HIT140_Tasks_1_2_Complete.ipynb`.
6. Choose **Run All**.
7. Read every output and check there are no errors.
8. Open `outputs/slide_ready_results.txt`. This contains the real numeric results for the PowerPoint.

Alternatively:

`python HIT140_Tasks_1_2_Full_Analysis.py`

## Why results are not hard-coded

The notebook obtains the actual World Cup 2026 data and calculates the real results. Statistical numbers must not be invented. The sample is reproducible because a fixed random seed is used.

## Data and Objective 1 coverage

Each task includes:
- analytic question formulation;
- data acquisition;
- data cleaning and wrangling;
- a merge operation;
- missing-value handling;
- feature construction;
- population and unit-of-observation definition;
- pre-specified inclusion/exclusion rules;
- random sampling;
- descriptive statistics;
- 95% t confidence intervals;
- independent two-sample t-tests;
- State → Plan → Solve → Conclude;
- Python plots;
- interpretation and limitations.

## Important assessment reminders

- All wrangling, statistics and visualisation are done in Python.
- Keep the datasets generated in `data/` with the Python code in the team's OneDrive/GitHub/BitBucket repository.
- Make sure you can explain every line and every statistical decision.
- Do not use the prohibited "average goals scored by a forward" example.
- Show CDU student ID at the beginning of your recorded segment.
- Keep your face visible during the presentation.
- Each individual presentation segment must be no longer than 3 minutes.
- The final team recording must follow the group time requirement.
- Sign and submit the AI Usage Declaration Form.
- Document actual collaboration in the lecturer-created Microsoft Teams space.
