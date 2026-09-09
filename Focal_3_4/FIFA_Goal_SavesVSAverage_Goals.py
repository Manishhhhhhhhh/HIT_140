import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import math
from pathlib import Path

#Adding SCRIPT_DIR makes sure data is accessible as long as csv's,
#are  within the same folder as this script.
SCRIPT_DIR = Path(__file__).parent
# creates the outputs folder if it doesn't exist yet to export
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)  
#importing data
df_2026Saves = pd.read_csv(SCRIPT_DIR /"FIFA_Goalkeeper_2026.csv", header=0)
#confirming column headers
print(df_2026Saves.columns)


#reduce dataset to only relevant
df_2026Saves = df_2026Saves[['Team', 'Goalkeeper Saves']].copy()
#check for excess rows
print(df_2026Saves.shape)

#To compare the average number of goals saved to the average number of goals per game,
#the number of games played should be taken into account.
#104 games x 2 as two teams play / 48 teams gives an average of 4.33 games played per team.
df_2026Saves['Goalkeeper Saves per game'] = df_2026Saves['Goalkeeper Saves'] / 4.33

#The random state is to ensure same sample selection each time tested however the number has no impact
sample_2026Saves = df_2026Saves.sample(n=30, random_state=52)

print(sample_2026Saves.shape)

#Aquire sample codes from relevant values
sample_2026Saves_values= sample_2026Saves["Goalkeeper Saves per game"].values

#asssigning syntatx for average, stand deviation and item count
x_bar_2026= st.tmean(sample_2026Saves_values)
s_2026= st.tstd(sample_2026Saves_values)
n_2026= len(sample_2026Saves_values)

#printing all values to test
print("\t Sample size (2026): %d" % n_2026)
print("\t Sample mean (2026): %.4f" % x_bar_2026)
print("\t Sample std. dev. (2026): %.4f" % s_2026)


#saving descriptive stats to CSV for slides/report
descriptive = pd.DataFrame([
    {"n": n_2026, "mean": x_bar_2026, "std_dev": s_2026},
])

descriptive.to_csv(OUTPUT_DIR / "goalsave_descriptive_statistics.csv", index=False)

#Run histogram to confirm distribution
#2026 histogram
max_val = sample_2026Saves_values.max()
min_val = sample_2026Saves_values.min()
the_range = max_val - min_val
bin_width = 0.4
bin_count= int(the_range/bin_width)

plt.hist(sample_2026Saves_values, color='blue', edgecolor='black', bins=bin_count)
plt.title("Histogram of Goal Saves for each Team (2026 Sample)")
plt.xlabel("Saves")
plt.ylabel("Number of Teams")
plt.savefig(OUTPUT_DIR /"Goalsaves_histogram_2026.png", dpi=200, bbox_inches='tight')
plt.show()

#Confidence interval is calculated to determine how accurate the sample selection is relative to the population.
#The z score or % confidence interval that the sample is accurate to population
#0.975 ensures 2.5% to either side thus giving 95%
z_score = st.norm.ppf(q = 0.975)

#2026 CI
std_err_2026 = s_2026 / math.sqrt(n_2026)
mrg_err = z_score * std_err_2026
ci_low_2026 = x_bar_2026 - mrg_err
ci_upp_2026 = x_bar_2026 + mrg_err

print("Confidence Interval of the mean: %.4f to %.4f" % (ci_low_2026, ci_upp_2026))

#saving CI results to CSV
ci_table = pd.DataFrame([
    {"mean": x_bar_2026, "ci_lower": ci_low_2026, "ci_upper": ci_upp_2026},
])
ci_table.to_csv(OUTPUT_DIR /"Goalsave_confidence_intervals.csv", index=False)

#One Sample T test as the sample data for goal saves is compared to the complete average goals 
#null hypothesis 2026 mean = 2.96
#alternative hypothesis 2026 > 2.96
t_stats, p_val = st.ttest_1samp(sample_2026Saves_values, 2.96, alternative='greater')
print("\n Computing t* ...")
print("\t t-statistic (t*): %.4f" % t_stats)

print("\n Computing p-value ...")
print("\t p-value: %.4f" % p_val)

print("\n Conclusion:")
if p_val < 0.05:
    decision = "We reject the null hypothesis."
else:
    decision = "We fail to reject the null hypothesis."
print("\t " + decision)

#saving t-test results to CSV
pd.DataFrame([{
    "t_statistic": t_stats,
    "p_value": p_val,
    "decision": decision
}]).to_csv(OUTPUT_DIR /"Goalsave_ttest.csv", index=False)