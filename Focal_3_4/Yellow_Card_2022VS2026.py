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
#header=1 ensures the first group labels aren't read as headers as this isn't the relevant ones
df_2022yellow = pd.read_csv(SCRIPT_DIR / "2022_yellowcards.csv", header=1)
df_2026yellow = pd.read_csv(SCRIPT_DIR / "2026_yellowcards.csv", header=1)

#confirming column headers
print(df_2022yellow.columns)
print(df_2026yellow.columns)

#reduce dataset to only relevant and tag year 
#90s data used for yellow cards instead of raw,
#this accounts for different playtimes across teams due to eliminations.
df_2022yellow = df_2022yellow[['Squad', 'CrdY', '90s']].copy()
df_2026yellow = df_2026yellow[['Squad', 'CrdY', '90s']].copy()
df_2022yellow['year'] = 2022
df_2026yellow['year'] = 2026

#check for excess rows
print(df_2022yellow.shape)
print(df_2026yellow.shape)

#prevent division by zero before calculating the rate below accounting for possible errors
df_2022yellow = df_2022yellow[df_2022yellow['90s'] > 0].copy()
df_2026yellow = df_2026yellow[df_2026yellow['90s'] > 0].copy()

#calculate yellow cards per 90 minutes played
df_2022yellow['CrdY_per90'] = df_2022yellow['CrdY'] / df_2022yellow['90s']
df_2026yellow['CrdY_per90'] = df_2026yellow['CrdY'] / df_2026yellow['90s']

#combine data using ignore_index to avoid duplications and verify data
df_combined = pd.concat([df_2022yellow, df_2026yellow], ignore_index=True)
print(df_combined.head())
print(df_combined.shape)

#DF combined filtered by year for comparison, only possible now after filtering columns and tables has been established.
df_2022yellow = df_combined[df_combined["year"]== 2022]
df_2026yellow = df_combined[df_combined["year"]== 2026]

#sample sizes are selected at a large number that both year have more then.
#Sampling both years ensures an even comparison, while 2022 it doesn't affect much as the difference in possible samples is minimal.
#The random state is to ensure same sample selection each time tested however the number has no impact so I just made it the population sizes.
sample_2022yellow = df_2022yellow.sample(n=30, random_state=32)
sample_2026yellow = df_2026yellow.sample(n=30, random_state=48)

print(sample_2022yellow.shape)
print(sample_2026yellow.shape)

#Aquire sample codes from relevant values found
sample_2022yellow_values= sample_2022yellow["CrdY_per90"].values
sample_2026yellow_values= sample_2026yellow["CrdY_per90"].values

#asssigning syntatx for average, stand deviation and item count
x_bar_2022= st.tmean(sample_2022yellow_values)
x_bar_2026= st.tmean(sample_2026yellow_values)
s_2022= st.tstd(sample_2022yellow_values)
s_2026= st.tstd(sample_2026yellow_values)
n_2022= len(sample_2022yellow_values)
n_2026= len(sample_2026yellow_values)

#printing all values to test
print("\t Sample size (2022): %d" % n_2022)
print("\t Sample size (2026): %d" % n_2026)
print("\t Sample mean (2022): %.4f" % x_bar_2022)
print("\t Sample mean (2026): %.4f" % x_bar_2026)
print("\t Sample std. dev. (2022): %.4f" % s_2022)
print("\t Sample std. dev. (2026): %.4f" % s_2026)

#saving descriptive stats to CSV for slides/report
descriptive = pd.DataFrame([
    {"year": 2022, "n": n_2022, "mean": x_bar_2022, "std_dev": s_2022},
    {"year": 2026, "n": n_2026, "mean": x_bar_2026, "std_dev": s_2026},
])

descriptive.to_csv(OUTPUT_DIR / "yellowcard_descriptive_statistics.csv", index=False)

#Run histogram to confirm distribution
#bin width calculations, kept small as per 90 rates are small
max_val = sample_2022yellow_values.max()
min_val = sample_2022yellow_values.min()
the_range = max_val - min_val
bin_width = 0.2
bin_count= int(the_range/bin_width)

plt.hist(sample_2022yellow_values, color='blue', edgecolor='black', bins=bin_count)
plt.title("Histogram of Yellow Cards per 90 for each Team (2022 Sample)")
plt.xlabel("Yellow Cards per 90")
plt.ylabel("Number of Teams")
plt.savefig(OUTPUT_DIR /"yellowcard_histogram_2022.png", dpi=200, bbox_inches='tight')
plt.show()

#2026 histogram
max_val = sample_2026yellow_values.max()
min_val = sample_2026yellow_values.min()
the_range = max_val - min_val
bin_width = 0.2
bin_count= int(the_range/bin_width)

plt.hist(sample_2026yellow_values, color='blue', edgecolor='black', bins=bin_count)
plt.title("Histogram of Yellow Cards per 90 for each Team (2026 Sample)")
plt.xlabel("Yellow Cards per 90")
plt.ylabel("Number of Teams")
plt.savefig(OUTPUT_DIR /"yellowcard_histogram_2026.png", dpi=200, bbox_inches='tight')
plt.show()

#Confidence interval is calculated to determine how accurate the sample selection is relative to the population.
#The z score or % confidence interval that the sample is accurate to population
#0.975 ensures 2.5% to either side thus giving 95%
z_score = st.norm.ppf(q = 0.975)

#standard error calculation, larger samples are more accurate and wider data ranges are less accurate.
std_err_2022 = s_2022 / math.sqrt(n_2022)
mrg_err_2022 = z_score * std_err_2022
ci_low_2022 = x_bar_2022 - mrg_err_2022
ci_upp_2022 = x_bar_2022 + mrg_err_2022
print("Confidence Interval of the mean (2022): %.4f to %.4f" % (ci_low_2022, ci_upp_2022))

#2026 CI
std_err_2026 = s_2026 / math.sqrt(n_2026)
mrg_err_2026 = z_score * std_err_2026
ci_low_2026 = x_bar_2026 - mrg_err_2026
ci_upp_2026 = x_bar_2026 + mrg_err_2026
print("Confidence Interval of the mean (2026): %.4f to %.4f" % (ci_low_2026, ci_upp_2026))

#saving CI results to CSV
ci_table = pd.DataFrame([
    {"year": 2022, "mean": x_bar_2022, "ci_lower": ci_low_2022, "ci_upper": ci_upp_2022},
    {"year": 2026, "mean": x_bar_2026, "ci_lower": ci_low_2026, "ci_upper": ci_upp_2026},
])
ci_table.to_csv(OUTPUT_DIR /"yellowcard_confidence_intervals.csv", index=False)

#Two sample T test to be input to compare the two sets of data from different years.
#Equal_var=False because the data sets have different populations, this can also be seen visually from the histogram
#alternatve=two-sided simple comparison to see if one equals the other 
#null hypothesis 2022 mean = 2026 mean
#alternative hypothesis 2022 mean =/= 2026 mean
t_stats, p_val = st.ttest_ind_from_stats(x_bar_2022, s_2022, n_2022, x_bar_2026, s_2026, n_2026, equal_var=False, alternative='two-sided')
print("\n Computing t* ...")
print("\t t-statistic (t*): %.2f" % t_stats)

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
}]).to_csv(OUTPUT_DIR /"yellowcard_ttest.csv", index=False)