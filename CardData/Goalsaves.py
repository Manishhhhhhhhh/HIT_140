import pandas as pd

import scipy.stats as st
import matplotlib.pyplot as plt
import math

#Is the mean number of goals saved more then the average number of goals for 2026


#importing data
#header=1 ensures the first group labels aren't read as headers as this isn't the relevant ones
df_2026Saves = pd.read_csv("CardData/FIFA_Goalkeeper_2026.csv", header=0)

#confirming column headers
print(df_2026Saves.columns)

#reduce dataset to only relevant and tag year 
df_2026Saves = df_2026Saves[['Team', 'Goalkeeper Saves']].copy()

#check for excess rows
print(df_2026Saves.shape)

#The random state is to ensure same sample selection each time tested however the number has no impact
sample_2026Saves = df_2026Saves.sample(n=30, random_state=48)

print(sample_2026Saves.shape)

#Aquire sample codes from relevant values
sample_2026Saves_values= sample_2026Saves["Goalkeeper Saves"].values

#asssigning syntatx for average, stand deviation and item count
x_bar_2026= st.tmean(sample_2026Saves_values)
s_2026= st.tstd(sample_2026Saves_values)
n_2026= len(sample_2026Saves_values)

#printing all values to test
print("\t Sample size (2026): %d" % n_2026)
print("\t Sample mean (2026): %.2f" % x_bar_2026)
print("\t Sample std. dev. (2026): %.2f" % s_2026)


#2026 histogram
max_val = sample_2026Saves_values.max()
min_val = sample_2026Saves_values.min()
the_range = max_val - min_val
bin_width = 2
bin_count= int(the_range/bin_width)

plt.hist(sample_2026Saves_values, color='blue', edgecolor='black', bins=bin_count)
plt.title("Histogram of Goal Saves for each Team (2026 Sample)")
plt.xlabel("Saves")
plt.ylabel("Number of Teams")
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

print("Confidence Interval of the mean: %.2f to %.2f" % (ci_low_2026, ci_upp_2026))


#One Sample T test as the sample data for goal saves is compared to the complete average goals 
#null hypothesis 2026 mean = 2.96
#alternative hypothesis 2026 > 2.96
t_stats, p_val = st.ttest_1samp(sample_2026Saves_values, 2.96, alternative='greater')
print("\n Computing t* ...")
print("\t t-statistic (t*): %.2f" % t_stats)

print("\n Computing p-value ...")
print("\t p-value: %.4f" % p_val)

print("\n Conclusion:")
if p_val < 0.05:
    print("\t We reject the null hypothesis.")
else:
    print("\t We accept the null hypothesis.")