import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load your cleaned dataset
df = pd.read_csv('cleaned_shootings.csv')

# Step 1: Filter the data to include only unarmed individuals
unarmed_df = df[df['armed'] == 'unarmed']

# Step 2: Count the number of unarmed individuals by race
unarmed_race_counts = unarmed_df['race'].value_counts()
print("Unarmed Individuals by Race:")
print(unarmed_race_counts)

# Step 3: Count the number of unarmed individuals who were shot by race
unarmed_shot_counts = unarmed_df[unarmed_df['manner_of_death'] == 'shot']['race'].value_counts()
print("\nUnarmed Individuals Shot by Race:")
print(unarmed_shot_counts)

# Step 4: Calculate the proportion of unarmed individuals shot by race
proportion_shot = unarmed_shot_counts / unarmed_race_counts
print("\nProportion of Unarmed Individuals Shot by Race:")
print(proportion_shot)

# Step 5: Create a contingency table for the chi-square test
contingency_table = pd.crosstab(unarmed_df['race'], unarmed_df['manner_of_death'] == 'shot')
print("\nContingency Table:")
print(contingency_table)

# Step 6: Perform the chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2}")
print(f"P-value: {p}")

# Step 7: Interpret the results
if p < 0.05:
    print("\nThere is a statistically significant difference in the likelihood of being shot while unarmed across races.")
else:
    print("\nThere is no statistically significant difference in the likelihood of being shot while unarmed across races.")

# Step 8: Visualize the results
# Bar plot of unarmed individuals shot by race
sns.barplot(x=proportion_shot.index, y=proportion_shot.values)
plt.title('Proportion of Unarmed Individuals Shot by Race')
plt.xlabel('Race')
plt.ylabel('Proportion Shot')
plt.show()