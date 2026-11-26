import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# -----------------------------
# 1. Load and Inspect the Dataset
# -----------------------------
df = pd.read_csv('/Users/brayantchidalu/Documents/Data analyst/project 1/cleaned_shootings.csv')
print("Columns in dataset:", df.columns.tolist())
print("First 5 rows:")
print(df.head())

# -----------------------------
# 2. Check if a Racism Indicator Exists
# -----------------------------
# If you have a direct indicator column (e.g., 'racism_indicator'), use it.
# Otherwise, you might need to create a proxy. For this script, we assume it exists.
if 'racism_indicator' not in df.columns:
    # For demonstration, we create a proxy: For example, if 'victim_race' is present and
    # if the victim's race is not the majority (say, 'White'), we flag it as a potential racist context.
    # Note: This is only a placeholder and should be replaced with your own criteria.
    if 'victim_race' in df.columns:
        df['racism_indicator'] = df['victim_race'].apply(lambda x: 0 if x == 'White' else 1)
    else:
        raise ValueError("No racism indicator column or proxy variable available.")

# Convert racism_indicator to a categorical variable for plotting
df['racism_indicator'] = df['racism_indicator'].astype('category')

# -----------------------------
# 3. Univariate Analysis
# -----------------------------
# Visualize the distributions for the key variables

# Arms Category
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='arms_category', order=df['arms_category'].value_counts().index)
plt.title('Distribution of Arms Category')
plt.xlabel('Arms Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Threat Level
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='threat_level', order=df['threat_level'].value_counts().index)
plt.title('Distribution of Threat Level')
plt.xlabel('Threat Level')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Signs of Mental Illness
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='signs_of_mental_illness', order=df['signs_of_mental_illness'].value_counts().index)
plt.title('Distribution of Signs of Mental Illness')
plt.xlabel('Signs of Mental Illness')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# If They Are Armed
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='if_they_are_armed', order=df['if_they_are_armed'].value_counts().index)
plt.title('Distribution of Armed Status')
plt.xlabel('If They Are Armed')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Bivariate Analysis: Compare by Racism Indicator
# -----------------------------
# a. Arms Category vs. Racism
arms_race_table = pd.crosstab(df['arms_category'], df['racism_indicator'])
print("Contingency Table: Arms Category vs. Racism Indicator")
print(arms_race_table)

plt.figure(figsize=(10, 6))
arms_race_table.plot(kind='bar', stacked=True)
plt.title('Arms Category by Racism Indicator')
plt.xlabel('Arms Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# b. Threat Level vs. Racism
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='threat_level', hue='racism_indicator', order=df['threat_level'].value_counts().index)
plt.title('Threat Level Distribution by Racism Indicator')
plt.xlabel('Threat Level')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# c. Signs of Mental Illness vs. Racism
mental_illness_table = pd.crosstab(df['signs_of_mental_illness'], df['racism_indicator'])
print("Contingency Table: Signs of Mental Illness vs. Racism Indicator")
print(mental_illness_table)

plt.figure(figsize=(8, 6))
mental_illness_table.plot(kind='bar', stacked=True)
plt.title('Signs of Mental Illness by Racism Indicator')
plt.xlabel('Signs of Mental Illness')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# d. If They Are Armed vs. Racism
armed_table = pd.crosstab(df['if_they_are_armed'], df['racism_indicator'])
print("Contingency Table: Armed Status vs. Racism Indicator")
print(armed_table)

plt.figure(figsize=(8, 6))
armed_table.plot(kind='bar', stacked=True)
plt.title('Armed Status by Racism Indicator')
plt.xlabel('If They Are Armed')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Statistical Testing
# -----------------------------
# Perform Chi-Squared tests on the contingency tables to check for significance

def run_chi2_test(contingency_table, var_name):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Squared Test for {var_name}:")
    print(f"Chi2 Statistic: {chi2:.2f}, p-value: {p:.4f}, Degrees of Freedom: {dof}")
    if p < 0.05:
        print("Result: Statistically significant association.")
    else:
        print("Result: No statistically significant association.")

run_chi2_test(arms_race_table, "Arms Category vs. Racism")
run_chi2_test(mental_illness_table, "Signs of Mental Illness vs. Racism")
run_chi2_test(armed_table, "Armed Status vs. Racism")

# For threat level, if it's ordinal/numeric, you might consider converting it to a numeric code.
# Here is an example assuming threat levels can be mapped to numeric values.
if df['threat_level'].dtype == 'object':
    threat_mapping = {level: i for i, level in enumerate(sorted(df['threat_level'].unique()), 1)}
    df['threat_level_numeric'] = df['threat_level'].map(threat_mapping)
    print("Threat level mapping:", threat_mapping)

    # Compare average threat level by racism indicator
    threat_by_race = df.groupby('racism_indicator')['threat_level_numeric'].mean()
    print("\nAverage Threat Level by Racism Indicator:")
    print(threat_by_race)

    # Visualize using a box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='racism_indicator', y='threat_level_numeric')
    plt.title('Threat Level (Numeric) by Racism Indicator')
    plt.xlabel('Racism Indicator (0 = Non-Racist, 1 = Racist)')
    plt.ylabel('Threat Level (Numeric)')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 6. Interpret and Report
# -----------------------------
# At this point, you can document your observations:
# - Compare distributions from the plots and contingency tables.
# - Discuss whether certain arms categories, threat levels, or armed statuses show a statistically significant association with the racism indicator.
# - Note any patterns in the signs of mental illness across groups.
#
# Use these visual and statistical findings to support your narrative on the relationship between shootings and racism.

