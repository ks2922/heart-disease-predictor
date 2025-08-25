import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Load data
heart = pd.read_csv('/Users/kes/Desktop/School/BTYSTE 2020/BTYSTE 2021/Work/ML/heart.csv')
df = pd.DataFrame(heart)

# Initial cholesterol distribution
sns.histplot(data=df, x='Cholesterol', hue='HeartDisease', kde=True, bins=30, element='step')
plt.title('Cholesterol Distribution by Heart Disease Status')
plt.show()

# # Scatter plot
# plt.scatter(df['Cholesterol'], df['Age'], c=df['HeartDisease'], cmap='coolwarm', alpha=0.6)
# plt.colorbar(label='Heart Disease (0 = No, 1 = Yes)')
# plt.xlabel('Cholesterol')
# plt.ylabel('Age')
# plt.title('Cholesterol vs Age colored by Heart Disease')
# plt.show()

# Stratified random sampling imputation for Cholesterol == 0
for status in [0, 1]:
    # Create a boolean mask for rows where:
    # - HeartDisease matches the current status
    # - Cholesterol is 0 (i.e., suspected missing/invalid value)
    mask = (df['HeartDisease'] == status) & (df['Cholesterol'] == 0)
    # Select valid (non-zero) Cholesterol values for this specific group (same heart disease status)
    valid_vals = df.loc[(df['HeartDisease'] == status) & (df['Cholesterol'] > 0), 'Cholesterol']
    
    # Randomly sample cholesterol values from the valid ones for this group,
    # with the number of samples equal to the number of zero entries (mask.sum())
    # `replace=True` allows repeated values (important if valid_vals is small)
    imputed_vals = np.random.choice(valid_vals, size=mask.sum(), replace=True)
    df.loc[mask, 'Cholesterol'] = imputed_vals

# Updated distribution plot after imputation (all)
plt.figure(figsize=(8, 6))
sns.histplot(
    data=df,
    x='Cholesterol',
    hue='HeartDisease',
    kde=True,
    bins=30,
    palette={0: 'blue', 1: 'orange'},
    element='step',
    stat='count',
    common_norm=False
)
plt.title('Cholesterol Distribution by Heart Disease Status (After Imputation)')
plt.xlabel('Cholesterol')
plt.ylabel('Count')
plt.legend(title='HeartDisease', labels=['No', 'Yes'])
plt.grid(True)
plt.tight_layout()
plt.show()

# New: Distribution plot only for HeartDisease == 1
plt.figure(figsize=(8, 6))
sns.histplot(df[df['HeartDisease'] == 1]['Cholesterol'], bins=30, kde=True, color='orange')
plt.title('Cholesterol Distribution for Heart Disease Positive (After Imputation)')
plt.xlabel('Cholesterol')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.show()

