import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

heart = pd.read_csv('/Users/kes/Desktop/School/BTYSTE 2020/BTYSTE 2021/Work/ML/heart-disease-predictor/heart.csv')

df = pd.DataFrame(heart)

# for specific features like cholesterol with missing values, probably better to replace NaN with medians.
# # Replace 0 cholesterol values with NaN
# df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
# # Impute missing cholesterol values with median
# df['Cholesterol'].fillna(df['Cholesterol'].median(), inplace=True)

# For HeartDisease = 0
median_chol_0 = df.loc[(df['HeartDisease'] == 0) & (df['Cholesterol'] > 0), 'Cholesterol'].median()
df.loc[(df['HeartDisease'] == 0) & (df['Cholesterol'] == 0), 'Cholesterol'] = median_chol_0

# For HeartDisease = 1
median_chol_1 = df.loc[(df['HeartDisease'] == 1) & (df['Cholesterol'] > 0), 'Cholesterol'].median()
df.loc[(df['HeartDisease'] == 1) & (df['Cholesterol'] == 0), 'Cholesterol'] = median_chol_1

# # Stratified random sampling imputation for Cholesterol == 0
# for status in [0, 1]:
#     # Create a boolean mask for rows where:
#     # - HeartDisease matches the current status
#     # - Cholesterol is 0 (i.e., suspected missing/invalid value)
#     mask = (df['HeartDisease'] == status) & (df['Cholesterol'] == 0)
#     # Select valid (non-zero) Cholesterol values for this specific group (same heart disease status)
#     valid_vals = df.loc[(df['HeartDisease'] == status) & (df['Cholesterol'] > 0), 'Cholesterol']
    
#     # Randomly sample cholesterol values from the valid ones for this group,
#     # with the number of samples equal to the number of zero entries (mask.sum())
#     # `replace=True` allows repeated values (important if valid_vals is small)
#     imputed_vals = np.random.choice(valid_vals, size=mask.sum(), replace=True)
#     df.loc[mask, 'Cholesterol'] = imputed_vals

df = df.dropna()
print(df.shape)


X = df.drop('HeartDisease', axis=1) # data
y = df['HeartDisease'] # target

# columns: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease
# 918 rows x 12 columns
# Target variable = HeartDisease, 11 features
# Try decision tree / random forest


# If a column has more than two unique values, pd.get_dummies() separates them into a column per unique value, and then assigns either 0 or 1 depending if the value is present or not.
X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size = 0.2, random_state = 20) 


# Define your hyperparameters; vary these
max_depth = [2, 10, 20, 50, 100]
max_leaves = [10, 50, 100, 1000]
learning_rate = [0.01, 0.1, 0.3]
n_estimators = [5, 50, 200, 500, 1000]

# Putting hyperparameters into a dataframe to easily view results 
df_depth_max = pd.DataFrame({'Max Depth': max_depth})
df_leaves_max = pd.DataFrame({'Max Leaves': max_leaves})
df_merge_one = df_depth_max.merge(df_leaves_max, how='cross')
df_learning_rate = pd.DataFrame({'Learning Rate': learning_rate})
df_merge_two = df_merge_one.merge(df_learning_rate, how='cross')
df_n_est = pd.DataFrame({'Number of Estimators': n_estimators})
df_merge = df_merge_two.merge(df_n_est, how='cross')

# Creating lists to put results and models into
results = []
models = []


# Loop through the combinations of hyperparameters
for i in range(len(df_merge)):
    dmax = int(df_merge.iloc[i]['Max Depth'])
    lmax = int(df_merge.iloc[i]['Max Leaves'])
    lr = float(df_merge.iloc[i]['Learning Rate'])
    n_est = int(df_merge.iloc[i]['Number of Estimators'])

    # Define and fit model
    clf = XGBClassifier(max_depth = dmax, max_leaves = lmax, learning_rate = lr, n_estimators = n_est, random_state = 1)
    clf.fit(X_train, y_train)


    # Predict on test data and calculate classification accuracy
    y_pred = clf.predict(X_test)
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    # Use ROC
    y_probs = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_probs)
    #print('k-NN ROC-AUC: {:.3f}'.format(auc_score))
    
    # Add results to dataframe
    results_dict = {'Max Depth': dmax, 'Max Leaves': lmax, 'Learning Rate': lr, 'Number of Estimators': n_est, 'Accuracy': accuracy, 'AUC Score': auc_score}
    results.append(results_dict)
    models.append(clf)

    # # plotting roc curve, labelling the hyperparameters
    # fpr, tpr, _ = roc_curve(y_test, y_probs)
    # plt.plot(fpr, tpr, label=f"{dmax}, {minsplit}, {minleaf}, (AUC = {auc_score:.3f})")

# Convert results list to DataFrame
results_df = pd.DataFrame(results)

# Add metrics to df_merge
df_merge['Accuracy'] = results_df['Accuracy']
df_merge['AUC Score'] = results_df['AUC Score']

# Compute average score
df_merge['Avg Score'] = df_merge['Accuracy'] + df_merge['AUC Score']

# Get top 10 best performing models by Avg Score
best_10 = df_merge.nlargest(10, 'Avg Score').reset_index()

print(best_10[['Max Depth', 'Max Leaves', 'Learning Rate', 'Accuracy', 'Number of Estimators', 'AUC Score', 'Avg Score']])


plt.figure(figsize=(8,6))

for idx in best_10['index']:  # these are indices from your original df_merge/models list
    clf = models[idx]

    # Predict probabilities on X_test
    y_probs = clf.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = df_merge.loc[idx, 'AUC Score']

    # Get hyperparams for legend
    dmax = df_merge.loc[idx, 'Max Depth']
    lmax = df_merge.loc[idx, 'Max Leaves']
    lr = df_merge.loc[idx, 'Learning Rate']
    n_est = df_merge.loc[idx, 'Number of Estimators']

    plt.plot(fpr, tpr, label=f"Depth={dmax}, Leaves={lmax}, Learning Rate={lr}, Number of Estimators={n_est}, AUC={auc_score:.3f}")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Top 10 ROC Curves for XGBoost")
plt.legend(loc='lower right', fontsize=8)
plt.grid(True)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

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

# plt.title('Cholesterol Distribution by Heart Disease Status (After Imputation)')
# plt.xlabel('Cholesterol')
# plt.ylabel('Count')
# plt.legend(title='HeartDisease', labels=['No', 'Yes'])
# plt.grid(True)
# plt.tight_layout()
# plt.show()
