import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

heart = pd.read_csv('/Users/kes/Desktop/School/BTYSTE 2020/BTYSTE 2021/Work/ML/heart-disease-predictor/data/heart.csv')

df = pd.DataFrame(heart)

# for specific features like cholesterol with missing values, probably better to replace NaN with medians.

# # VERSION 1
# # Replace 0 cholesterol values with NaN
# df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
# # Impute missing cholesterol values with median
# df['Cholesterol'].fillna(df['Cholesterol'].median(), inplace=True)

# # VERSION 2
# # For HeartDisease = 0
# median_chol_0 = df.loc[(df['HeartDisease'] == 0) & (df['Cholesterol'] > 0), 'Cholesterol'].median()
# df.loc[(df['HeartDisease'] == 0) & (df['Cholesterol'] == 0), 'Cholesterol'] = median_chol_0

# # For HeartDisease = 1
# median_chol_1 = df.loc[(df['HeartDisease'] == 1) & (df['Cholesterol'] > 0), 'Cholesterol'].median()
# df.loc[(df['HeartDisease'] == 1) & (df['Cholesterol'] == 0), 'Cholesterol'] = median_chol_1

# VERSION 3
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

df = df.dropna()
print(df.shape)


X = df.drop('HeartDisease', axis=1) # data
y = df['HeartDisease'] # target

# columns: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease
# 918 rows x 12 columns
# Target variable = HeartDisease, 11 features
# Try decision tree / random forest



X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size = 0.2, random_state = 20) 


# Define your hyperparameters; vary these
max_depth = np.arange(10,100,10)
min_samples_split = np.arange(2,20,5)
min_samples_leaf = np.arange(1,20,5)


# Putting hyperparameters into a dataframe to easily view results 
df_depth_max = pd.DataFrame({'Max Depth': max_depth})
df_minsampsplit = pd.DataFrame({'Min Samples Split': min_samples_split})
df_merge_one = df_depth_max.merge(df_minsampsplit, how='cross')
df_minsampleaf = pd.DataFrame({'Min Samples Leaf': min_samples_leaf})
df_merge = df_merge_one.merge(df_minsampleaf, how='cross')

# Creating lists to put results and models into
results = []
models = []


# Loop through the combinations of hyperparameters
for i in range(len(df_merge)):
    dmax = df_merge.iloc[i]['Max Depth']
    minsplit = df_merge.iloc[i]['Min Samples Split']
    minleaf = df_merge.iloc[i]['Min Samples Leaf']

    # Define and fit model
    clf = tree.DecisionTreeClassifier(max_depth = dmax, 
                                  min_samples_split = minsplit, 
                                  min_samples_leaf = minleaf,
                                  random_state = 1)
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
    results_dict = {'Max Depth': dmax, 'Min Samples Split': minsplit, 'Min Samples Leaf': minleaf, 'Accuracy': accuracy, 'AUC Score': auc_score}
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

print(best_10[['Max Depth', 'Min Samples Split', 'Min Samples Leaf', 'Accuracy', 'AUC Score', 'Avg Score']])


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
    minsplit = df_merge.loc[idx, 'Min Samples Split']
    minleaf = df_merge.loc[idx, 'Min Samples Leaf']

    plt.plot(fpr, tpr, label=f"Depth={dmax}, Split={minsplit}, Leaf={minleaf}, AUC={auc_score:.3f}")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Top 10 ROC Curves for Decision Tree Classifier")
plt.legend(loc='lower right', fontsize=8)
plt.grid(True)
plt.show()





