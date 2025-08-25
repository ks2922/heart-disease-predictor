# Heart Disease Predictor

## Overview

This project builds a predictive model to assess the likelihood of heart disease based on clinical features such as cholesterol levels, age, blood pressure, and more. It demonstrates end-to-end handling of missing data, data preprocessing, visualization, model training, and evaluation, with potential applications in educational settings or as a prototype for clinical decision support tools.

---

## Dataset

- **Source**: This project uses a combination of five heart disease datasets originally from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/), accessed via a consolidated dataset on Kaggle:  
  [https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data)

- **Included Datasets**:
  - Cleveland
  - Hungarian
  - Switzerland
  - Long Beach VA  
    *(These four are part of the [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) collection from UCI.)*
  - Statlog (Heart)  
    *(Available separately on UCI at: [https://archive.ics.uci.edu/dataset/145/statlog+heart](https://archive.ics.uci.edu/dataset/145/statlog+heart))*

These datasets contain structured patient-level clinical data and are commonly used for binary classification tasks related to heart disease diagnosis.

- **Format**: CSV

- **Features**:
  - Age: age of the patient [years]
  - Sex: sex of the patient [M: Male, F: Female]
  - ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
  - RestingBP: resting blood pressure [mm Hg]
  - Cholesterol: serum cholesterol [mm/dl]
  - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
  - RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
  - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
  - ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
  - Oldpeak: oldpeak = ST [Numeric value measured in depression]
  - ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- **Target**: HeartDisease: output class [1: heart disease, 0: Normal]

---
## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ks2922/heart-disease-predictor.git
   cd heart-disease-predictor

2. Install dependencies:

   pip install -r requirements.txt

## Usage

Run the main prediction script: python src/heart_xgb.py

## Models

This project includes two machine learning models, implemented in separate scripts:

- **Decision Tree**: A simple decision tree classifier for heart disease prediction.
- **XGBoost**: A powerful gradient boosting model for improved prediction accuracy.

Both models were trained on the combined dataset after preprocessing and imputation.

The XGBoost model performed better, with highest accuracy of 89.67% and highest AUC score of 0.965.

## Cholesterol Imputation

Missing cholesterol values were imputed using three different methods:

1. Median cholesterol value computed from the entire dataset.
2. Median cholesterol value computed separately for positive and negative heart disease cases.
3. Stratified random sampling: For missing cholesterol values (coded as 0), cholesterol values were randomly sampled within each heart disease status group (positive or negative) from the existing valid cholesterol measurements. This preserves the distribution of cholesterol within each group.


After testing all three approaches with both the Decision Tree and XGBoost models, version 2 (median imputation separated by heart disease status) yielded the best model performance and was chosen for the final analysis.


## Visualizations

- Cholesterol distribution before and after imputation are shown in a dedicated script and saved plots.
- ROC curves and other model evaluation visuals are saved in the `graphs/` folder.
