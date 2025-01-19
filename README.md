# LIVER-PATIENT-PREDICTION
# Liver Patient Prediction Project

## Project Overview
This project focuses on predicting liver diseases using patient data. The dataset consists of 583 records with information such as age, gender, various medical test results, and a target variable indicating liver disease.

## Key Objectives
1. **Identify Patients with Liver Disease**: Use classification models to predict liver disease.
2. **Handle Imbalanced Data**: Apply techniques like SMOTE to address class imbalance.
3. **Evaluate Various Models**: Compare models like Logistic Regression, KNN, Decision Tree, Random Forest, and XGBoost.

## Dataset
- **Source**: Indian Liver Patient Dataset.
- **Features**: Age, gender, bilirubin levels, proteins, enzymes, and more.
- **Target**: Binary classification - liver disease (1) or no liver disease (0).

## Technologies and Libraries
- Data Analysis: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`
- Machine Learning: `scikit-learn`, `xgboost`, `imblearn`

## Workflow
### Data Preprocessing
1. Handled missing values by imputing with mean.
2. Encoded categorical variables like gender.
3. Addressed outliers using statistical techniques.
4. Balanced the dataset using SMOTE.

#### Sample Code: Handling Missing Values
```python
# Fill missing values in Albumin/Globulin ratio with mean
import pandas as pd

data['A_G'] = data['A_G'].fillna(data['A_G'].mean())
print("Missing values handled successfully.")
```

### Models Implemented
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

### Visualizing Data
#### Gender Distribution vs Liver Disease
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=data, x='Gender', hue='Target')
plt.title('Gender vs Liver Disease')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(['No Liver Disease', 'Liver Disease'])
plt.show()
```

![Gender Distribution](./images/gender_vs_disease.png)

#### Age Distribution of Patients
```python
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', hue='Target', kde=True)
plt.title('Age Distribution by Target')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend(['No Liver Disease', 'Liver Disease'])
plt.show()
```

![Age Distribution](./images/age_distribution.png)

## Model Evaluation
- Metrics: Accuracy, Recall, Precision, F1 Score, AUC-ROC.
- Best Model: **XGBoost** achieved the highest performance with an F1 Score of 85.03% and reduced false negatives.

### Sample Code: Training XGBoost Model
```python
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Initialize and train the model
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Results
- The XGBoost model was identified as the best-performing model for predicting liver disease.
- Balancing the dataset significantly improved recall and F1 scores, reducing false negatives.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
   ```
2. Execute the script:
   ```bash
   python Liver_Patient_Prediction.py
   ```
3. Review results and visualizations.

## Future Enhancements
- Incorporate additional medical tests for improved predictions.
- Optimize model hyperparameters further for better accuracy.
- Integrate the model into a real-time healthcare system.

## Author
**Team ID**: PTID-CDS-JUL-23-1658

**Category**: Healthcare Prediction
