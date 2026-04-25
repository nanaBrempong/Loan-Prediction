import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('loan_data.csv')

# 1. Handle missing values by dropping rows with any missing value for simplicity
data = data.dropna()

# 2. Outlier treatment using IQR capping for main numeric columns
for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[col] = np.where(data[col] < lower, lower, data[col])
    data[col] = np.where(data[col] > upper, upper, data[col])

# 3. Discretization
data['ApplicantIncome_bin'] = pd.qcut(data['ApplicantIncome'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
data['CoapplicantIncome_bin'] = pd.qcut(data['CoapplicantIncome'], 4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
data['LoanAmount_bin'] = pd.qcut(data['LoanAmount'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
term_bins = [0, 120, 240, data['Loan_Amount_Term'].max()]
term_labels = ['Short', 'Medium', 'Long']
data['Loan_Amount_Term_bin'] = pd.cut(data['Loan_Amount_Term'], bins=term_bins, labels=term_labels, include_lowest=True)

# 4. Clean and encode Dependents column (replace '3+' with 3)
if 'Dependents' in data.columns:
    data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)

# 5. Encode categorical variables using LabelEncoder
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area',
            'ApplicantIncome_bin', 'CoapplicantIncome_bin', 'LoanAmount_bin', 'Loan_Amount_Term_bin', 'Loan_Status']:
    data[col] = le.fit_transform(data[col])

# 6. Feature selection
X = data.drop(['Loan_Status', 'Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1, errors='ignore')
y = data['Loan_Status']

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train and evaluate classifiers
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f'{name} Accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred))

# 9. Output accuracy scores to file
with open('output.txt', 'w') as f:
    for name, acc in results.items():
        f.write(f'{name} Accuracy: {acc:.4f}\\n')