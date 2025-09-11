# -*- coding: utf-8 -*-

## Loading libraries and data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------------------------------------------
# Load Data
df = pd.read_csv('T.csv')
print(df.head())
print(df.shape)
print(df.info())

# Drop unwanted column
if 'customerID' in df.columns:
    df = df.drop(['customerID'], axis=1)

# Map binary values
if "SeniorCitizen" in df.columns:
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# Select numerical columns
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Encode categorical variables and store encoders
encoders = {}
for col in df.columns:
    if df[col].dtype == 'object' and col != 'Churn':  # exclude target
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Encode target separately if it's categorical
if df['Churn'].dtype == 'object':
    le_target = LabelEncoder()
    df['Churn'] = le_target.fit_transform(df['Churn'])
    encoders['Churn'] = le_target

# Split data
X = df.drop(columns=['Churn'])
y = df['Churn'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=40, stratify=y
)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# -----------------------------------------------------------------
# Models

# KNN
knn_model = KNeighborsClassifier(n_neighbors=11)
knn_model.fit(X_train, y_train)
print("KNN accuracy:", knn_model.score(X_test, y_test))

# Random Forest
model_rf = RandomForestClassifier(
    n_estimators=500, oob_score=True, n_jobs=-1,
    random_state=50, max_features="sqrt", max_leaf_nodes=30
)
model_rf.fit(X_train, y_train)
print("Random Forest accuracy:", accuracy_score(y_test, model_rf.predict(X_test)))

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
print("Logistic Regression accuracy:", lr_model.score(X_test, y_test))

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
print("Decision Tree accuracy:", dt_model.score(X_test, y_test))

# AdaBoost
a_model = AdaBoostClassifier()
a_model.fit(X_train, y_train)
print("AdaBoost accuracy:", accuracy_score(y_test, a_model.predict(X_test)))

# Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
print("Gradient Boosting accuracy:", accuracy_score(y_test, gb.predict(X_test)))

# -----------------------------------------------------------------
# Save models and preprocessing objects
joblib.dump(gb, "churn_model.pkl")             # final model
joblib.dump(scaler, "scaler.pkl")              # scaler
joblib.dump(X.columns.tolist(), "feature_order.pkl")  # feature order
joblib.dump(encoders, "encoders.pkl")          # label encoders

print("✅ Models, encoders, and preprocessing files saved successfully!")
