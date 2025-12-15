import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/manan/Downloads/CVD_dataset.csv")

# Clean / rename incorrect column names
df = df.rename(columns={
    "Smoki0g Status": "Smoking Status",
    "Famil1 Histor1 of CVD": "Family History of CVD",
    "Blood Pressure (mmHg)": "Blood Pressure",
})

df.columns

# If Blood Pressure column contains text values like "130/80"
if df["Blood Pressure"].dtype == "object":
    df[['BP_Systolic', 'BP_Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
    df['BP_Systolic'] = pd.to_numeric(df['BP_Systolic'], errors='coerce')
    df['BP_Diastolic'] = pd.to_numeric(df['BP_Diastolic'], errors='coerce')

df = df.drop(columns=['Blood Pressure'])
df = df.drop(columns=['Height (cm)'])
df.isnull().sum()

# numeric columns filling with median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# categorical columns filling with mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

df.info()
df.head()
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], kde=True, color='blue')
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['BMI'], kde=True, color='green')
plt.title("BMI Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Total Cholesterol (mg/dL)'], kde=True, color='red')
plt.title("Cholesterol Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Fasting Blood Sugar (mg/dL)'], kde=True, color='purple')
plt.title("Blood Sugar Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Estimated LDL (mg/dL)'], kde=True, color='brown')
plt.title("LDL Distribution")
plt.show()

# CVD RISK LEVEL ANALYSIS

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='CVD Risk Level', palette='viridis')
plt.title("CVD Risk Level Counts")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Sex', hue='CVD Risk Level', palette='coolwarm')
plt.title("CVD Risk Level by Sex")
plt.show()

# LIFESTYLE FACTORS VS RISK

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Smoking Status', hue='CVD Risk Level')
plt.title("Smoking vs CVD Risk")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Diabetes Status', hue='CVD Risk Level')
plt.title("Diabetes vs CVD Risk")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Physical Activity Level', hue='CVD Risk Level')
plt.title("Physical Activity Level vs CVD Risk")
plt.xticks(rotation=20)
plt.show()

# CORRELATION HEATMAP

plt.figure(figsize=(12,6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# OUTLIER CHECK (Boxplots)

num_cols = ['BMI', 'Total Cholesterol (mg/dL)', 
            'Fasting Blood Sugar (mg/dL)', 'Estimated LDL (mg/dL)', 'Age']

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col], color='orange')
    plt.title(f"Boxplot of {col}")
    plt.show()

from sklearn.preprocessing import LabelEncoder

# Remove duplicates created earlier
df = df.drop(columns=['BP_Systolic', 'BP_Diastolic'])

# Clean column names
df = df.rename(columns={
    "Smoki0g Status": "Smoking Status",
    "Famil1 Histor1 of CVD": "Family History of CVD",
})

# ENCODE TARGET
le = LabelEncoder()
df['CVD Risk Level Encoded'] = le.fit_transform(df['CVD Risk Level'])

# Drop original target after encoding
target = 'CVD Risk Level Encoded'

# ENCODE categorical columns
cat_cols = ['Sex', 'Physical Activity Level', 'Blood Pressure Category']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# SELECT FEATURE COLUMNS
X = df_encoded.drop(columns=['CVD Risk Level', 'CVD Risk Level Encoded', 'CVD Risk Score'])
y = df_encoded['CVD Risk Level Encoded']


print("Final Feature Count:", X.shape[1])
print("X Shape:", X.shape)
print("y Distribution:\n", y.value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
from imblearn.over_sampling import SMOTE

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)


# Logistic Regression

lr = LogisticRegression(max_iter=300, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("\n Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))
print("Accuracy: ",accuracy_score( y_test , y_pred_lr),"\n") 
#  Decision Tree

dt = DecisionTreeClassifier(
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n Decision Tree Report:")
print(classification_report(y_test, y_pred_dt))
print("Accuracy: ",accuracy_score( y_test , y_pred_dt),"\n") 

# Random Forest

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n Random Forest Report:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy: ",accuracy_score( y_test , y_pred_rf),"\n") 

# Random Forest + SMOTE

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

rf_sm = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf_sm.fit(X_train_sm, y_train_sm)
y_pred_rf_sm = rf_sm.predict(X_test)

print("\n Random Forest + SMOTE Report:")
print(classification_report(y_test, y_pred_rf_sm))
print("Accuracy: ",accuracy_score( y_test , y_pred_rf_sm),"\n") 
#  XGBOOST CLASSIFIER

from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    objective='multi:softprob', 
    num_class=3,
    eval_metric='mlogloss'
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\nXGBOOST REPORT:")
print(classification_report(y_test, y_pred_xgb, zero_division=0))
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))

