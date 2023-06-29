!pip install pandas sklearn xgboost fuzzywuzzy python-Levenshtein


# Importing Required Libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fuzzywuzzy import fuzz
import string

# Importing Required Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# Load dataset
df = pd.read_excel('/content/Train Dataset.xlsx')

# Convert columns to string
for column in df.columns:
    df[column] = df[column].astype(str)

# Extract TF-IDF features for each split address field
vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, stop_words='english')
X1 = vectorizer.fit_transform(df['Biiling Address1'])
X2 = vectorizer.fit_transform(df['Biiling Address2'])
X3 = vectorizer.fit_transform(df['Billing Post Code'])
X4 = vectorizer.fit_transform(df['Matched Property Address1'])
X5 = vectorizer.fit_transform(df['Matched Property Address2'])
X6 = vectorizer.fit_transform(df['Matched Property Post Code'])
X7 = vectorizer.fit_transform(df['Matched PropertyAU'])

# Concatenate all the features together
X = hstack([X1, X2, X3, X4, X5, X6, X7])

# Define target variable
y = df['Is_Match'].astype(int)  # Convert back to int for the model

# Create train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Logistic Regression Model
lr = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_pred))


# Training Random Forest Model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Random Forest Classifier Accuracy: ", accuracy_score(y_test, y_pred))

# Training XGBoost Model
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print("XGBoost Classifier Accuracy: ", accuracy_score(y_test, y_pred))

# Training SVM Model
svc = svm.SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("SVM Classifier Accuracy: ", accuracy_score(y_test, y_pred))


#MODEL TEST RUN

# Read the test data
df_test = pd.read_excel('/content/Test Dataset.xlsx')
df_admin_aus = pd.read_excel('/content/Admin AUs.xlsx')

# Convert test data columns to string
for column in df_test.columns:
    df_test[column] = df_test[column].astype(str)

for column in df_admin_aus.columns:
    df_admin_aus[column] = df_admin_aus[column].astype(str)

# Combine the test dataset with the 'Admin AUs' dataset by creating a Cartesian product
df_test['key'] = 1
df_admin_aus['key'] = 1
df_combined = pd.merge(df_test, df_admin_aus, on='key').drop('key', axis=1)

# Extract TF-IDF features for each split address field in the combined dataset
X1_test = vectorizer.transform(df_combined['Site Billing Ref'])
X2_test = vectorizer.transform(df_combined['Biiling Address1'])
X3_test = vectorizer.transform(df_combined['Biiling Address2'])
X4_test = vectorizer.transform(df_combined['Billing Post Code'])
X5_test = vectorizer.transform(df_combined['Property Address Line1'])
X6_test = vectorizer.transform(df_combined['Property Address Line2'])
X7_test = vectorizer.transform(df_combined['Property Post Code'])
X8_test = vectorizer.transform(df_combined['PropertyAU'])

# Concatenate all the features together
X_test = hstack([X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, X7_test, X8_test])

# Make predictions with the trained model
y_pred_test = lr.predict(X_test)

# Add predictions to the combined dataframe
df_combined['Is_Match_Prediction'] = y_pred_test

# Save the combined dataframe with predictions to an Excel file
df_combined.to_excel('/content/Predicted_Matches.xlsx', index=False)

