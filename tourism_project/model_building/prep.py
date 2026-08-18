import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("tourism_project/data/tourism.csv")
df.drop(columns=["CustomerID"], inplace=True)


# Handling missing values in dataset:

# For numerical columns, filling it with median of column
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, filling it with mode of column
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)


# Processing the Gender column which is having data entry issue (e.g., "Fe Male" should be "Female")
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].str.strip().replace({'Fe Male': 'Female', 'Fe male': 'Female'})

# Encode categorical columns
label_encoder = LabelEncoder()

# List of categorical columns to encode
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
                        'Designation', 'ProductPitched', 'Passport', 'OwnCar']


for col in categorical_features:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))


# NOTE: 'Type' is intentionally left as raw strings (H/L/M).
# The training pipeline one-hot-encodes it, and the Streamlit app also sends
# raw H/L/M values. Encoding it here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.


X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

#Train test split of the dataset
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Saving the datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
#print("Type values kept as:", sorted(X["Type"].unique()))
