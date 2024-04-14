import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from logisticregression import LogisticRegression

data, meta = arff.loadarff('Rice_Cammeo_Osmancik.arff')
df = pd.DataFrame(data).sample(frac=1, random_state= 100)
print(df.head())
print(df.describe())
target = df['Class'].apply(lambda x: -1 if x == b'Cammeo' else 1)

features = df.drop(columns=['Class'])

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

features_standardized_df = pd.DataFrame(features_standardized, columns=features.columns)

print(features_standardized_df.head())
print(features_standardized_df.describe().round(2))

print(target.head())
