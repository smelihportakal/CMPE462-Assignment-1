import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler


def getXy():
    data, meta = arff.loadarff('Rice_Cammeo_Osmancik.arff')
    df = pd.DataFrame(data).sample(frac=1, random_state= 100)
    target = df['Class'].apply(lambda x: -1 if x == b'Cammeo' else 1) 
    features = df.drop(columns=['Class'])
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)
    features_standardized_df = pd.DataFrame(features_standardized, columns=features.columns)
    X = features_standardized_df.values
    y = target.values
    return X, y