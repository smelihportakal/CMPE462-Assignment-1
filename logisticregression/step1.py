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
df = pd.DataFrame(data).sample(frac=1)
print(df.head())
print(df.describe())
print(df.isnull().sum())
target = df['Class'].apply(lambda x: -1 if x == b'Cammeo' else 1)

features = df.drop(columns=['Class'])

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

features_standardized_df = pd.DataFrame(features_standardized, columns=features.columns)

print(features_standardized_df.head())
print(features_standardized_df.describe().round(2))

print(target.head())
X = features_standardized_df.values
y = target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(learning_rate=0.5, num_iterations=500, regularization= 'l2', lambda_param=0.1)
model.fit(X_train, y_train, method='gd')
print("model gd time : ", model.training_time)

modelsgd = LogisticRegression(learning_rate=0.5, num_iterations=500, regularization= 'l2', lambda_param=0.1)
modelsgd.fit(X_train, y_train, method='sgd')  
print("model sgd time : ", modelsgd.training_time)


def plot_loss_convergence(gd_loss_history, sgd_loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(gd_loss_history, label='Gradient Descent')
    plt.plot(sgd_loss_history, label='Stochastic Gradient Descent', alpha=0.7)
    plt.title('Loss Convergence: GD vs. SGD')
    plt.xlabel('Iterations / Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss_convergence(model.loss_history,modelsgd.loss_history)

predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.2f}")
