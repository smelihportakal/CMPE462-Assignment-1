from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from logisticregression import LogisticRegression
from lrdataset import getXy
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score

data = np.genfromtxt('wdbc.data',delimiter=",", usecols=range(2, 32))
results = np.genfromtxt('wdbc.data',delimiter=",", dtype="str", usecols=(1))
results = np.where(results == 'M', -1, 1)

np.random.seed(9)

train_ratio = 0.7
train_size = int(len(data) * train_ratio)
indices = np.arange(len(data))
np.random.shuffle(indices)

train_indices = indices[:train_size]
train_data = data[train_indices]
train_results = results[train_indices]

test_indices = indices[train_size:]
test_data = data[test_indices]
test_results = results[test_indices]


model = LogisticRegression(learning_rate=0.1, num_iterations=500)
model.fit(train_data, train_results, method='gd')
print("model gd time : ", model.training_time)


predictions_gd = model.predict(test_data)
accuracy_gd = accuracy_score(test_results, predictions_gd)
print(f"Model GD accuracy: {accuracy_gd:.8f}")

predictions_gd = model.predict(train_data)
accuracy_gd = accuracy_score(train_results, predictions_gd)
print(f"Model train GD accuracy: {accuracy_gd:.8f}")