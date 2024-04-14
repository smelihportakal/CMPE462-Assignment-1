from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from logisticregression import LogisticRegression
from lrdataset import getXy
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score

X, y = getXy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

modelgd = LogisticRegression(learning_rate=0.1, num_iterations=100)
modelgd.fit(X_train, y_train, method='gd')

modelsgd = LogisticRegression(learning_rate=0.1, num_iterations=100)
modelsgd.fit(X_train, y_train, method='sgd')  

modelgdl2 = LogisticRegression(learning_rate=0.1, num_iterations=100, regularization= 'l2', lambda_param=10)
modelgdl2.fit(X_train, y_train, method='gd')

modelsgdl2 = LogisticRegression(learning_rate=0.1, num_iterations=100, regularization= 'l2', lambda_param=10)
modelsgdl2.fit(X_train, y_train, method='sgd')  


predictions_gd = modelgd.predict(X_test)
accuracy_gd = accuracy_score(y_test, predictions_gd)
print(f"Model GD accuracy: {accuracy_gd:.8f}")

predictions_gd = modelgd.predict(X_train)
accuracy_gd = accuracy_score(y_train, predictions_gd)
print(f"Model train GD accuracy: {accuracy_gd:.8f}")

predictions_sgd = modelsgd.predict(X_test)
accuracy_sgd = accuracy_score(y_test, predictions_sgd)
print(f"Model SGD accuracy: {accuracy_sgd:.8f}")

predictions_sgd = modelsgd.predict(X_train)
accuracy_sgd = accuracy_score(y_train, predictions_sgd)
print(f"Model train SGD accuracy: {accuracy_sgd:.8f}")

predictions_gdl2 = modelgdl2.predict(X_test)
accuracy_gdl2 = accuracy_score(y_test, predictions_gdl2)
print(f"Model GD l2 norm accuracy: {accuracy_gdl2:.8f}")

predictions_gdl2 = modelgdl2.predict(X_train)
accuracy_gdl2 = accuracy_score(y_train, predictions_gdl2)
print(f"Model train GD l2 norm accuracy: {accuracy_gdl2:.8f}")

predictions_sgdl2 = modelsgdl2.predict(X_test)
accuracy_sgdl2 = accuracy_score(y_test, predictions_sgdl2)
print(f"Model SGD l2 norm accuracy: {accuracy_sgdl2:.8f}")

predictions_sgdl2 = modelsgdl2.predict(X_train)
accuracy_sgdl2 = accuracy_score(y_train, predictions_sgdl2)
print(f"Model train SGD l2 norm accuracy: {accuracy_sgdl2:.8f}")