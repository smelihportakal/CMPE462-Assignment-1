from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from logisticregression import LogisticRegression
from lrdataset import getXy
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score

X, y = getXy()

kf = KFold(n_splits=5, random_state=50, shuffle=True)

lambda_values = [0.01, 0.1, 1, 10,100]
average_accuracies = []

for lambda_param in lambda_values:
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LogisticRegression(learning_rate=0.1, num_iterations=200, regularization='l2', lambda_param=lambda_param)
        
        model.fit(X_train, y_train, method='sgd')

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
    
    average_accuracy = np.mean(accuracies)
    average_accuracies.append(average_accuracy)

best_lambda = lambda_values[np.argmax(average_accuracies)]
best_accuracy = max(average_accuracies)
print(f"Best Lambda: {best_lambda}, with an average accuracy of: {best_accuracy}")
