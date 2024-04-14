from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from logisticregression import LogisticRegression
from lrdataset import getXy
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score

X, y = getXy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

model = LogisticRegression(learning_rate=0.1, num_iterations=500)
model.fit(X_train, y_train, method='gd')
print("model gd time : ", model.training_time)

modelsgd = LogisticRegression(learning_rate=0.1, num_iterations=500)
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
print(f"Model GD accuracy: {accuracy:.2f}")


predictions = modelsgd.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model SGD accuracy: {accuracy:.2f}")