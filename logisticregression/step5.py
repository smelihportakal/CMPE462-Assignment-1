
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from logisticregression import LogisticRegression
from lrdataset import getXy

X, y = getXy()

def plot_loss_convergence(histories, learning_rate_values):
    plt.figure(figsize=(10, 6))
    for i in range(len(histories)):
        plt.plot(histories[i], label=f'{learning_rate_values[i]}')
    plt.title('Effect of Step Size')
    plt.xlabel('Iterations / Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

learning_rate_values = [0.01, 0.05, 0.1, 0.5, 1]
histories = []

for learning_rate_param in learning_rate_values:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(learning_rate=learning_rate_param, num_iterations=500)
    model.fit(X_train, y_train, method='sgd')
    histories.append(model.loss_history)
    predictions = model.predict(X_test)

plot_loss_convergence(histories, learning_rate_values)