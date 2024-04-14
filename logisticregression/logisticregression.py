import numpy as np
import time

class LogisticRegression():
    
    def __init__(self, learning_rate=0.001, num_iterations=1000, regularization='none', lambda_param=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.loss_history = []
        self.training_time = 0
        self.lambda_param = lambda_param
        self.regularization = regularization

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y):
        n = X.shape[0]
        z = np.dot(X, self.weights)
        cost = np.mean(np.log(1 + np.exp(-y * z))) 
        if self.regularization == 'l2':
            cost += (self.lambda_param / (2 * n)) * np.sum(np.square(self.weights))
        return cost
    
    def compute_gradient(self, X, y):
        n = X.shape[0]
        z = np.dot(X, self.weights)
        sigmoid_values = self.sigmoid( - y * z)
        gradient = -np.dot(X.T, y * sigmoid_values) / n
        if self.regularization == 'l2':
            gradient += (self.lambda_param / n) * self.weights

        return gradient

    def fit(self, X, y, method = 'gd'):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)  
        X = np.hstack((np.ones((n_samples, 1)), X)) 

        if method == 'gd':
            self.gradient_descent(X,y)
        if method == 'sgd':
            self.stochastic_gradient_descent(X,y)
        

    def gradient_descent(self, X, y):
        start_time = time.time()
        last_cost = 9999
        
        for i in range(self.num_iterations):
            gradient = self.compute_gradient(X, y)
            self.weights -= self.learning_rate * gradient
            self.loss_history.append(self.compute_cost(X, y))
            self.training_time = time.time() - start_time
            cost_dif = abs(self.window_average(10) - last_cost)
            if cost_dif < 0.0005 and i > 10:
                break
            last_cost = self.window_average(10)

    def stochastic_gradient_descent(self, X, y):   
        start_time = time.time()
        n = X.shape[0]
        last_cost = 9999
        np.random.seed(7)
        for t in range(self.num_iterations):
            i = np.random.randint(X.shape[0])  
            X_i = X[i, :].reshape(1, -1)
            y_i = y[i].reshape(-1)
            z = y_i * np.dot(X_i, self.weights)
            gradient = y_i * X_i * (1 / (1 + np.exp(z)))
            gradient = gradient.reshape(self.weights.shape)
            if self.regularization == 'l2':
                gradient -= (self.lambda_param / n) * self.weights

            self.weights += self.learning_rate * gradient
            self.loss_history.append(self.compute_cost(X, y))
            self.training_time = time.time() - start_time
            cost_dif = abs(self.window_average(20) - last_cost)
            if cost_dif < 0.0001  and t > 20:
                break
            last_cost = self.window_average(20)

    def predict(self, X):
        n_samples = X.shape[0]
        X = np.hstack((np.ones((n_samples, 1)), X))
        logits = np.dot(X, self.weights)
        return np.where(self.sigmoid(logits) >= 0.5, 1, -1)
    
    def window_average(self, window_size):
        if len(self.loss_history) < window_size:
            return np.mean(self.loss_history)
        else:
            return np.mean(self.loss_history[-window_size:])