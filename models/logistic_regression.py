import numpy as np
from typing import List

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit the logistic regression model to the input data.

        Parameters:
        X (numpy.ndarray): Input features of shape (n_samples, n_features).
        y (numpy.ndarray): Target values of shape (n_samples,).

        Returns:
        None
        """
        # Initialize model parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent optimization
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Make binary predictions using the trained logistic regression model.

        Parameters:
        X (numpy.ndarray): Input features of shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Binary predictions (0 or 1) of shape (n_samples,).
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)

        # Convert probabilities to binary predictions (0 or 1)
        y_pred_binary = np.where(y_pred > 0.5, 1, 0)

        return y_pred_binary

    def evaluate(self, X_test, y_test, categories: List):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        y_pred = self.predict(X_test)
        # Corrected code snippet
        y_pred = [categories[1] if pred == 0 else 1 for pred in y_pred]

        # Calculate confusion matrix
        tp = np.sum([np.sum((y == 1) & (y2 == 1)) for y, y2 in zip(y_pred, y_test)])
        tn = np.sum([np.sum((y2 == categories[1]) & (y == categories[1])) for y, y2 in zip(y_pred, y_test)])
        fp = np.sum([np.sum((y2 == categories[1]) & (y == 1)) for y, y2 in zip(y_pred, y_test)])
        fn = np.sum([np.sum((y2 == 1) & (y == categories[1])) for y, y2 in zip(y_pred, y_test)])
        # Calculate accuracy (avoid division by zero)
        if tp + tn + fp + fn == 0:
            accuracy = 0.0
        else:
            accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Calculate precision (avoid division by zero)
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        # Calculate recall (avoid division by zero)
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        confusion_matrix = np.array([[tp, fp], [fn, tn]])

        return accuracy, precision, recall, confusion_matrix
