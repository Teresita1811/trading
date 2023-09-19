import numpy as np


class LinearRegression:
    def __int__(self):
        self.weights = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the input data.

        Parameters:
        X (numpy.ndarray): Input features of shape (n_samples, n_features).
        y (numpy.ndarray): Target values of shape (n_samples,).

        Returns:
        None
        """
                # Add a bias term to the input features (X)
        X_bias = np.c_[np.ones(X.shape[0]), X]

        # Calculate the weights using the normal equation
        self.weights = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
        ...
    def predict(self, X):
            """
            Make predictions using the trained linear regression model.

            Parameters:
            X (numpy.ndarray): Input features of shape (n_samples, n_features).

            Returns:
            numpy.ndarray: Predicted values of shape (n_samples,).
            """
            if self.weights is None:
                raise ValueError("Model has not been trained. Call fit() first.")

            # Add a bias term to the input features (X)
            X_bias = np.c_[np.ones(X.shape[0]), X]

            # Make predictions
            predictions = X_bias @ self.weights

            return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate the linear regression model using R-squared (R2), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

        Parameters:
        X_test (numpy.ndarray): Test features of shape (n_samples, n_features).
        y_test (numpy.ndarray): True labels of shape (n_samples,).

        Returns:
        float: R-squared (R2) value.
        float: Mean Squared Error (MSE).
        float: Root Mean Squared Error (RMSE).
        """
        if self.weights is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        y_pred = self.predict(X_test)

        # Calculate R-squared (R2)
        mean_y = np.mean(y_test)
        ss_total = np.sum((y_test - mean_y) ** 2)
        ss_residual = np.sum((y_test - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((y_test - y_pred) ** 2)

        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)

        return r2, mse, rmse
