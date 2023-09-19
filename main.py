import numpy as np
from models.logistic_regression import LogisticRegression
from models.linear_regression import LinearRegression

# Logistic Regression
a = np.loadtxt("files/p1_1.txt", delimiter='\t', dtype=float, usecols=range(155))

y_train = [y[0] for y in a]
X_train = [y[1:] for y in a]

y_train = np.array(y_train)
X_train = np.array(X_train)

model = LogisticRegression(learning_rate=0.1, num_iterations=1000)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_train)

accuracy, precision, recall, confusion_matrix = model.evaluate(X_train, y_train, categories=[1,2])

print("Logistic Regression")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n{confusion_matrix}")

# Linear Regression
b = np.loadtxt("files/Reg_1.txt", delimiter='\t', dtype=float)

y_train = np.array([y[0] for y in b])
X_train = np.array([y[1:] for y in b])

model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_train)

print("Linear Regression")

r2, mse, rmse = model.evaluate(X_train, y_train)

print(f"R-squared (R2): {r2}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
