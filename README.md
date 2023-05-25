# Regression
 Simples Linear Regression, Multiple Simple Linear Regression, Polynomial Regression, Logistic Regression



# Linear Regression 

import numpy as np
from sklearn.linear_model import LinearRegression

A = np.array(input("Enter the values of independent variable (A): ").split(), dtype=float).reshape((-1, 1))
B = np.array(input("Enter the values of dependent variable (B): ").split(), dtype=float)

model = LinearRegression()

model.fit(A, B)

A_new = np.array(input("Enter the value of new data point: "), dtype=float).reshape((-1, 1))

B_pred = model.predict(A_new)

print("Predicted value: ", B_pred[0])



# Multiple Linear Regression

import numpy as np
from sklearn.linear_model import LinearRegression

num_data_points = int(input("Enter the number of data points: "))
num_independent_variables = int(input("Enter the number of independent variables: "))

A = np.empty((num_data_points, num_independent_variables))
B = np.empty(num_data_points)

print("Enter the values for independent variables (A): ")
for i in range(num_data_points):
    values = input().split()
    A[i] = [float(value) for value in values]

print("Enter the values for dependent variable (B): ")
values = input().split()
B = [float(value) for value in values]

model = LinearRegression()

model.fit(A, B)

print("Enter the values for new data point: ")
new_data = input().split()
A_new = np.array([float(value) for value in new_data]).reshape((1, -1))

B_pred = model.predict(A_new)

print("Predicted value: ", B_pred[0])



# Polynomial Regression

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

num_data_points = int(input("Enter the number of data points: "))
degree = int(input("Enter the degree of polynomial: "))

A = np.empty((num_data_points, 1))
B = np.empty(num_data_points)

print("Enter the values for independent variable (A): ")
for i in range(num_data_points):
    A[i] = float(input())

print("Enter the values for dependent variable (B): ")
for i in range(num_data_points):
    B[i] = float(input())

poly_features = PolynomialFeatures(degree=degree)
A_poly = poly_features.fit_transform(A)

model = LinearRegression()

model.fit(A_poly, B)

new_data = float(input("Enter the value for new data point: "))
A_new = np.array([[new_data]])

A_new_poly = poly_features.transform(A_new)

B_pred = model.predict(A_new_poly)

print("Predicted value: ", B_pred[0])



# Logical Regression

import numpy as np
from sklearn.linear_model import LogisticRegression

num_data_points = int(input("Enter the number of data points: "))
num_features = int(input("Enter the number of features: "))

A = np.empty((num_data_points, num_features))
B = np.empty(num_data_points)

print("Enter the values for features (A): ")
for i in range(num_data_points):
    values = input().split()
    A[i] = [float(value) for value in values]

print("Enter the values for target variable (B): ")
values = input().split()
B = [int(value) for value in values]

model = LogisticRegression()

model.fit(A, B)

print("Enter the values for new data point: ")
new_data = input().split()
A_new = np.array([float(value) for value in new_data]).reshape((1, -1))

B_pred = model.predict(X_new)

print("Predicted value:", B_pred[0])
