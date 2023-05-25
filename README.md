# Regression
 Simples Linear Regression, Multiple Simple Linear Regression, Polynomial Regression, Logistic Regression



# SIMPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression

# Independent variable (A)
A = []
num_samples = int(input("Enter the number of samples: "))
print("Enter the values for A:")
for _ in range(num_samples):
    A_value = float(input())
    A.append([A_value])

# Dependent variable (B)
B = []
print("Enter the values for B:")
for _ in range(num_samples):
    B_value = float(input())
    B.append(B_value)

# Linear Regression Model
model = LinearRegression()

# Model
model.fit(A, B)

# Predictions
A_test = []
num_test_samples = int(input("Enter the number of test samples: "))
print("Enter the test values for A:")
for _ in range(num_test_samples):
    A_test_value = float(input())
    A_test.append([A_test_value])

predictions = model.predict(A_test)

# Print coefficients and intercept
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

# Print predictions
print("Predictions: ", predictions)



# MULTIPLE lINEAR REGRESSION

from sklearn.linear_model import LinearRegression

# Independent variables (A)
A = []
num_samples = int(input("Enter the number of samples: "))
num_features = int(input("Enter the number of independent variables: "))

print("Enter the values for A:")
for _ in range(num_samples):
    A_values = []
    for _ in range(num_features):
        A_value = float(input())
        A_values.append(A_value)
    A.append(A_values)

# Dependent variable (B)
B = []
print("Enter the values for B:")
for _ in range(num_samples):
    B_value = float(input())
    B.append(y_value)



# Linear regression model
model = LinearRegression()

# Model
model.fit(A, B)

# Predictions
A_test = []
num_test_samples = int(input("Enter the number of test samples: "))
print("Enter the test values for A:")
for _ in range(num_test_samples):
    a_test_values = []
    for _ in range(num_features):
        a_test_value = float(input())
        a_test_values.append(a_test_value)
    A_test.append(a_test_values)

predictions = model.predict(A_test)

# Coefficients and Intercept
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

# Predictions
print("Predictions: ", predictions)




# POLYNOMIAL REGRESSION

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Independent variable (A)
A = []
num_samples = int(input("Enter the number of samples: "))
print("Enter the values for A:")
for _ in range(num_samples):
    A_value = float(input())
    A.append([A_value])

# Dependent variable (B)
B = []
print("Enter the values for B:")
for _ in range(num_samples):
    B_value = float(input())
    B.append(B_value)

# Degree of the polynomial
D = int(input("Enter the degree of the polynomial: "))

# Transform the input data to polynomial features
poly_features = PolynomialFeatures(degree=D)
A_poly = poly_features.fit_transform(A)

# Linear regression model
model = LinearRegression()

# Model
model.fit(A_poly, B)

# Predictions
A_test = []
num_test_samples = int(input("Enter the number of test samples: "))
print("Enter the test values for A:")
for _ in range(num_test_samples):
    a_test_value = float(input())
    A_test.append([a_test_value])

A_test_poly = poly_features.transform(A_test)
predictions = model.predict(A_test_poly)

# Coefficients and Intercept
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

# Predictions
print("Predictions: ", predictions)



# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

# Independent variables (A)
A = []
num_samples = int(input("Enter the number of samples: "))
num_features = int(input("Enter the number of independent variables: "))

print("Enter the values for A:")
for _ in range(num_samples):
    A_values = []
    for _ in range(num_features):
        A_value = float(input())
        A_values.append(A_value)
    A.append(x_values)

# Binary labels (B)
B = []
print("Enter the binary labels (0 or 1):")
for _ in range(num_samples):
    label = int(input())
    B.append(label)

# Logistic regression model
model = LogisticRegression()

# Model
model.fit(A, B)

# Predictions
A_test = []
num_test_samples = int(input("Enter the number of test samples: "))
print("Enter the test values for A:")
for _ in range(num_test_samples):
    a_test_values = []
    for _ in range(num_features):
        a_test_value = float(input())
        a_test_values.append(a_test_value)
    A_test.append(a_test_values)

predictions = model.predict(A_test)

# Coefficients and Intercept
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

# Predictions
print("Predictions: ", predictions)
