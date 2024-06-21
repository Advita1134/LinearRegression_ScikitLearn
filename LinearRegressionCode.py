"""
Linear Regression

This code is supervised learning, so it uses given data to create to targets for new data.

First, it makes a line that is as close as possible to the data points. 
Then, it uses that line when a new feature is given.
It finds that feature on the line in the x-axis and uses the line to find the target on the y-axis.
"""


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Data
X = [[3], [5], [8], [9], [6], [2]]
y = [76, 67, 105, 93, 80, 65]

X_train,X_test,y_train,y_test = train_test_split(X,y)

# Model
model = LinearRegression()

# Training
model.fit(X_train,y_train)

m = model.coef_[0] # indexing into a numpy array
b = model.intercept_
#print(m)
#print(b)

# Testing
y_pred = model.predict(X_test)

plt.scatter(X_train,y_train, s = 40, c = "blue", label = "Training Data")
plt.scatter(X_test,y_pred, s = 100, c = "red", marker = "*", label = "Testing Data")

## Marking the Line
xforplot = max(X)[0] # X was a list of lists, so it gave an answer as a list. We had to take the first value of the list so that the answer will be an integer instead of another list.
yforplot = xforplot*m + b
plt.plot([0,xforplot],[b,yforplot])

plt.legend()
plt.axis("equal");# To make the axis equal sizes. ";" prevents it from printing unnecessary numbers.
