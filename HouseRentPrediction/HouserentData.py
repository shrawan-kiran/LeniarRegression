# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:07:29 2024

@author: Sravan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv(r'C:\Users\admin\HouseRentPrediction\House_data.csv')

#print(dataset.isna().any().any())

space = dataset.sqft_living
price=dataset.price

x=np.array(space).reshape(-1,1)
y=np.array(price)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regrsr = LinearRegression()
regrsr.fit(xtrain, ytrain)

pred = regrsr.predict(xtest)

plt.scatter(xtrain, ytrain, color= 'green')
plt.plot(xtrain, regrsr.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()


plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regrsr.predict(xtrain), color = 'black')
plt.title ("Visuals for Test Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()


# Optional: Output the coefficients of the linear model
print(f"Intercept: {regrsr.intercept_}")
print(f"Coefficient: {regrsr.coef_}")

# Compare predicted and actual salaries from the test set
comparison = pd.DataFrame({'Actual': ytest, 'Predicted': pred})
print(comparison)

y_1250 = regrsr.predict([[1250.00]])
print(f"Predicted Rent for 1250 sqft is: ${y_1250[0]:,.2f}")

# Save the trained model to disk
filename = 'linear_regression_model_HouseRent.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regrsr, file)
print("Model has been pickled and saved as linear_regression_model_HouseRent.pkl")
import os
cwd = os.getcwd()