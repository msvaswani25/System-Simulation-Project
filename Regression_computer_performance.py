# -*- coding: utf-8 -*-
"""
@author: MS Vaswani
"""
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.cross_validation
import sklearn.metrics
from sklearn.linear_model import LinearRegression

#Loading data
input_file = "machine_data.csv"
df = pd.read_csv(input_file, header = 0, delimiter=",")



#Organizing Data
X=df.drop('PRP',axis=1)

#Initializing linear regression model
lm=LinearRegression()

#training model
lm.fit(X,df.PRP)

#trained model details
print('Estimeted intercept coefficient: ',lm.intercept_)
print('Number of coefficients: ',len(lm.coef_))


#relation between cache and performance (Origional)
plt.scatter(df.CACH,df.PRP)
plt.xlabel("Cache Memory(CACH)")
plt.ylabel('Published relative performance (PRP)')
plt.title("Relationship between CACH and PRP")
plt.show()

#relation between cache and performance (Predicted)
plt.scatter(df.CACH,lm.predict(X))
plt.xlabel("Cache: $Y_i$")
plt.ylabel('Predicted performance: $\hat{Y}_i$')
plt.title("Cache vs Predicted Performance: $Y_i$ vs $\hat{Y}_i$")
plt.show()

#R^2 (coefficient of determination) regression score function.
r2 = sklearn.metrics.r2_score(df.PRP, lm.predict(X))
print("R^2 Score: ", r2)





