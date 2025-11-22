import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

file = pd.read_csv(r"C:\Users\dharu\OneDrive\Desktop\Data Science Bootcamp\finance_economics_dataset.csv")
test_features = ["Trading Volume","Government Debt (Billion USD)","Open Price","Daily High"]
X = file[test_features]
y = file["Close Price"]
stan = StandardScaler()
stan_X = stan.fit_transform(X)
lin_model = LinearRegression()
lin_model.fit(stan_X,y)