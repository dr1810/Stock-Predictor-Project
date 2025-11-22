import streamlit as st
import pandas as pd
import numpy as np
import plotly
import matplotlib.pyplot as plt
import sklearn.linear_model
from model import lin_model,stan
from sklearn.preprocessing import StandardScaler
st.title("Stock Price Predictor")
trade_volume = st.slider("Select the Trade Volume")
debt = st.number_input("Enter the government debt")
open = st.number_input("Enter Open Price")
daily = st.slider("Select Daily High")
pred = np.array([trade_volume,debt,open,daily])
new = stan.transform(pred.reshape(1,-1))
st.write(f"The predicted stock value is {float(lin_model.predict(new))}")