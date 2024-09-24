# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:38:14 2024

@author: Sravan
"""

import streamlit as st
import pickle
import numpy as np

model=pickle.load(open(r'C:\Users\admin\HouseRentPrediction\linear_regression_model_HouseRent.pkl', 'rb'))

st.title('House rent Prediction App')
st.write("This app predicts Rent baed on living room sqft using a simple liner regression model")

sqft = st.number_input("Enter area in SqFt:", min_value=500.00, max_value=5000.0, value=500.0, step=50.0)

if st.button("Predict Rent"):
    # Make a prediction using the trained model
    area_input = np.array([[sqft]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(area_input)
   
    # Display the result
    st.success(f"The predicted Rent for {sqft} sqft area is: ${prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model was trained using a dataset Living room area in Sqft and Rent range.")