# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:38:59 2024

@author: Sravan
"""

import streamlit as st
import pickle
import numpy as np

model=pickle.load(open(r'C:\Users\admin\Streamlit Demo\linear_regression_model_spyder.pkl', 'rb'))

st.title('Salary Prediction App')
st.write("This app predicts salary baed on yrs of Exp using a simple liner regression model")

yrs_Exp = st.number_input("Enter yrs of exp:", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

if st.button("Predict Salary"):
    # Make a prediction using the trained model
    experience_input = np.array([[yrs_Exp]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(experience_input)
   
    # Display the result
    st.success(f"The predicted salary for {yrs_Exp} years of experience is: ${prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model was trained using a dataset of salaries and years of experience.")