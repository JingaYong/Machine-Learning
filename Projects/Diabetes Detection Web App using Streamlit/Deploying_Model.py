# Import Packages

import streamlit as sl
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Loading the saved model

#rb- read binary
loaded_model = pickle.load(open("trained_model.sav", 'rb'))

# Prediction Function
def diabetes_prediction(input_data):
    sc = StandardScaler()
    ip = np.asarray(input_data)
    ip = ip.reshape(1, -1)
    ip = sc.fit_transform(ip)
    pred = loaded_model.predict(ip)
    # print(pred)

    if pred[0] == 0:
        return "Person is not diabetic"
    else:
        return "Person is diabetic"

# Web App GUI
def main():
    # Giving title
    sl.title("Diabetes Prediction Web Application")

    # Getting input
    Pregnancies = sl.text_input("Number of pregnancies:")
    Glucose = sl.text_input("Glucose level:")
    BloodPressure = sl.text_input("Blood Pressure value:")
    SkinThickness = sl.text_input("Skin Thickness value:")
    Insulin = sl.text_input("Insulin level:")
    BMI = sl.text_input("BMI value:")
    DiabetesPedigreeFunction = sl.text_input("Diabetes Pedigree Function value:")
    Age = sl.text_input("Age of the person:")

    diagnosis = ''

    # create a button
    if sl.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                         DiabetesPedigreeFunction, Age])

    sl.success(diagnosis)
    
# Main Method
if __name__ == "__main__":
    main()

# To run file - 
# open terminal -> streamlit run file_name.py
