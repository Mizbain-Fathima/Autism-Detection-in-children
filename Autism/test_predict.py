import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Path to honey purity model file
model_path = r'E:\ML_projects\Autism\autism_model.pkl'

# Input features 
test_input = {
    'A1_Score': 1,
    'A2_Score': 0,
    'A3_Score': 1,
    'A4_Score': 0,
    'A5_Score': 1,
    'A6_Score': 0,
    'A7_Score': 1,
    'A8_Score': 0,
    'A9_Score': 1,
    'A10_Score': 0,
    'Age_Mons': 36,
    'Qchat-10-Score': 8,
    'Sex': 1,                   
    'Ethnicity': 2,             
    'Jaundice': 0,
    'Family_mem_with_ASD': 1,
    'Who completed the test': 3 
}


def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model(model_path)
test_df = pd.DataFrame([test_input])

scaler = StandardScaler()
test_df_scaled = scaler.fit_transform(test_df)
prediction = model.predict(test_df_scaled)
result = "ASD Traits Detected" if prediction[0] == 1 else "No ASD Traits Detected"
print("Prediction:", result)