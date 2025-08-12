import joblib
import pandas as pd
import numpy as np

# 📌 Load trained model and label binarizer
model = joblib.load("blood_model.pkl")
mlb = joblib.load("label_binarizer.pkl")

# 📌 Define expected features
expected_features = ["Age", "Gender", "Height", "Weight", "Systolic_BP", "Diastolic_BP", 
                     "Cholesterol", "Glucose", "Smoking", "Alcohol", "Physical_Activity"]

print("Please input the following patient details:")
try:
    Age = float(input("Age (in years): "))
    Gender = int(input("Gender (1 for Male, 2 for Female): "))
    Height = float(input("Height (in cm): "))
    Weight = float(input("Weight (in kg): "))
    Systolic_BP = float(input("Systolic Blood Pressure: "))
    Diastolic_BP = float(input("Diastolic Blood Pressure: "))
    Cholesterol = float(input("Cholesterol (mg/dL): "))
    Glucose = float(input("Glucose (mg/dL): "))
    Smoking = int(input("Smoking (0 for Non-smoker, 1 for Smoker): "))
    Alcohol = int(input("Alcohol (0 for Non-drinker, 1 for Drinker): "))
    Physical_Activity = int(input("Physical Activity (0 for Inactive, 1 for Active): "))
except Exception as e:
    print("❌ Error in input. Please enter numeric values correctly.")
    exit(1)

# 📌 Create input dictionary
input_data = {
    "Age": Age,
    "Gender": Gender,
    "Height": Height,
    "Weight": Weight,
    "Systolic_BP": Systolic_BP,
    "Diastolic_BP": Diastolic_BP,
    "Cholesterol": Cholesterol,
    "Glucose": Glucose,
    "Smoking": Smoking,
    "Alcohol": Alcohol,
    "Physical_Activity": Physical_Activity
}

# 📌 Convert to DataFrame
input_df = pd.DataFrame([input_data], columns=expected_features)

# 📌 Predict probabilities
probabilities = model.predict_proba(input_df)

# 📌 Set threshold (default 0.3 or 30%)
threshold = 0.3
predicted_labels = (probabilities >= threshold).astype(int)

# 📌 Convert predictions back to disease names
diseases = mlb.inverse_transform(predicted_labels)[0]

# 📌 Print Prediction
print("\n🩺 **Predicted Conditions:**", ", ".join(diseases) if diseases else "No Disease Detected")
