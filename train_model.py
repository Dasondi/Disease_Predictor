import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score

# 📌 Load dataset
dataset_path = "cardio_train_with_disease_prediction.csv"
df = pd.read_csv(dataset_path, delimiter=";")

# 📌 Rename columns for consistency
df.rename(columns={
    "age": "Age",
    "gender": "Gender",
    "height": "Height",
    "weight": "Weight",
    "ap_hi": "Systolic_BP",
    "ap_lo": "Diastolic_BP",
    "cholesterol": "Cholesterol",
    "gluc": "Glucose",
    "smoke": "Smoking",
    "alco": "Alcohol",
    "active": "Physical_Activity",
    "disease_prediction": "Condition"  # Target disease labels
}, inplace=True)

# 📌 Remove unnecessary columns
df.drop(columns=["id", "cardio"], errors="ignore", inplace=True)  # Removed "cardio" (Cardio Risk Level)

# 📌 Convert Age from days to years
df["Age"] = (df["Age"] / 365).astype(int)

# 📌 Drop duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# 📌 Convert multiple diseases into separate columns (Multi-Label Classification)
mlb = MultiLabelBinarizer()
df["Condition"] = df["Condition"].apply(lambda x: x.split(", "))  # Convert string to list
y = mlb.fit_transform(df["Condition"])  # Convert diseases into binary format

# 📌 Define features (X) and target (y)
X = df.drop(columns=["Condition"])

# 📌 Handle class imbalance using undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# 📌 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 📌 Train Multi-Output Model with OneVsRestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
model = OneVsRestClassifier(rf).fit(X_train, y_train)

# 📌 Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Training Complete! Accuracy: {accuracy:.2f}")

# 📌 Save trained model & label binarizer
joblib.dump(model, "blood_model.pkl")
joblib.dump(mlb, "label_binarizer.pkl")
print("💾 Model saved as 'blood_model.pkl' and 'label_binarizer.pkl'")
