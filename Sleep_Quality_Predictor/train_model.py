import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("sleep_data.csv")

# Convert Gender to numbers
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Encode target column
encoder = LabelEncoder()
df["Quality"] = encoder.fit_transform(df["Quality"])

# Features and target
X = df[["Age", "Gender", "Sleep_Duration", "Stress_Level", "Physical_Activity", "BMI"]]
y = df["Quality"]

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model and encoder
pickle.dump(model, open("sleep_model.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))

print(" Model trained and saved successfully")
