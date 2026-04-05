from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# =========================
# LOAD & TRAIN MODEL
# =========================

train_data = pd.read_csv("Training.csv")

if 'Unnamed: 133' in train_data.columns:
    train_data = train_data.drop('Unnamed: 133', axis=1)

X = train_data.drop("prognosis", axis=1)
y = train_data["prognosis"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Get all symptoms
symptoms_list = list(X.columns)

# =========================
# PREDICTION FUNCTION
# =========================

def predict_from_symptoms(symptom_names):
    input_data = [0] * len(X.columns)

    for symptom in symptom_names:
        if symptom in X.columns:
            index = list(X.columns).index(symptom)
            input_data[index] = 1

    prediction = model.predict([input_data])
    return le.inverse_transform(prediction)[0]

# =========================
# ROUTES
# =========================

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify(symptoms_list)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    result = predict_from_symptoms(symptoms)

    return jsonify({"disease": result})

# =========================
# RUN SERVER
# =========================

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)