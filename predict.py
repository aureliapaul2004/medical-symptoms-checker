import joblib
import pandas as pd

model = joblib.load("disease_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# 132 symptoms
symptoms = [0] * 132

# Example: itching and skin rash
symptoms[0] = 1
symptoms[1] = 1

prediction = model.predict([symptoms])

disease = encoder.inverse_transform(prediction)

print("Predicted Disease:", disease[0])
