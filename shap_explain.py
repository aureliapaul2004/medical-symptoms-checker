import pandas as pd
import joblib
import shap

df = pd.read_csv("training_data.csv")
df = df.drop(columns=["Unnamed: 133"])

X = df.drop("prognosis", axis=1)

model = joblib.load("disease_model.pkl")

sample = X.iloc[[0]]

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(sample)

print(type(shap_values))

