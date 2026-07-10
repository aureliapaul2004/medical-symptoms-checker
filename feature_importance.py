import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("training_data.csv")
df = df.drop(columns=["Unnamed: 133"])

X = df.drop("prognosis", axis=1)

# Load model
model = joblib.load("disease_model.pkl")

# Take 100 samples for speed
sample = X.sample(100, random_state=42)

# Create explainer
explainer = shap.TreeExplainer(model)

# Get SHAP values
shap_values = explainer.shap_values(sample)

# Handle multiclass output
if isinstance(shap_values, list):
    vals = np.mean([np.abs(v).mean(axis=0) for v in shap_values], axis=0)
else:
    vals = np.abs(shap_values).mean(axis=(0, 2))

# Create importance dataframe
importance = pd.DataFrame({
    "Feature": sample.columns,
    "Importance": vals
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(20)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance["Feature"], importance["Importance"])
plt.gca().invert_yaxis()

plt.title("Top 20 Important Symptoms")
plt.tight_layout()

plt.savefig("feature_importance.png")

print("feature_importance.png saved")
