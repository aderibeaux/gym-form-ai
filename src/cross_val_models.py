import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("data/dataset.csv")

feature_cols = [
    "rep_duration_frames",
    "hip_range",
    "bottom_knee_angle",
    "bottom_hip_angle",
    "bottom_torso_angle",
    "bottom_shin_angle"
]

X = df[feature_cols]
y = df["label"]

logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

rf = RandomForestClassifier(n_estimators=100, random_state=42)

#5-fold cross validation
logreg_scores = cross_val_score(logreg, X, y, cv=5)
rf_scores = cross_val_score(rf, X, y, cv=5)

print("Logistic Regression CV Accuracy:", logreg_scores.mean())
print("Random Forest CV Accuracy:", rf_scores.mean())

#So Logistic Regression got 89.4% cross-validation accuracy, while Random forest
#got 93.9%. Since random forest has many decision trees, so it can learn more
#complex patterns, while logisitc just tries to make a straight boundary. This suggests
#that nonlinear relationships between the features such as hip angle, knee angle, and torso
#angle are important for predicting bulgarian split squat form quality