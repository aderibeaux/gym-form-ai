import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/dataset.csv")
#chat told me to use grid search and helped with debugging
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

rf = RandomForestClassifier(random_state = 42)

param_grid = {
   "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 4, 6] 
}

grid_search = GridSearchCV(
    estimator = rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X,y)

print("Best parameters:", grid_search.best_params_)
print("best cross-validation accuracy:", grid_search.best_score_)

