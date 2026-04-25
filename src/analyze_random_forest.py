import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth= None,
    min_samples_split=2,
    random_state=42
    )

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.savefig("docs/confusion_matrix.png")
plt.show()

importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature Importances:")
print(importance_df)

plt.figure(figsize=(8, 5))
plt.bar(importance_df["feature"], importance_df["importance"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("docs/feature_importance.png")
plt.show()


joblib.dump(rf_model, "models/squat_model.pkl")
print("Saved model as squat_model.pkl")

#The confusion matrix shows that the model 
#occasionally confuses good reps with shallow reps. 
#This probably happens when the squat depth is close 
#to the threshold between the two classes, making the 
#joint angle features similar (one of the most challenging inputs). Additionally, one shallow 
#rep was misclassified as upright, which suggests that both problems
#could have been occuring at the same time because it is not hard
#to do both errors at the same time. For example, a shallow squat 
#may also have an upright torso, making it difficult for 
#the classifier to distinguish between these categories.
