import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/dataset.csv")
print(df.head())

X = df.drop(columns=["label"])
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify = y
)

#preprocessing --> chat helped with this
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter = 5000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

#changed it to the stuff above
#model = LogisticRegression(max_iter = 1000)
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
#chat told me to do this line for precision, recall, f1score and accuracy
print(classification_report(y_test, y_pred))