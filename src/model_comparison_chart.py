import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("model_comparison_results.csv")
plt.figure(figsize=(8,5))

plt.bar(df["Model"], df["F1 Score"])

plt.ylabel("F1 Score")
plt.title("Model Performance Comparison")

plt.ylim(0,1)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("model_comparison_chart.png", dpi=300)

plt.show()