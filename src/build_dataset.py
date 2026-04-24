import pandas as pd

files = [
    ("set1_good.csv", "good"),
    ("set2_shallow.csv", "shallow"),
    ("set3_upright.csv", "upright"),
    ("set4_good.csv", "good"),
    ("set6_shallow.csv", "shallow"),
    ("set7_shallow.csv", "shallow"),
    ("set8_good.csv", "good"),
    ("set10_upright.csv", "upright"),
    ("set11_good.csv", "good"),
    ("set12_upright.csv", "upright"),
    ("set13_shallow.csv", "shallow"),
    ("set14_diffA_good.csv", "good"),
    ("set15_diffA_shallow.csv", "shallow"),
    ("set16_diffA_upright.csv", "upright"),
    ("set17_diffA_good.csv", "good"),
    ("set18_diffA_good.csv", "good"),
    ("set19_diffA_shallow.csv", "shallow"),
    ("set20_diffA_upright.csv", "upright"),
    ("madison_good.csv", "good"),
    ("madison_shallow.csv", "shallow"),
    ("madison_upright.csv", "upright"),
    ("madison2_good.csv", "good"),
    ("nicole_good.csv", "good"),
    ("nicole_shallow.csv", "shallow"),
    ("nicole2_good.csv", "good"),

]

dfs = []

for file, label in files:
    df = pd.read_csv(f"data/accumulatedData/{file}")
    df["label"] = label

    dfs.append(df)


dataset = pd.concat(dfs, ignore_index = True)
dataset.to_csv("data/dataset.csv", index=False)

print("Dataset created successfully")
print(dataset.head())