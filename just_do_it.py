import pandas as pd

df = pd.read_csv("saved_results/m1/m1_summary.csv")
df = df[df["target_size"]<31]
df.to_csv("m2.csv", index=False)