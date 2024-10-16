import pandas as pd

df = pd.read_csv("../data/lucas.csv")
df = df.sample(frac=0.01)
df.to_csv("../data/lucas_min.csv", index=False)