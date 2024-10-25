import pandas as pd


df = pd.read_csv("../../data/lucas_r.csv")
print(df["oc"].min())
print(df["oc"].max())

