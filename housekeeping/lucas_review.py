import pandas as pd


df = pd.read_csv("../data/lucas_lc0_s_r.csv")
unique_values_with_counts = df['lc'].value_counts()
print(unique_values_with_counts)
distinct_count = df['lc'].nunique()
print(distinct_count)

df = pd.read_csv("../data/lucas_texture_r.csv")
unique_values_with_counts = df['lc'].value_counts()
print(unique_values_with_counts)
distinct_count = df['texture'].nunique()
print(distinct_count)

df = pd.read_csv("../data/lucas_r.csv")
print(df["oc"].min())
print(df["oc"].max())