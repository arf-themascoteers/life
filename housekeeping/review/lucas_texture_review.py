import pandas as pd


df = pd.read_csv("../../data/lucas_texture_r.csv")
unique_values_with_counts = df['texture'].value_counts()
print(unique_values_with_counts)
distinct_count = df['texture'].nunique()
print(distinct_count)
