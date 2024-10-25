import pandas as pd


df = pd.read_csv("../data/lucas_lc0_r.csv")
unique_values_with_counts = df['lc'].value_counts()
print(unique_values_with_counts)
