import pandas as pd
import numpy as np


df = pd.read_csv("../../data/lucas_texture_4_r.csv")
unique_values_with_counts = df['texture'].value_counts()
print(unique_values_with_counts)
distinct_count = df['texture'].nunique()
print(distinct_count)
print(len(df))
data = df.iloc[:,:-1].to_numpy()
print(np.min(data))
print(np.max(data))
