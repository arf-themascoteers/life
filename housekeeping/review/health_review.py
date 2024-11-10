import pandas as pd


df = pd.read_csv("../../data_raw/ghisaconus_health.csv")
unique_values_with_counts = df['health'].value_counts()
print(unique_values_with_counts)
print(len(df.columns))
print(len(df))
