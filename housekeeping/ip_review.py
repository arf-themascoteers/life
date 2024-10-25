import pandas as pd


df = pd.read_csv("../data/indian_pines.csv")
unique_values_with_counts = df['class'].value_counts()
print(unique_values_with_counts)
