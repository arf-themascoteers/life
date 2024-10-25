import pandas as pd


df = pd.read_csv("../data/ghisaconus.csv")
unique_values_with_counts = df['crop'].value_counts()
print(unique_values_with_counts)
