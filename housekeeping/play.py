import pandas as pd

df = pd.read_csv("../data_raw/indian_pines.csv")
print(df.iloc[0,-2:])