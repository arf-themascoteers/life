import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("saved_results/p10/p10_summary.csv")

scaler = MinMaxScaler()
df[['oa', 'time']] = scaler.fit_transform(df[['oa', 'time']])

plt.figure(figsize=(10, 6))
plt.plot(df['props'], df['oa'], label='OA')
plt.plot(df['props'], df['time'], label='Time')

plt.xlabel('Props')
plt.legend()
plt.show()
