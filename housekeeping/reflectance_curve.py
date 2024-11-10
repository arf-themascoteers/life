import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'../data_raw/ghisaconus_health.csv')
df_aggregated = df.groupby('health').mean()
cols = df.columns
cols = [col for col in df.columns if col != "health"]
i_cols = [int(s[1:]) for s in cols]

for row in df_aggregated.itertuples():
    val = row[0]
    ref = row[1:]
    plt.plot(i_cols, ref, label=val.replace("_", " ").title())

plt.legend()

plt.xlabel("Wavelength (nm)", fontsize=15)
plt.ylabel("Reflectance (%)", fontsize=15)
plt.legend(fontsize=13)
plt.margins(0.01)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("ghisaconus_health.png")
