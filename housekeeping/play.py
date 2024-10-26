import pandas as pd

data = {
    'a': [1, 1, 2, 2, 1,],
    'b': [2, 2, 3, 3, 2],
    'c': [3, 3, 4, 4, 30],
    'd': [10, 20, 30, 40, 50],
    'e': [15, 25, 35, 45, 55],
    'f': ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
}
df = pd.DataFrame(data)

result_df = df.groupby(['a', 'b', 'c']).agg(
    {'d': 'mean', 'e': 'mean', 'f': lambda x: '---'.join(x)}
).reset_index()

print(result_df)
