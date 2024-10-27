from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pandas as pd

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

data = pd.read_csv("../data/lucas_r.csv")

idx = [i for i in range(100,4100,200)]

X = data.iloc[:, idx]
y = data.iloc[:, -1]

model = SVR()
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

print(grid_search.best_params_)
print(grid_search.best_score_)
