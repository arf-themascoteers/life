from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.svm import SVR

data = pd.read_csv("../data/lucas_r.csv")

idx = [i for i in range(100,4100,10)]

X = data.iloc[:, idx]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = SVR(C=100, kernel='rbf', gamma=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(r2, rmse)