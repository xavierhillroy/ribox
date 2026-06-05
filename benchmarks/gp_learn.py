from gplearn.genetic import SymbolicRegressor
import pandas as pd, numpy as np

df = pd.read_csv("nguyen1.csv")
X, y = df[["x"]].values, df["y"].values

scores = []
for run in range(30):
    est = SymbolicRegressor(population_size=1000, generations=20,
                            function_set=('add','sub','mul'),
                            random_state=run)
    est.fit(X, y)
    scores.append(np.mean((y - est.predict(X))**2))  # MSE, same metric as d1
print("gplearn median MSE:", np.median(scores))