from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Prepare data
X = np.array([3, 5, 7, 9, 11]).reshape(-1, 1)
Y = np.array([40, 55, 65, 80, 95])

# 2. Create model
model = LinearRegression()

# 3. Train model
model.fit(X, Y)

# 4. Predict new value
predicted = model.predict([[12]])
print(predicted)

# 5. See the slope and intercept
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# 6. (Optional) See how good the model is
print("Model score (R²):", model.score(X, Y))

