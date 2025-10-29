import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Example data: study hours vs test scores
hours = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
scores = np.array([35, 40, 50, 55, 65, 70])

# Create and train the model
model = LinearRegression()
model.fit(hours, scores)

# Make a prediction
predicted_score = model.predict([[7]])
print(f"Predicted score for 7 study hours: {predicted_score[0]:.2f}")

# Plot the data and prediction line
plt.scatter(hours, scores, color="blue", label="Actual data")
plt.plot(hours, model.predict(hours), color="red", label="Regression line")
plt.xlabel("Hours Studied")
plt.ylabel("Test Score")
plt.legend()
plt.show()
