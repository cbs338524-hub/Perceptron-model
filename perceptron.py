import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(len(X)):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0 else 0

                update = self.lr * (y[i] - y_predicted)
                self.weights += update * X[i]
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return [1 if i >= 0 else 0 for i in linear_output]

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,0,0,1])

model = Perceptron(learning_rate=0.1, epochs=10)
model.fit(X, y)

predictions = model.predict(X)

print("Predictions:", predictions)
X = np.array([
[0,0],
[0,1],
[1,0],
[1,1]
])

y = np.array([0,1,1,1])

model = Perceptron()
model.fit(X,y)

print(model.predict(X))
accuracy = np.sum(y == model.predict(X)) / len(y)
print("Accuracy:", accuracy)
import matplotlib.pyplot as plt
# Updated by Chethan

plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Dataset Visualization")
plt.show()
## Team Members
- Akhil Adapur (GitHub: Akhil200626)
- Chethan BS (GitHub: cbs338524)
- Abdul Nafi (GitHub: Abdul-Nafey-11)
- Al Mohammad Areez (GitHub: AlMohammadAreez)
