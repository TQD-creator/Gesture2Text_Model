# In file: 2_src/model.py
import numpy as np
from collections import Counter
import joblib

class MyModel:
    """
    A custom k-Nearest Neighbors (k-NN) model built
    from scratch using only Numpy.
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        print(f"MyModel (k-NN from Scratch) initialized with k={self.k}.")

    def _euclidean_distance(self, p1, p2):
        # Helper function to calculate distance
        return np.sqrt(np.sum((p1 - p2)**2))

    def train(self, X, y):
        # k-NN is simple: "training" is just memorizing the data
        self.X_train = X
        # We must convert labels ('A', 'B'...) to numbers
        self.y_train, self.labels = pd.factorize(y)
        print("Training complete (data memorized).")
        
    def predict(self, X):
        predictions = [self._predict_one(x) for x in X]
        # Convert numeric predictions back to string labels
        return self.labels[predictions]
        
    def _predict_one(self, x_test):
        # 1. Calculate distance from x_test to ALL training points
        distances = [self._euclidean_distance(x_test, x_train) for x_train in self.X_train]
        
        # 2. Get the indices of the 'k' nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]
        
        # 3. Get the labels of those neighbors
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        
        # 4. Vote for the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0] # Return the most common label

    def save(self, path):
        # For k-NN, we save the "memorized" data
        joblib.dump((self.X_train, self.y_train, self.k, self.labels), path)
        
    def load(self, path):
        self.X_train, self.y_train, self.k, self.labels = joblib.load(path)