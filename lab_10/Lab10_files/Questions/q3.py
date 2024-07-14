import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = []

    def initialise(self, X_train):
        """
        Initialize the self.centroids class variable, using the "k-means++" method, 
        Pick a random data point as the first centroid,
        Pick the next centroids with probability directly proportional to their distance from the closest centroid
        Function returns self.centroids as an np.array
        USE np.random for any random number generation that you may require 
        (Generate no more than K random numbers). 
        Do NOT use the random module at ALL!
        """
        # TODO
        n_samples, n_features = X_train.shape
        centroids = np.zeros((self.n_clusters, n_features))
        centroids[0] = X_train[np.random.randint(0, n_samples)]

        for k in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(X_train - centroid, axis=1)**2 for centroid in centroids[:k]], axis=0)
            probabilities = distances / np.sum(distances)
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            for i, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[k] = X_train[i]
                    break
        
        self.centroids = centroids
        return self.centroids
        # END TODO
    def fit(self, X_train):
        """
        Updates the self.centroids class variable using the two-step iterative algorithm on the X_train dataset.
        X_train has dimensions (N,d) where N is the number of samples and each point belongs to d dimensions
        Ensure that the total number of iterations does not exceed self.max_iter
        Function returns self.centroids as an np array
        """
        # TODO
        for _ in range(self.max_iter):
            distances = np.array([np.linalg.norm(X_train - centroid, axis=1) for centroid in self.centroids])
            classification = np.argmin(distances, axis=0)
            new_centroids = np.array([X_train[classification == k].mean(axis=0) for k in range(self.n_clusters)])

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return self.centroids
        # END TODO
    
    def evaluate(self, X):
        """
        Given N data samples in X, find the c   luster that each point belongs to 
        using the self.centroids class variable as the centroids.
        Return two np arrays, the first being self.centroids 
        and the second is an array having length equal to the number of data points 
        and each entry being between 0 and K-1 (both inclusive) where K is number of clusters.
        """
        # TODO
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids])
        classification = np.argmin(distances, axis=0)

        return self.centroids, classification
        # END TODO

def evaluate_loss(X, centroids, classification):
    loss = 0
    for idx, point in enumerate(X):
        loss += np.linalg.norm(point - centroids[classification[idx]])
    return loss
