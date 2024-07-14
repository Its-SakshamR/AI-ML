import numpy as np
import pickle as pkl

class LDA:
    def __init__(self,k):
        self.n_components = k
        self.linear_discriminants = None

    def fit(self, X, y):
        """
        X: (n,d,d) array consisting of input features
        y: (n,) array consisting of labels
        return: Linear Discriminant np.array of size (d*d,k)
        """
        # TODO
        n, d, _ = X.shape
        d_2 = d ** 2
        X = X.reshape(n, d_2)
        overall_mean = np.mean(X, axis=0)
        
        Sb = np.zeros((d_2, d_2))
        Sw = np.zeros((d_2, d_2))
        
        for class_value in np.unique(y):
            X_class = X[y == class_value]
            class_mean = np.mean(X_class, axis=0)
            Sw += np.dot((X_class - class_mean).T, (X_class - class_mean))
            class_mean_diff = class_mean - overall_mean
            Sb += X_class.shape[0] * np.outer(class_mean_diff, class_mean_diff)
        
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[:, sorted_indices][:, :self.n_components]
        
        self.linear_discriminants = eigenvectors
        return self.linear_discriminants           # Modify as required
        #END TODO 
    
    def transform(self, X, w):
        """
        w:Linear Discriminant array of size (d*d,1)
        return: np-array of the projected features of size (n,k)
        """
        # TODO
        X_flat = X.reshape(X.shape[0], -1)
        projected=np.dot(X_flat, w)     # Modify as required
        return projected                   # Modify as required
        # END TODO
