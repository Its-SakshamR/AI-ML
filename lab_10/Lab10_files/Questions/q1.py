import pickle as pkl
import numpy as np

def pca(X: np.array, k: int) -> np.array:
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    Return (a*b,k) np array comprising the k normalised basis vectors comprising the k-dimensional subspace for all images
    where the first column must be the most relevant principal component and so on
    """
    # TODO
    N, a, b = X.shape
    X_flat = X.reshape(N, -1)  # Reshape each image to a single vector
    
    # Mean centering
    mean_vec = np.mean(X_flat, axis=0)
    X_centered = X_flat - mean_vec
    
    # Covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Eigen decomposition
    eig_values, eig_vectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    
    # Select the top k eigenvectors
    top_k_eig_vectors = eig_vectors[:, :k]
    
    # Normalize the basis vectors
    normalized_basis = top_k_eig_vectors / np.linalg.norm(top_k_eig_vectors, axis=0)
    
    return normalized_basis
    #END TODO
    

def projection(X: np.array, basis: np.array):
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    basis is an (a*b,k) array comprising of k normalised vectors
    Return (n,k) np array comprising the k dimensional projections of the N images on the normalised basis vectors
    """
    # TODO
    N, a, b = X.shape
    X_flat = X.reshape(N, -1)  # Reshape each image to a single vector
    
    # Projection
    projections = np.dot(X_flat, basis)
    
    return projections
    # END TODO
    