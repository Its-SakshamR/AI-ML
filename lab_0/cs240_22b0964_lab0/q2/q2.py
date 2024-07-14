import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None
    final_data = None
    dimensions = 2

    # TODO: transform init_array to final_data using PCA
    data = pd.read_csv("pca_data.csv")

    standard_data = data-np.mean(data, axis=0)

    eigenvalues, eigenvectors = np.linalg.eig(np.cov(standard_data.T))  # calculating covariance matrix and then 

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:,sorted_indices]

    required_eigenvectors = sorted_eigenvectors[:, :2]

    transformed_data = np.dot(standard_data, required_eigenvectors)

    final_data = pd.DataFrame(transformed_data, columns = ['C1','C2'])
    final_data.to_csv('transform.csv', index = False)
    # END TODO

    return np.round(sorted_eigenvalues,4), final_data


if __name__ == '__main__':
    init_array = pd.read_csv("pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)

    # TODO: plot and save a scatter plot of final_data to out.png
    plt.scatter(final_data['C1'],final_data['C2'])
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.show()
    # END TODO
