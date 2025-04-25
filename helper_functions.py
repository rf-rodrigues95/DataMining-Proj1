import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def center_(x, cluster):
    return [np.mean(x[cluster, j]) for j in range(x.shape[1])]

# Used to group dataPoints in clusters
def distNorm(x ,remains, ranges, p):
    z = x[remains, :]
    az = np.tile(p, (len(remains), 1))
    rz = np.tile(ranges, (len(remains), 1))
    dz = (z - az) / rz
    return np.sum(dz * dz, axis=1)

# To See if it belongs to Cluster X?
def separCluster(x, remains, ranges, a, b):
    dista = distNorm(x, remains, ranges, a)
    distb = distNorm(x, remains, ranges, b)
    return [remains[i] for i in np.where(dista < distb)[0]]

def anomalousPattern(x, remains, ranges, centroid, me):
    while True:
        cluster = separCluster(x, remains, ranges, centroid, me)
        if not cluster:
            break
        newcenter = center_(x, cluster)
        if np.allclose(centroid, newcenter):
            break
        centroid = newcenter
    return cluster, centroid

def dist(x, remains, ranges, p):
    return np.sum(((x[remains] - p) / ranges) ** 2, axis=1)

# Optimized Vectorized implementation
# Validation / Criterion
def xie_beni_index(U, centers, X):
    um = U ** 2
    dist_sq = np.sum((X[np.newaxis, :, :] - centers[:, np.newaxis, :]) ** 2, axis=2)
    compactness = np.sum(um * dist_sq)

    center_dist_sq = np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(center_dist_sq, np.inf)
    min_center_dist_sq = np.min(center_dist_sq)

    if min_center_dist_sq == 0:
        return np.inf
    return compactness / (X.shape[0] * min_center_dist_sq)

def pca_application(x_norm, y_values, init_centroids = None):
    # Covariance matrix of the normalized dataset
    covmatrix = np.cov(x_norm.T)

    # Eigenvalues and eigenvectors of the covariance matrix
    e, v = np.linalg.eig(covmatrix)

    # Order eigenvalues and eigenvectors in descending order
    order = np.argsort(e)[::-1]  # Sort eigenvalues in descending order
    e = e[order]
    v = v[:, order]

    # Print eigenvectors and eigenvalues
    #print("Eigenvectors:\n", v)
    print("\nEigenvalues:\n", e)

    # Generate PCA component space (PCA scores)
    pc = np.dot(x_norm, v)

    # Set data to a Pandas DataFrame for easier plotting
    names = ["PC_" + str(x + 1) for x in range(pc.shape[1])]
    names.append('target')
    data_pca = pd.DataFrame(data=np.c_[pc, y_values], columns=names)
    data_pca['target'] = data_pca['target'].astype(int)
    if init_centroids.any():
        init_centroids_pca = np.dot(init_centroids, v)

        return data_pca, init_centroids_pca
    else:
        return data_pca
    
def plot_clustering(x, labels, centers, title, score):
    plt.figure(figsize=(8, 6))
    for j in np.unique(labels):
        plt.scatter(x[labels == j, 0], x[labels == j, 1], label=f'Cluster {j+1}')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X', label='Centers')
    plt.title(f"{title} (Score={score:.4f})")
    plt.legend()
    plt.grid(True)
    plt.show()