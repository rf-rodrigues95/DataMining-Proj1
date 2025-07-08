# PAD-Project1: Data Analysis & Mining  
**NOVA University of Lisbon** – 2024–2025  
**Course:** Data Analysis and Mining  
**Final Grade:** 16.3  
**Group:** 
  - Ricardo Rodrigues (rf-rodrigues95)
  - Alberto Dicembre (Nsivaa)

## Overview

This project explores advanced techniques in data analysis and clustering, using both numerical and network data. The work includes preprocessing, visualization, clustering, and evaluation using internal and external metrics across multiple datasets and methods.

- **Part 1** focused on classical clustering methods using a numerical dataset (of our own choosing) with at least 10 numerical features. This part involved implementing and evaluating various clustering algorithms, performing EDA, dimensionality reduction with PCA, and interpreting clustering results both quantitatively and visually.

- **Part 2** centered on spectral clustering and community detection using provided network datasets. Unlike Part 1, which covered a broad range of clustering techniques, Part 2 was exclusively dedicated to applying and analyzing spectral methods on graph data, emphasizing parameter sensitivity and structural discovery in networks.

Key techniques used:

- **Exploratory Data Analysis (EDA):** Statistical summaries, correlation matrices, anomaly detection, and dimensionality insights.
- **Fuzzy Clustering:** Applied **Fuzzy C-Means (FCM)** and **Anomalous Pattern FCM**, analyzing clustering quality using metrics like Silhouette Score, Xie-Beni index, and Adjusted Rand Index.
- **Principal Component Analysis (PCA):** Used for dimensionality reduction and to visualize cluster structures in lower dimensions.
- **Spectral Clustering for Community Detection:** Implemented the algorithm by Ng, Jordan, and Weiss (2002) on benchmark network datasets (**PolBooks**, **Football**) to study the effect of hyperparameters (number of clusters `k`, Gaussian kernel width `σ`, and Laplacian type). Clustering performance was evaluated using **Normalized Mutual Information (NMI)** and **Modularity Score**.

The project provides an end-to-end analysis pipeline for clustering, combining both structured data mining and graph-based community detection. All methods are compared, visualized, and interpreted with respect to real-world metadata and theoretical expectations.

## Contents

- `./`: Jupyter notebooks with code and visualizations  
- `/data/`: Directory for datasets
- `/report/`: Contains the final written report, providing a comprehensive summary of the entire project, including methodology, experiments, results, and critical analysis
- `/assignment/`: Contains the project specification(Part I and Part II) provided by the course instructor, outlining the objectives, requirements, and evaluation criteria
- `./requirements.txt`: Python packages needed to run the notebooks

## How to Run

```bash
git clone https://github.com/rf-rodrigues95/data-mining-project1.git
cd PAD-Project1
pip install -r requirements.txt
jupyter notebook
