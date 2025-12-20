import os
import logging
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


logger = logging.getLogger(__name__)

class ClusterManager:
    def __init__(self, config):
        self.cfg = config['clustering']
        self.figures_dir = self.cfg['figures_dir']
        self.random_state = self.cfg['random_state']
        self.init = self.cfg['n_init']
        self.target_col = config['data']['target_col']
        
        os.makedirs(self.figures_dir, exist_ok=True)

    def run_clustering_pipeline(self, train_pca_df, test_pca_df):
        logger.info("Clustering: Starting optimal K search...")

        # 1. Prepare data (Drop the target column so it doesn't bias the clusters)
        X_train_clust = train_pca_df.drop(columns=[self.target_col]).values
        X_test_clust = test_pca_df.drop(columns=[self.target_col]).values

        # 2. Hierarchical Sampling (Dendrogram function)
        self.plot_dendrogram(X_train_clust)

        # 3. Find Optimal K (Elbow Method function)
        optimal_k = self.find_optimal_k(X_train_clust)

        # 4. Final Model Fit
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=self.init)
        
        # Add Cluster labels to the DataFrames
        train_pca_df['Cluster'] = kmeans_final.fit_predict(X_train_clust)
        test_pca_df['Cluster'] = kmeans_final.predict(X_test_clust)

        # 5. Visualize Results
        self.plot_clusters_2D(train_pca_df, kmeans_final)
        # 6. Final Conversion to NumPy for Neural Networks/Trees
        logger.info("Clustering: Converting final DataFrames to NumPy arrays for modeling")
        X_train_array = train_pca_df.values
        X_test_array = test_pca_df.values

        return X_train_array, X_test_array
    
    def find_optimal_k(self, X_clust):
            inertia = []
            k_range = range(1, 9)
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.init)
                km.fit(X_clust)
                inertia.append(km.inertia_)

            kn = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
            optimal_k = kn.knee or 4 # Default to 4 if knee is not found
            
            self.plot_elbow(k_range, inertia, optimal_k)
            logger.info(f"Clustering: Mathematical optimal K determined as {optimal_k}")
            return optimal_k

    def plot_elbow(self, k_range, inertia, optimal_k):
        """Saves the Elbow Method visualization."""
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, inertia, marker='o', color='purple', label='Inertia')
        plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K = {optimal_k}')
        plt.title('Clustering Elbow Method (Inertia)', fontsize=14)
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Inertia', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        clustering_elbow = os.path.join(self.figures_dir, "clustering_elbow.png")
        plt.savefig(clustering_elbow)
        plt.close()

    def plot_dendrogram(self, X_clust):
        logger.info("Clustering: Generating sampled dendrogram...")
        np.random.seed(self.random_state)
        
        # Sample 1000 points safely
        n_samples = min(1000, X_clust.shape[0])
        indices = np.random.choice(X_clust.shape[0], n_samples, replace=False)
        sample_pca = X_clust[indices, :]

        Z = linkage(sample_pca, method='ward')

        plt.figure(figsize=(10, 6))
        dendrogram(Z, truncate_mode='lastp', p=10)
        
        plt.title("Hierarchical Clustering Dendrogram (Sample of 1000)", fontsize=14)
        plt.xlabel("Cluster Size (Number of points in node)", fontsize=12)
        plt.ylabel("Ward Distance", fontsize=12)
        
        plot_dendogram = os.path.join(self.figures_dir, "clustering_dendrogram.png")
        plt.savefig(plot_dendogram, bbox_inches='tight')
        plt.close()
        logger.info(f"Dendrogram saved to: {plot_dendogram}")

    def plot_clusters_2D(self, train_pca_df, kmeans_model):
        """Visualizes the clusters using the first two Principal Components."""
        plt.figure(figsize=(12, 8))

        # Scatter plot colored by Cluster
        sns.scatterplot(
            x='PC1', 
            y='PC2', 
            hue='Cluster', 
            data=train_pca_df, 
            palette='viridis', 
            alpha=0.5, 
            edgecolor=None
        )

        # Plot the centroids
        centers = kmeans_model.cluster_centers_
        plt.scatter(
            centers[:, 0], 
            centers[:, 1], 
            c='red', 
            marker='X', 
            s=250, 
            label='Centroids',
            edgecolor='white'
        )

        plt.title('Final Driver Segments (PC1 vs PC2)', fontsize=15)
        plt.legend(title='Risk Cluster')
        plt.grid(True, linestyle='--', alpha=0.5)

        cluster2D_scatterplot = os.path.join(self.figures_dir, "clustering_segments_2d.png")
        plt.savefig(cluster2D_scatterplot, bbox_inches='tight')
        plt.close()
        logger.info(f"Cluster scatter plot saved to: {cluster2D_scatterplot}")