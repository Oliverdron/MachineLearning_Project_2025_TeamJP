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
        self.target_col = config['data']['target_col']
        self.cfg = config['clustering']
        self.k_min = self.cfg['k_min']
        self.k_max = self.cfg['k_max']
        self.init = self.cfg['n_init']
        self.random_state = self.cfg['random_state']
        self.figures_dir = self.cfg['figures_dir']
        
        os.makedirs(self.figures_dir, exist_ok=True)

    def run_clustering_pipeline(self, train_df, test_df) -> tuple[pd.Series, pd.Series]:
        """
        Executes the clustering pipeline:
            1. Prepares data by dropping the target column
            2. Generates a dendrogram for hierarchical clustering visualization
            3. Determines the optimal number of clusters (K) using the Elbow Method
            4. Fits the KMeans model with the optimal K and assigns cluster labels to dataframes
        
        Args:
            train_df (pd.DataFrame): The training dataset
            test_df (pd.DataFrame): The testing dataset

        Returns:
            tuple[pd.Series, pd.Series]: Cluster assignments for training and testing datasets
        """
        logger.info("Clustering: Starting optimal K search...")

        # 1. Prepare data (Drop the target column so it doesn't bias the clusters)
        X_train_clust = train_df.drop(columns=[self.target_col]).values
        X_test_clust = test_df.drop(columns=[self.target_col]).values

        logger.debug("Clustering: Data prepared for clustering | X_train_clust: values=%s & shape=%s | X_test_clust values=%s & shape=%s",
                        X_train_clust, X_train_clust.shape, X_test_clust, X_test_clust.shape)

        # 2. Hierarchical Sampling (Dendrogram function)
        self.plot_dendrogram(X_train_clust)

        # 3. Find Optimal K (Elbow Method function)
        optimal_k = self.find_optimal_k(X_train_clust)

        # 4. Final Model Fit
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=self.init)

        # 5. Predict clusters on training data to visualize
        train_clusters = pd.Series(
            kmeans_final.fit_predict(X_train_clust),
            index=train_df.index,
            name='cluster_id'
        )

        test_clusters = pd.Series(
            kmeans_final.predict(X_test_clust),
            index=test_df.index,
            name='cluster_id'
        )

        logger.info("Clustering: Pipeline complete. Returning cluster assignments.")

        return train_clusters, test_clusters
    

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
