import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class PCAManager:
    def __init__(self, config):
        self.numeric_cols = config['columns']['numeric'] # Seperating all the numeric columns to scale
        self.target_col = config['data']['target_col'] # Getting the target column 
        self.cfg = config['pca']  # Accessing parameters under pca from settings.json
        self.variance_cutoff = self.cfg['variance_cutoff'] # 95 % in our case
        self.random_state = self.cfg['random_state'] # random state for reproducibility value
        self.figures_dir = self.cfg['figures_dir'] # Saving figures to this directory
        
        # Ensure directories exist
        os.makedirs(self.figures_dir, exist_ok=True)

    def run_pca_pipeline(self, train_df, test_df) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs the PCA pipeline on the provided training and testing dataframes.

        Args:
            train_df (pd.DataFrame): The training dataset
            test_df (pd.DataFrame): The testing dataset

        Logic:
            1. Scale numeric features using StandardScaler
            2. Combine scaled numeric features with (encoded) categorical features
            3. Fit PCA on the training data to determine the number of components needed to reach the variance cutoff
            4. Transform both training and testing datasets using the fitted PCA

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Transformed training and testing datasets with principal components and target variable
        """
        logger.info("Scaling numeric features and preparing full dataset...")

        # 1. Scale ONLY numeric columns (StandardScaler)
        scaler = StandardScaler()
        train_num_scaled = scaler.fit_transform(train_df[self.numeric_cols])
        test_num_scaled = scaler.transform(test_df[self.numeric_cols])

        logger.debug("Numeric features scaled | train_num_scaled values=%s & shape=%s | test_num_scaled values=%s & shape=%s",
                     train_num_scaled, train_num_scaled.shape,
                     test_num_scaled, test_num_scaled.shape)

        # 2. Identify and isolate Categorical (One-Hot) in order to merge with the new scaled numeric_cols
        # We exclude numeric and target
        ignore = self.numeric_cols + [self.target_col]
        encoded_cols = [c for c in train_df.columns if c not in ignore]
        feature_names = self.numeric_cols + encoded_cols
        
        # 3. Combine into one matrix for PCA
        X_train_full = np.hstack([train_num_scaled, train_df[encoded_cols].values]) # merges the new scaled numeric with the categorical columns + target
        X_test_full = np.hstack([test_num_scaled, test_df[encoded_cols].values])

        # 4. Fit PCA on training data only (prevent data leakage)
        pca_full = PCA(random_state=self.random_state)
        pca_full.fit(X_train_full)

        # 5. Determine optimal components for variance cutoff (e.g., 95%)
        exp_var = pca_full.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)
        eigenvalues = pca_full.explained_variance_

        n_components = np.argmax(cum_var >= self.variance_cutoff) + 1 # how many components are needed to reach the variance cutoff
        
        logger.info(f"Retaining {n_components} components for {self.variance_cutoff * 100}% variance.")

        # 6. Transform data by refitting PCA with optimal number of components
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_train_pca = pca.fit_transform(X_train_full)
        X_test_pca  = pca.transform(X_test_full)

        # X axis labels for plots 
        PC_number = np.arange(1, len(exp_var) + 1)

        # 7. Save Plots
        self.scree_plot_bar(exp_var, cum_var, n_components) # Scree Bar Plot
        self.scree_plot_line(exp_var, PC_number, n_components)  # Scree Line Plot
        self.scree_plot_linekaiser(eigenvalues, PC_number)  # Scree Line Kaiser Plot
        self.loadingsheatmap_plot(pca, feature_names)  # Loadings Heatmap Plot
        self.scatter_plot2D(X_train_pca, exp_var)  # 2D Scatter Plot

        # 8. Create Final Handoff DataFrames
        pca_columns = [f'PC_{i+1}' for i in range(n_components)]
        
        train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns)
        test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns)

        # Attach Target variables (ensures rows line up)
        train_pca_df[self.target_col] = train_df[self.target_col].values
        test_pca_df[self.target_col] = test_df[self.target_col].values

        logger.debug("PCA transformed DataFrames created | train_pca_df values=%s & shape=%s | test_pca_df values=%s & shape=%s",
                     train_pca_df, train_pca_df.shape,
                     test_pca_df, test_pca_df.shape)

        logger.info(f"Pipeline complete. Returning DataFrames with {n_components} PCs.")

        return train_pca_df, test_pca_df
    

    def scree_plot_bar(self, exp_var, cum_var, n_components):
        logger.info("Generating and saving Scree Bar Plot...")
       # 1. Determine how many PCs to show (the number of components we have, capped at 20 for readability)
        n_display = min(len(exp_var), n_components, 20)

        # 2. Create a list of colors: Green for 'Keepers', Red for the rest
        colors = ['green' if i < n_components else 'red' for i in range(n_display)]

        # 3. Plotting
        plt.figure(figsize=(12, 6))

        # Plot the bars
        plt.bar(range(1, n_display + 1), exp_var[:n_display], 
                color=colors, alpha=0.6, label='Individual Variance')

        # Adding percentage labels on top of bars
        for i, v in enumerate(exp_var[:n_display]):
            plt.text(i + 1, v + 0.01, f"{v*100:.1f}%", 
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Plot the cumulative staircase
        plt.step(range(1, n_display + 1), cum_var[:n_display], 
                 where='mid', label='Cumulative Variance', color='black', linewidth=2)

        # Add the threshold line (pulled from JSON config)
        plt.axhline(y=self.variance_cutoff, color='blue', linestyle='--', label=f'{self.variance_cutoff*100}% Variance Goal')

        # cutoff marker
        plt.axvline(x=n_components, linestyle=":", linewidth=2)
        plt.text(n_components + 0.2, 0.02, f"cutoff = PC{n_components}", rotation=90, va="bottom")

        # Labels and Title
        plt.title(f"Scree Plot: {n_components} PCs required for {self.variance_cutoff*100}% Variance", fontsize=14)
        plt.xlabel("Principal Component Index", fontsize=12)
        plt.ylabel("Proportion of Variance Explained", fontsize=12)
        plt.xticks(range(1, n_display + 1)) 
        plt.ylim(0, 1.15) 
        plt.legend(loc='upper left')
        plt.grid(axis='y', linestyle=':', alpha=0.7)

        # 4. Saving and closing the plot 
        savepath_screebar = os.path.join(self.figures_dir, "pca_scree_bars.png")
        plt.savefig(savepath_screebar)
        plt.close()
        
        logger.info(f"Scree Bar Plot saved to: {savepath_screebar}")
    

    def scree_plot_line(self, exp_var, PC_number, n_components=None):
        cum_var = np.cumsum(exp_var)

        plt.figure(figsize=(10, 6))
        plt.plot(PC_number, exp_var, marker="o", linestyle="-", label="Individual")
        plt.plot(PC_number, cum_var, marker=None, linestyle="--", linewidth=2, label="Cumulative")

        plt.axhline(y=self.variance_cutoff, linestyle="--", label=f"{self.variance_cutoff*100:.1f}% goal")

        if n_components is not None:
            plt.axvline(x=n_components, linestyle=":", linewidth=2)
            plt.text(n_components + 0.2, 0.02, f"cutoff = PC{n_components}", rotation=90, va="bottom")

        plt.title('Scree Plot: Individual vs Cumulative Explained Variance', fontsize =14)
        plt.xlabel('Principal Component Number', fontsize=12)
        plt.ylabel('Proportion of Variance Explained', fontsize=12) 
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')

        # Saving and closing the plot 
        savepath_screeline = os.path.join(self.figures_dir, "pca_scree_line.png")
        plt.savefig(savepath_screeline)
        plt.close()
        
        logger.info(f"Scree Line Plot saved to: {savepath_screeline}")


    def scree_plot_linekaiser(self, eigenvalues, PC_number):
        plt.figure(figsize=(10,6))
        plt.plot(PC_number, eigenvalues, marker='o', linestyle='-', color='b')
        plt.title('ScreePlot (Line Style) Kaiser Method', fontsize =14)
        plt.xlabel('Principal Component Number', fontsize=12)
        plt.ylabel('Absolute Eigen Values', fontsize=12) 
        plt.axhline(y= 1, color='r', linestyle='--', label='Kaiser Threshold (Eigenvalue=1)')
        k_kaiser = int(np.sum(np.array(eigenvalues) > 1.0))
        plt.text(0.98, 0.95, f"Kaiser suggests K = {k_kaiser} PCs", transform=plt.gca().transAxes, ha="right", va="top")
        plt.legend(loc = 'upper right')
        plt.grid()
        savepath_screelinekaiser = os.path.join(self.figures_dir, "pca_scree_line_kaiser.png")
        plt.savefig(savepath_screelinekaiser)
        plt.close()
        
        logger.info(f"Scree Line Kaiser Plot saved to: {savepath_screelinekaiser}")


    def loadingsheatmap_plot(self, pca, feature_names, eps: float = 0.005):
        # 1. Get loadings
        loadings = pd.DataFrame(
            pca.components_.T, 
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=feature_names
        )

        # Only show the top 30 most 'important' features to keep it clean
        # If we have 100+ one-hot columns, a heatmap of everything is unreadable.
        if len(feature_names) > 30:
            # Calculate total absolute influence across all PCs
            importance = loadings.abs().sum(axis=1).sort_values(ascending=False)
            loadings = loadings.loc[importance.head(30).index]
            logger.info("Heatmap: Only showing top 30 features for readability.")

        mask = loadings.abs() < eps  # Annotate only significant loadings
        annot = loadings.where(~mask).map(lambda x: "" if pd.isna(x) else f"{x:.2f}")  # Blank out small values for clarity
        v = loadings.abs().values.max()

        plt.figure(figsize=(14, 12)) # Wider and taller for One-Hot labels
        sns.heatmap(loadings,
                    annot=annot,
                    cmap='RdBu',
                    center=0,
                    vmin=-v, vmax=v,
                    fmt="", 
                    linewidths=.5,
                    cbar_kws={"shrink": .8}
                )
        plt.title(
            f"PCA Loadings: Feature Contribution (Top {loadings.shape[0]}, |loading| ≥ {eps})",
            fontsize=16
        )
        
        #Saving and closing the plot
        savepath_heatmap = os.path.join(self.figures_dir, "pca_loadings_heatmap.png")
        plt.savefig(savepath_heatmap, bbox_inches='tight') 
        plt.close()
    

    def scatter_plot2D(self, X_train_pca, exp_var):
        """Creates a 2D scatter plot of the first two Principal Components."""
        
        plt.figure(figsize=(10, 7))
        
        # Plotting PC1 vs PC2
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.4, c='teal', edgecolors='white')

        # Using exp_var to show how much variance each axis represents
        plt.xlabel(f'PC1 ({exp_var[0]*100:.1f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({exp_var[1]*100:.1f}%)', fontsize=12)
        
        plt.title('PCA 2D Projection: Scatter Plot Principal Components', fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.6)

        # Saving and closing the plot
        savepath_scatter = os.path.join(self.figures_dir, "pca_scatter_2d.png")
        plt.savefig(savepath_scatter, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PCA 2D Scatter Plot saved to: {savepath_scatter}")