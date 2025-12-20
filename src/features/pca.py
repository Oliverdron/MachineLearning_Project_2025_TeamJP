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
        self.cfg = config['pca']  # Accessing parameters under pca from settings.json
        self.numeric_cols = config['columns']['numeric'] # Seperating all the numeric columns to scale
        self.target_col = config['data']['target_col'] # Getting the target column 
        self.figures_dir = self.cfg['figures_dir'] # Saving figures to this directory
        self.random_state = self.cfg['random_state'] # random state for reproducibility value
        self.variance_cutoff = self.cfg['variance_cutoff'] # 95 % in our case
        
        # Ensure directories exist
        os.makedirs(self.figures_dir, exist_ok=True)

    def run_pca_pipeline(self, train_df, test_df):
        logger.info("PCA: Scaling numeric features and preparing full dataset...")

        # 1. Scale ONLY numeric columns (StandardScaler)
        scaler = StandardScaler()
        train_num_scaled = scaler.fit_transform(train_df[self.numeric_cols])
        test_num_scaled = scaler.transform(test_df[self.numeric_cols])

        # 2. Identify and isolate Categorical (One-Hot) in order to merge with the new scaled numeric_cols
        # We exclude numeric and target
        ignore = self.numeric_cols + [self.target_col]
        cat_cols = [c for c in train_df.columns if c not in ignore]
        feature_names = self.numeric_cols + cat_cols
        
        # 3. Combine into one matrix for PCA
        X_train_full = np.hstack([train_num_scaled, train_df[cat_cols].values]) # merges the new scaled numeric with the categorical columns + target
        X_test_full = np.hstack([test_num_scaled, test_df[cat_cols].values])

        # 4. Fit PCA on training data only
        pca_full = PCA(random_state=self.random_state)
        pca_full.fit(X_train_full)

        # 5. Determine optimal components for variance cutoff (e.g., 95%)
        exp_var = pca_full.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)
        eigenvalues = pca_full.explained_variance_

        variance_cutoff = self.variance_cutoff # e.g., 95 % in our case
        n_components = np.argmax(cum_var >= self.variance_cutoff) + 1 # how many components are needed to reach the variance cutoff
        
        logger.info(f"PCA: Retaining {n_components} components for {self.cfg['variance_cutoff']*100}% variance.")

        # 6. Transform data by refitting PCA with optimal number of components
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_train_pca = pca.fit_transform(X_train_full)
        X_test_pca  = pca.transform(X_test_full)

        # X axis labels for plots 
        PC_number = np.arange(1, len(exp_var) + 1)

        # 7. Save Plots
        self.scree_plot_bar(exp_var, cum_var, n_components) # Scree Bar Plot
        self.scree_plot_line(exp_var, PC_number)  # Scree Line Plot
        self.scree_plot_linekaiser(eigenvalues, PC_number)  # Scree Line Kaiser Plot
        self.loadingsheatmap_plot(pca, feature_names)  # Loadings Heatmap Plot
        self.scatter_plot2D(X_train_pca, exp_var)  # 2D Scatter Plot

        # 8. Create Final Handoff DataFrames
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        
        train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns)
        test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns)

        # Attach Target variables (ensures rows line up)
        train_pca_df[self.target_col] = train_df[self.target_col].values
        test_pca_df[self.target_col] = test_df[self.target_col].values

        logger.info(f"PCA: Pipeline complete. Returning DataFrames with {n_components} PCs.")

        return train_pca_df, test_pca_df
    
    # FOLLOWING ARE FUNCITONS FOR GRAPHS AND WILL BE CALLED IN run_pca_pipeline 

    def scree_plot_bar(self, exp_var, cum_var, n_components):
        logger.info("PCA: Generating and saving Scree Bar Plot...")
       # 1. Determine how many PCs to show (up to 10 or max available)
        n_display = min(10, len(cum_var))

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

        # Labels and Title
        plt.title(f"PCA Scree Plot: {n_components} PCs required for {self.variance_cutoff*100}% Variance", fontsize=14)
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
    
    def scree_plot_line(self, exp_var, PC_number):
        plt.figure(figsize=(10,6))
        plt.plot(PC_number, exp_var, marker='o', linestyle='-', color='g')
        plt.title('Scree Plot (Line Style) Elbow Method', fontsize =14)
        plt.xlabel('Principal Component Number', fontsize=12)
        plt.ylabel('Proportion of Variance Explained', fontsize=12) 
        plt.grid()

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
        plt.legend(loc = 'upper right')
        plt.grid()
        savepath_screelinekaiser = os.path.join(self.figures_dir, "pca_scree_line_kaiser.png")
        plt.savefig(savepath_screelinekaiser)
        plt.close()
        
        logger.info(f"Scree Line Kaiser Plot saved to: {savepath_screelinekaiser}")

    def loadingsheatmap_plot(self, pca, feature_names):
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

        plt.figure(figsize=(14, 12)) # Wider and taller for One-Hot labels
        sns.heatmap(loadings, annot=True, cmap='RdBu', center=0, fmt=".2f", 
                    linewidths=.5, cbar_kws={"shrink": .8})
        plt.title("PCA Loadings: Feature Contribution (Top 30)", fontsize=16)
        
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