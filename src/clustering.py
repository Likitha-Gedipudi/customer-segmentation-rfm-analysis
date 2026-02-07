"""
Clustering Module
Implements K-Means clustering for customer segmentation with optimal K selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def prepare_features(
    rfm_df: pd.DataFrame,
    feature_cols: list = None
) -> tuple:
    """
    Prepare and scale features for clustering.
    
    Parameters:
    -----------
    rfm_df : DataFrame with RFM metrics
    feature_cols : List of columns to use as features (default: RFM + AOV)
    
    Returns:
    --------
    Tuple of (scaled_features, scaler, feature_columns)
    """
    
    if feature_cols is None:
        feature_cols = ['recency', 'frequency', 'monetary', 'avg_order_value']
    
    # Extract features
    features = rfm_df[feature_cols].copy()
    
    # Handle any missing values
    features = features.fillna(features.median())
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    print(f"Prepared {len(feature_cols)} features for clustering")
    print(f"Features: {feature_cols}")
    
    return scaled_features, scaler, feature_cols


def find_optimal_k(
    scaled_features: np.ndarray,
    k_range: range = range(2, 11),
    save_plot: str = None
) -> dict:
    """
    Find optimal number of clusters using Elbow and Silhouette methods.
    
    Parameters:
    -----------
    scaled_features : Scaled feature array
    k_range : Range of K values to test
    save_plot : Path to save the evaluation plot
    
    Returns:
    --------
    Dictionary with evaluation metrics for each K
    """
    
    results = {
        'k': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': []
    }
    
    print("Evaluating cluster counts...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        
        results['k'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(scaled_features, labels))
        results['davies_bouldin'].append(davies_bouldin_score(scaled_features, labels))
        
        print(f"  K={k}: Silhouette={results['silhouette'][-1]:.3f}, "
              f"Davies-Bouldin={results['davies_bouldin'][-1]:.3f}")
    
    # Create evaluation plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Elbow plot
    axes[0].plot(results['k'], results['inertia'], 'bo-')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    
    # Silhouette plot
    axes[1].plot(results['k'], results['silhouette'], 'go-')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    
    # Davies-Bouldin plot
    axes[2].plot(results['k'], results['davies_bouldin'], 'ro-')
    axes[2].set_xlabel('Number of Clusters (K)')
    axes[2].set_ylabel('Davies-Bouldin Index')
    axes[2].set_title('Davies-Bouldin Analysis')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(save_plot, dpi=150, bbox_inches='tight')
        print(f"Saved evaluation plot to: {save_plot}")
    
    plt.close()
    
    # Find optimal K (highest silhouette)
    optimal_idx = np.argmax(results['silhouette'])
    optimal_k = results['k'][optimal_idx]
    print(f"\nOptimal K based on Silhouette: {optimal_k}")
    
    return results


def apply_kmeans(
    scaled_features: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> tuple:
    """
    Apply K-Means clustering.
    
    Returns:
    --------
    Tuple of (cluster_labels, kmeans_model)
    """
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    labels = kmeans.fit_predict(scaled_features)
    
    print(f"\nK-Means clustering with K={n_clusters}")
    print(f"Silhouette Score: {silhouette_score(scaled_features, labels):.3f}")
    
    return labels, kmeans


def profile_clusters(
    rfm_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_cols: list
) -> pd.DataFrame:
    """
    Create cluster profiles with descriptive statistics.
    
    Returns:
    --------
    DataFrame with cluster profiles
    """
    
    df = rfm_df.copy()
    df['cluster'] = cluster_labels
    
    # Calculate cluster statistics
    profiles = df.groupby('cluster').agg({
        'customer_id': 'count',
        'recency': 'mean',
        'frequency': 'mean', 
        'monetary': ['mean', 'sum'],
        'avg_order_value': 'mean',
        'R_score': 'mean',
        'F_score': 'mean',
        'M_score': 'mean'
    }).round(2)
    
    profiles.columns = ['customer_count', 'avg_recency', 'avg_frequency',
                        'avg_monetary', 'total_revenue', 'avg_order_value',
                        'avg_R', 'avg_F', 'avg_M']
    
    profiles['revenue_pct'] = (profiles['total_revenue'] / 
                                profiles['total_revenue'].sum() * 100).round(1)
    profiles['customer_pct'] = (profiles['customer_count'] / 
                                 profiles['customer_count'].sum() * 100).round(1)
    
    return profiles.sort_values('avg_monetary', ascending=False)


def assign_cluster_labels(profiles: pd.DataFrame) -> dict:
    """
    Assign descriptive labels to clusters based on their characteristics.
    
    Returns:
    --------
    Dictionary mapping cluster numbers to labels
    """
    
    labels = {}
    profiles_sorted = profiles.sort_values('avg_monetary', ascending=False)
    
    cluster_order = profiles_sorted.index.tolist()
    
    # Label based on relative position and characteristics
    label_templates = [
        'VIP Champions',
        'Loyal High-Value',
        'Regular Customers',
        'Occasional Buyers',
        'At-Risk',
        'Dormant',
        'Lost'
    ]
    
    for i, cluster in enumerate(cluster_order):
        if i < len(label_templates):
            labels[cluster] = label_templates[i]
        else:
            labels[cluster] = f'Cluster {cluster}'
    
    return labels


def run_clustering_analysis(
    rfm_df: pd.DataFrame,
    n_clusters: int = None,
    save_path: str = None,
    plots_dir: str = None
) -> tuple:
    """
    Run complete clustering pipeline.
    
    Parameters:
    -----------
    rfm_df : DataFrame with RFM scores
    n_clusters : Number of clusters (if None, will be determined automatically)
    save_path : Path to save clustered data
    plots_dir : Directory to save plots
    
    Returns:
    --------
    Tuple of (clustered_df, cluster_profiles)
    """
    
    # Prepare features
    feature_cols = ['recency', 'frequency', 'monetary', 'avg_order_value']
    scaled_features, scaler, features = prepare_features(rfm_df, feature_cols)
    
    # Find optimal K if not specified
    if n_clusters is None:
        plot_path = f"{plots_dir}/cluster_evaluation.png" if plots_dir else None
        results = find_optimal_k(scaled_features, save_plot=plot_path)
        n_clusters = results['k'][np.argmax(results['silhouette'])]
    
    # Apply clustering
    labels, model = apply_kmeans(scaled_features, n_clusters)
    
    # Profile clusters
    profiles = profile_clusters(rfm_df, labels, feature_cols)
    print("\nCluster Profiles:")
    print(profiles)
    
    # Assign descriptive labels
    cluster_labels = assign_cluster_labels(profiles)
    
    # Create final clustered DataFrame
    clustered_df = rfm_df.copy()
    clustered_df['cluster'] = labels
    clustered_df['cluster_label'] = clustered_df['cluster'].map(cluster_labels)
    
    if save_path:
        clustered_df.to_csv(save_path, index=False)
        print(f"\nSaved clustered data to: {save_path}")
    
    return clustered_df, profiles


if __name__ == "__main__":
    # Test with RFM data
    rfm_df = pd.read_csv("data/processed/rfm_scores.csv")
    clustered_df, profiles = run_clustering_analysis(
        rfm_df,
        save_path="data/processed/customer_segments.csv",
        plots_dir="reports/figures"
    )
