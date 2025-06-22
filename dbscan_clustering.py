import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_iris
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def load_iris_data():
    """Load the Iris dataset from sklearn."""
    print("--- Loading Iris Dataset ---")
    
    # Load the iris dataset
    iris = load_iris()
    
    # Create DataFrame with features and target
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['target'] = iris.target
    df_iris['species'] = [iris.target_names[i] for i in iris.target]
    
    print(f"Loaded {len(df_iris)} iris samples with {len(iris.feature_names)} features")
    print(f"Features: {iris.feature_names}")
    print(f"Species: {iris.target_names}")
    
    return df_iris

def find_optimal_eps(data, k=5):
    """Find optimal epsilon value using k-nearest neighbors elbow method."""
    print("--- Finding Optimal Epsilon ---")
    
    # Calculate k-nearest neighbors distances
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, _ = neighbors_fit.kneighbors(data)
    
    # Sort distances to k-th neighbor
    distances = np.sort(distances[:, k-1])
    
    # Find elbow point using second derivative
    second_derivative = np.gradient(np.gradient(distances))
    elbow_idx = np.argmax(second_derivative)
    optimal_eps = distances[elbow_idx]
    
    print(f"Optimal epsilon: {optimal_eps:.3f}")
    return optimal_eps, distances

def plot_elbow_curve(distances, optimal_eps):
    """Plot the elbow curve for epsilon selection."""
    plt.figure(figsize=(10, 6))
    plt.plot(distances, color='blue', linewidth=2)
    plt.axvline(x=np.argmax(np.gradient(np.gradient(distances))), 
                color='red', linestyle='--', linewidth=2, 
                label=f'Optimal eps: {optimal_eps:.3f}')
    plt.xlabel('Points (sorted by distance)')
    plt.ylabel('Distance to 5th Nearest Neighbor')
    plt.title('Elbow Method for Optimal Epsilon Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('elbow_curve.png', dpi=300, bbox_inches='tight')
    print("Elbow curve saved to elbow_curve.png")
    plt.show()

def plot_clustering_results(df, labels, eps, min_samples):
    """Plot DBSCAN clustering results for Iris data."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Sepal Length vs Sepal Width
    scatter1 = axes[0, 0].scatter(df['sepal length (cm)'], df['sepal width (cm)'], 
                                 c=labels, cmap='viridis', s=50, alpha=0.7)
    axes[0, 0].set_xlabel('Sepal Length (cm)')
    axes[0, 0].set_ylabel('Sepal Width (cm)')
    axes[0, 0].set_title('Sepal Length vs Sepal Width')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Petal Length vs Petal Width
    scatter2 = axes[0, 1].scatter(df['petal length (cm)'], df['petal width (cm)'], 
                                 c=labels, cmap='viridis', s=50, alpha=0.7)
    axes[0, 1].set_xlabel('Petal Length (cm)')
    axes[0, 1].set_ylabel('Petal Width (cm)')
    axes[0, 1].set_title('Petal Length vs Petal Width')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Sepal Length vs Petal Length
    scatter3 = axes[1, 0].scatter(df['sepal length (cm)'], df['petal length (cm)'], 
                                 c=labels, cmap='viridis', s=50, alpha=0.7)
    axes[1, 0].set_xlabel('Sepal Length (cm)')
    axes[1, 0].set_ylabel('Petal Length (cm)')
    axes[1, 0].set_title('Sepal Length vs Petal Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Cluster distribution
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
    axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color=colors)
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('Cluster Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'Iris Clustering with DBSCAN\nEps={eps}, Min_samples={min_samples}, Clusters={len(set(labels))-(1 if -1 in labels else 0)}', 
                 y=1.02, fontsize=14)
    plt.savefig('iris_clustering.png', dpi=300, bbox_inches='tight')
    print("Iris clustering saved to iris_clustering.png")
    plt.show()

def plot_true_vs_predicted(df):
    """Compare DBSCAN clustering with true species labels."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot true species
    scatter1 = axes[0].scatter(df['sepal length (cm)'], df['petal length (cm)'], 
                              c=df['target'], cmap='viridis', s=50, alpha=0.7)
    axes[0].set_xlabel('Sepal Length (cm)')
    axes[0].set_ylabel('Petal Length (cm)')
    axes[0].set_title('True Species Labels')
    axes[0].grid(True, alpha=0.3)
    
    # Plot DBSCAN clusters
    scatter2 = axes[1].scatter(df['sepal length (cm)'], df['petal length (cm)'], 
                              c=df['Cluster'], cmap='viridis', s=50, alpha=0.7)
    axes[1].set_xlabel('Sepal Length (cm)')
    axes[1].set_ylabel('Petal Length (cm)')
    axes[1].set_title('DBSCAN Clusters')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('true_vs_predicted.png', dpi=300, bbox_inches='tight')
    print("True vs predicted comparison saved to true_vs_predicted.png")
    plt.show()

def main():
    """Main function to run DBSCAN clustering on Iris dataset."""
    print("=== DBSCAN Clustering on Iris Dataset ===\n")
    
    # Load Iris dataset
    df_iris = load_iris_data()
    
    # Prepare features for clustering (exclude target and species columns)
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    data = df_iris[features].values
    
    # Standardize the data (important for DBSCAN)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print("Data standardized for clustering")
    
    # Find optimal epsilon using elbow method
    optimal_eps, distances = find_optimal_eps(data_scaled)
    
    # Plot elbow curve
    plot_elbow_curve(distances, optimal_eps)
    
    # Apply DBSCAN with optimal parameters
    print("--- Applying DBSCAN Clustering ---")
    min_samples = 3  # Small dataset, so smaller min_samples
    dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data_scaled)
    
    # Add cluster labels to dataframe
    df_iris['Cluster'] = labels
    
    # Plot clustering results
    plot_clustering_results(df_iris, labels, optimal_eps, min_samples)
    
    # Compare with true labels
    plot_true_vs_predicted(df_iris)
    
    # Print summary statistics
    print("\n--- Clustering Results ---")
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")
    print(f"Total samples: {len(df_iris)}")
    
    # Show cluster statistics
    print("\nCluster Statistics:")
    for cluster in sorted(set(labels)):
        if cluster == -1:
            cluster_name = "Noise"
        else:
            cluster_name = f"Cluster {cluster}"
        
        cluster_data = df_iris[df_iris['Cluster'] == cluster]
        print(f"{cluster_name}: {len(cluster_data)} samples")
        if cluster != -1:
            print(f"  Avg Sepal Length: {cluster_data['sepal length (cm)'].mean():.2f} cm")
            print(f"  Avg Sepal Width: {cluster_data['sepal width (cm)'].mean():.2f} cm")
            print(f"  Avg Petal Length: {cluster_data['petal length (cm)'].mean():.2f} cm")
            print(f"  Avg Petal Width: {cluster_data['petal width (cm)'].mean():.2f} cm")
    
    # Show species distribution in clusters
    print("\nSpecies Distribution in Clusters:")
    for cluster in sorted(set(labels)):
        if cluster == -1:
            cluster_name = "Noise"
        else:
            cluster_name = f"Cluster {cluster}"
        
        cluster_data = df_iris[df_iris['Cluster'] == cluster]
        species_counts = cluster_data['species'].value_counts()
        print(f"{cluster_name}:")
        for species, count in species_counts.items():
            print(f"  {species}: {count}")
    
    # Save results
    df_iris.to_csv('iris_clustering_results.csv', index=False)
    print("\nResults saved to iris_clustering_results.csv")

if __name__ == "__main__":
    main()