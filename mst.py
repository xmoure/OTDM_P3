import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import time


def find_clusters_with_mst_networkx(data, k):
    # Build a complete graph with nodes representing data points
    G = nx.complete_graph(len(data))

    # Add weights (distances) to the edges
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            # Euclidean distance between data points
            distance = np.linalg.norm(data[i] - data[j])
            G.add_edge(i, j, weight=distance)

    # Compute the minimum spanning tree of the graph
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')

    # Sort edges by weight in descending order and remove k-1 highest weighted edges
    edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    for i in range(k - 1):
        mst.remove_edge(edges[i][0], edges[i][1])

    # Identify clusters (connected components)
    clusters = list(nx.connected_components(mst))

    return [list(cluster) for cluster in clusters]


def plot_clusters(data, clusters, title, route):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(8, 6))

    for i, cluster in enumerate(clusters):
        points = data[cluster]
        plt.scatter(points[:, 0], points[:, 1], c=colors[i % len(colors)], label=f'Cluster {i + 1}')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.savefig(route)
    plt.show()


def plot_data(data, title, route):
    plt.figure(figsize=(6, 4))
    plt.scatter(data[:, 0], data[:, 1])
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(route)
    plt.show()


if __name__ == "__main__":

    # Execute with Iris
    k_iris = 3  # Number of clusters
    iris_df = pd.read_csv('data/iris_data.txt', sep=' ', header=None)
    iris_data = iris_df.to_numpy()
    plot_data(iris_data, "Iris Data","images/iris.png")
    start_time = time.time()
    iris_clusters = find_clusters_with_mst_networkx(iris_data, k_iris)
    end_time = time.time() - start_time
    print(f"Iris clustering execution time: {end_time:.6f} seconds")
    plot_clusters(iris_data[:, :2], iris_clusters, "Iris Clusters", "images/iris_cl_mst.png")

    # Execute with moon
    k_moon = 2  # Number of clusters
    moon_df = pd.read_csv('data/moon_data.txt', sep=' ', header=None)
    moon_data = moon_df.to_numpy()
    plot_data(moon_data, "Moon Data","images/moon.png")
    start_time = time.time()
    moon_clusters = find_clusters_with_mst_networkx(moon_data, k_moon)
    end_time = time.time() - start_time
    print(f"Moon clustering execution time: {end_time:.6f} seconds")
    plot_clusters(moon_data[:, :2], moon_clusters, "Moon Clusters", "images/moon_cl_mst.png")


    # Execute with blob
    k_blob = 3  # Number of clusters
    blob_df = pd.read_csv('data/blob_data.txt', sep=' ', header=None)
    blob_data = blob_df.to_numpy()
    plot_data(blob_data, "Blob Data","images/blob.png")
    start_time = time.time()
    blob_clusters = find_clusters_with_mst_networkx(blob_data, k_blob)
    end_time = time.time() - start_time
    print(f"Blob clustering execution time: {end_time:.6f} seconds")
    plot_clusters(blob_data[:, :2], blob_clusters, "Blob Clusters", "images/blob_cl_mst.png")

