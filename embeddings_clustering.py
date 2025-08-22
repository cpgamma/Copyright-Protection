import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

def apply_hdbscan(embeddings, min_cluster_size, min_samples):
    print("Computing cosine distance matrix...")
    print("Applying HDBSCAN clustering...")

    distance_matrix = cosine_distances(embeddings).astype(np.float64)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',  
        gen_min_span_tree=True,
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)

    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_points = list(cluster_labels).count(-1)
    print(f"HDBSCAN found {num_clusters} clusters")
    print(f"Number of noise points: {noise_points} ({noise_points/len(cluster_labels)*100:.2f}%)")

    return cluster_labels, clusterer


def create_shards_from_clusters(cluster_labels, distance_matrix, articles, metadata):
    """
    Create two shards by taking the two largest clusters as anchors
    and assigning all other data points based on similarity
    """
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]  
    num_clusters = len(unique_clusters)
    shard_assignments = np.full(len(cluster_labels), -1)

    if num_clusters == 2:
        print("Found exactly 2 clusters, creating shards directly")
        for i, label in enumerate(cluster_labels):
            if label != -1: 
                shard_assignments[i] = 0 if label == unique_clusters[0] else 1

    else:
        print(f"Found {num_clusters} clusters, using largest clusters as anchors")

        cluster_sizes = {c: np.sum(cluster_labels == c) for c in unique_clusters}

        if len(cluster_sizes) >= 2:
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
            anchor_clusters = [sorted_clusters[0][0], sorted_clusters[1][0]]
            print(f"Using clusters {anchor_clusters[0]} and {anchor_clusters[1]} as anchors")
            for i, label in enumerate(cluster_labels):
                if label == anchor_clusters[0]:
                    shard_assignments[i] = 0
                elif label == anchor_clusters[1]:
                    shard_assignments[i] = 1
        else:
            anchor_clusters = []


    other_points = np.where(shard_assignments == -1)[0]
    print(f"Assigning {len(other_points)} additional points to the two shards")


    similarity_matrix = 1 - distance_matrix

    for idx in other_points:
        shard0_mask = shard_assignments == 0
        shard1_mask = shard_assignments == 1

        if not np.any(shard0_mask):
            shard_assignments[idx] = 1
            continue
        elif not np.any(shard1_mask):
            shard_assignments[idx] = 0
            continue

        sim_to_shard0 = np.mean(similarity_matrix[idx, shard0_mask])
        sim_to_shard1 = np.mean(similarity_matrix[idx, shard1_mask])
        shard_assignments[idx] = 0 if sim_to_shard0 > sim_to_shard1 else 1

    print(f"Final shard distribution: {np.bincount(shard_assignments)}")


    shard0_indices = np.where(shard_assignments == 0)[0]
    shard1_indices = np.where(shard_assignments == 1)[0]

    shards = [
        [articles[i] for i in shard0_indices],
        [articles[i] for i in shard1_indices]
    ]

    shard_mapping = {
        'shard_0': shard0_indices.tolist(),
        'shard_1': shard1_indices.tolist()
    }

    return shard_assignments, similarity_matrix, shard0_indices, shard1_indices


