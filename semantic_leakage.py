import gc
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def create_random_shards(articles, embeddings):
    indices = np.arange(len(articles))
    np.random.shuffle(indices)

    mid = len(indices) // 2
    shard0_indices = indices[:mid]
    shard1_indices = indices[mid:]
    shard_assignments = np.zeros(len(articles), dtype=int)
    shard_assignments[shard1_indices] = 1
    similarity_matrix = cosine_similarity(embeddings)

    return shard_assignments, similarity_matrix, shard0_indices, shard1_indices


def get_article_preview(article, max_length=100):
    if isinstance(article, dict):
        for field in ['text', 'content', 'body', 'title', 'description']:
            if field in article and isinstance(article[field], str):
                text = article[field]
                break
        else:
            for key, value in article.items():
                if isinstance(value, str) and len(value) > 10:
                    text = value
                    break
            else:
                text = str(article)[:max_length]
    else:
        text = str(article)
    if len(text) > max_length:
        return text[:max_length] + '...'
    return text


def find_max_semantic_neighborhoods(source_indices, target_indices, similarity_matrix,
                                    articles, metadata, output_dir,
                                    similarity_threshold, top_n, chunk_size):
   
    print(f"Finding maximum semantic neighborhoods (similarity threshold: {similarity_threshold})...")
    max_neighborhoods = []

    for i in tqdm(range(0, len(source_indices), chunk_size)):
        chunk_end = min(i + chunk_size, len(source_indices))
        chunk_source_indices = source_indices[i:chunk_end]
        for j, source_idx in enumerate(chunk_source_indices):
            similarities = [similarity_matrix[source_idx, target_idx] for target_idx in target_indices]
            above_threshold = [idx for idx, sim in enumerate(similarities) if sim >= similarity_threshold]

            if above_threshold:
                similar_items = [(int(target_indices[idx]), float(similarities[idx])) for idx in above_threshold]
                similar_items.sort(key=lambda x: x[1], reverse=True)
                source_preview = get_article_preview(articles[source_idx], max_length=100)

                neighborhood = {
                    'source_index': int(source_idx),
                    'source_global_index': int(metadata['indices'][source_idx]) if 'indices' in metadata else int(source_idx),
                    'source_preview': source_preview,
                    'similar_items': [
                        {
                            'target_index': int(tgt_idx),
                            'target_global_index': int(metadata['indices'][tgt_idx]) if 'indices' in metadata else int(tgt_idx),
                            'similarity_score': float(sim),
                            'target_preview': get_article_preview(articles[tgt_idx], max_length=100)
                        }
                        for tgt_idx, sim in similar_items
                    ],
                    'neighborhood_size': len(similar_items),
                    'average_similarity': float(np.mean([sim for _, sim in similar_items]))
                }

                max_neighborhoods.append(neighborhood)

        gc.collect()
    max_neighborhoods.sort(key=lambda x: x['neighborhood_size'], reverse=True)

    if max_neighborhoods:
        sizes = [n['neighborhood_size'] for n in max_neighborhoods]
        max_size = max(sizes)
        avg_size = np.mean(sizes)
        print(f"Found {len(max_neighborhoods)} neighborhoods")
        print(f"Maximum neighborhood size: {max_size}")
        print(f"Average neighborhood size: {avg_size:.2f}")
    else:
        print("No neighborhoods found above the similarity threshold")

    return max_neighborhoods


def print_neighborhood_details(neighborhood, max_items=15):
    """Print detailed information about a semantic neighborhood"""
    print(f"Source (index: {neighborhood['source_index']})")
    print(f"Content snippet: {neighborhood['source_preview']}")
    print(f"Total similar items: {neighborhood['neighborhood_size']}")
    print(f"Average similarity: {neighborhood['average_similarity']:.4f}")
    print("\nTop similar items:")

    for i, item in enumerate(neighborhood['similar_items'][:max_items]):
        print(f"  {i+1}. Target (index: {item['target_index']}, similarity: {item['similarity_score']:.4f})")
        print(f"     Content: {item['target_preview']}\n")
    print("="*80)



def find_max_neighborhood_size(source_indices, target_indices, similarity_matrix,
                               similarity_threshold, sample_size=None):

    if sample_size and len(source_indices) > sample_size:
        source_sample = np.random.choice(source_indices, sample_size, replace=False)
    else:
        source_sample = source_indices

    max_size = 0

    for source_idx in source_sample:
        similarities = similarity_matrix[source_idx, target_indices]
        neighborhood_size = np.sum(similarities >= similarity_threshold)
        if neighborhood_size > max_size:
            max_size = neighborhood_size

    return max_size


def run_random_shards_experiment(articles, embeddings, metadata,
                                num_trials=50, similarity_threshold=0.87, sample_size=None):
    results = {
        'random_trials': []
    }
    
    distance_matrix = cosine_distances(embeddings).astype(np.float64)
    similarity_matrix = 1 - distance_matrix

    print(f"\n=== Running {num_trials} random shard trials ===")

    for trial in tqdm(range(num_trials)):
        _, _, random_shard0, random_shard1 = create_random_shards(articles, embeddings)

        max_size_0to1 = find_max_neighborhood_size(
            random_shard0, random_shard1, similarity_matrix,
            similarity_threshold, sample_size
        )

        max_size_1to0 = find_max_neighborhood_size(
            random_shard1, random_shard0, similarity_matrix,
            similarity_threshold, sample_size
        )

        results['random_trials'].append({
            'trial': trial,
            'shard0_to_shard1_max_size': int(max_size_0to1),
            'shard1_to_shard0_max_size': int(max_size_1to0),
            'overall_max_size': int(max(max_size_0to1, max_size_1to0))
        })

        gc.collect()

    random_max_sizes = [trial['overall_max_size'] for trial in results['random_trials']]
    results['random_stats'] = {
        'min_max_size': int(min(random_max_sizes)),
        'max_max_size': int(max(random_max_sizes)),
        'avg_max_size': float(np.mean(random_max_sizes)),
        'median_max_size': float(np.median(random_max_sizes)),
        'std_max_size': float(np.std(random_max_sizes))
    }

    print("\n=== Experiment Results ===")
    print(f"Random sharding statistics (across {num_trials} trials):")
    print(f"  Minimum max size: {results['random_stats']['min_max_size']}")
    print(f"  Maximum max size: {results['random_stats']['max_max_size']}")
    print(f"  Average max size: {results['random_stats']['avg_max_size']:.2f}")
    print(f"  Median max size: {results['random_stats']['median_max_size']:.2f}")
    print(f"  Standard deviation: {results['random_stats']['std_max_size']:.2f}")

    improvement_vs_avg = ((results['random_stats']['avg_max_size'] - 2) /
                        results['random_stats']['avg_max_size']) * 100

    improvement_vs_min = ((results['random_stats']['min_max_size'] - 2) /
                        results['random_stats']['min_max_size']) * 100 if results['random_stats']['min_max_size'] > 2 else 0

    print(f"\nSemantic sharding improves max neighborhood size by {improvement_vs_avg:.2f}% compared to average random sharding")
    if improvement_vs_min > 0:
        print(f"Semantic sharding improves max neighborhood size by {improvement_vs_min:.2f}% compared to best random sharding")



    return results

