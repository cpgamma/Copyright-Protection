# Secure and Efficient Copyright Protection for LLMs

## Overview
This repository contains the implementation for "Secure and Efficient Copyright Protection for LLMs" submitted to USENIX Security 2026.


## Installation & Setup
# Install dependencies
```python
pip install torch transformers datasets
pip install scikit-learn hdbscan tqdm numpy
pip install sentence-transformers  
```


## Dataset
Our evaluation uses a comprehensive news article dataset:
- **15,000+ articles** collected via Apify.com (2024-2025) from major news outlets:
  - CNN
  - NBC News  
  - The New York Times
  - The Washington Post
- **Additional articles** from The Guardian (2024-2025) via Guardian API


## Semantic Leakage Analysis
We analyze semantic leakage in random data sharding.

```python

results = run_random_shards_experiment(
    articles=articles,
    embeddings=embeddings, 
    metadata=metadata,
    num_trials=50,
    similarity_threshold=0.87
)
```

## Semantic Sharding Algorithm

```python
cluster_labels, clusterer = apply_hdbscan(
    embeddings, 
    min_cluster_size=,    
    min_samples=          
)

shard_assignments, similarity_matrix, shard0_indices, shard1_indices = create_shards_from_clusters(
    cluster_labels, distance_matrix, articles, metadata
)
```

## Models Training

```python
model_qx, tokenizer = train_full_model(
    model_name="meta-llama/Llama-3.2-1B",  
    dataset_path="dataset.jsonl", 
    output_dir="model_qx",
    seed=42
)

```

## Text generation

```python
protected_text_original = cp_delta_generate(
    model_q1=model_q1,      
    model_q2=model_q2,        
    tokenizer=tokenizer,
    prompt="",
    max_new_tokens=250,
    delta_type="max"          
)

protected_text_gamma_clipped = single_model_gamma_clipped_dynamic(
    model=full_model, 
    tokenizer=tokenizer, 
    prompt="", 
    max_new_tokens=250, 
    gamma=0.60
    )
```

## Security Evaluation

```python

embedding_files=[
        {
            'name': 'method_name',
            'embedding_path': 'method_embedding.npy',
            'json_path': 'generated_answers.jsonl'
        },
]
results = analyze_violations(
        embedding_files=embedding_files,
        eval_samples_embed=eval_embeddings,
        eval_samples=eval_samples,
        threshold=0.87,
        eva_dir=EVA_DIR
    )
```

## Utility Evaluation 

```python
evaluator = GeneralizedEvaluationRunner(
    openai_api_key="your-api-key",
    model="gpt-5-mini"
)
evaluator.load_prompts_file("prompts.jsonl")
methods = [
    ("method_1", lambda prompt: cp_delta_generate_original(model_q1, model_q2, tokenizer, prompt, 250, "max",0,3)),
    ("method_2", lambda prompt: single_model_gamma_clipped_dynamic(full_model, tokenizer, prompt, 250, 0.4,0,3)),
]
evaluator.register_evaluation_methods(methods)

results = evaluator.run_evaluation(
    num_iterations=10,
    prompt_text_field='text',  
    verbose=True
)
```