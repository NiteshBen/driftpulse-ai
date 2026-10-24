"""Example 02: High-Dimensional LLM Embedding & Retrieval Drift Detection."""

import random
from driftpulse import EmbeddingDriftDetector

random.seed(42)
dim = 64

# Baseline RAG query embedding vectors
ref_embeddings = [
    [random.gauss(0.0, 1.0) for _ in range(dim)]
    for _ in range(200)
]

# Current production query embeddings with semantic topic shift (centroid shift)
curr_embeddings = [
    [random.gauss(0.6, 1.0) for _ in range(dim)]  # Latent topic shift
    for _ in range(200)
]

detector = EmbeddingDriftDetector(cosine_threshold=0.10)
result = detector.compare_embeddings(ref_embeddings, curr_embeddings)

print("=" * 60)
print("  DRIFTPULSE LLM EMBEDDING DRIFT DIAGNOSIS")
print("=" * 60)
print(f"Cosine Centroid Distance: {result.cosine_centroid_distance}")
print(f"Drift Status:             {result.centroid_drift_status.value.upper()}")
print(f"Drift Flag:               {result.has_drift}")
print(f"Top Shifted Dimensions:   {result.top_shifted_dimensions[:3]}")
print("=" * 60)
