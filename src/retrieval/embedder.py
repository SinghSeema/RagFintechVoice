"""
Task 4 — Embeddings via BGE-M3 (HuggingFaceEmbedding).

BGE-M3 (BAAI/bge-m3):
  - 1024-dimensional dense vectors
  - MIT licensed, runs fully locally (no API call)
  - Handles Indian English and regulatory acronyms (RBI, CIBIL, PMLA, FEMA)
  - First call downloads ~570MB model to ~/.cache/huggingface/; subsequent
    runs load from cache in seconds.

Output: get_embed_model() -> HuggingFaceEmbedding
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

_MODEL_NAME = "BAAI/bge-m3"
_EMBED_BATCH_SIZE = 32   # safe default for CPU; reduce if OOM

# Singleton — BGE-M3 is ~2 GB in RAM; never load it twice in one process.
_embed_model: HuggingFaceEmbedding | None = None


def get_embed_model() -> HuggingFaceEmbedding:
    """
    Return the process-wide BGE-M3 embedding model (loaded once on first call).

    BGE-M3 occupies ~2 GB of RAM as PyTorch fp32 tensors.  All callers in the
    same process share this instance — do not call HuggingFaceEmbedding()
    directly elsewhere.
    """
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(
            model_name=_MODEL_NAME,
            embed_batch_size=_EMBED_BATCH_SIZE,
        )
    return _embed_model


if __name__ == "__main__":
    import numpy as np

    print(f"Loading {_MODEL_NAME} ...")
    embed_model = get_embed_model()

    # ADR Task 4 test texts
    texts = [
        "KYC documents for NRI account opening",
        "NRI customers must provide valid passport",
        "Housing loan LTV ratio guidelines",
    ]

    print("Embedding 3 texts ...")
    vectors = embed_model.get_text_embedding_batch(texts)

    # Verify shape
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        arr = np.array(vec)
        print(f"  [{i}] shape={arr.shape}  norm={np.linalg.norm(arr):.4f}  text={text!r}")

    print()

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    query_vec = embed_model.get_text_embedding("KYC documents for NRI account opening")

    sim0 = cosine_sim(query_vec, vectors[0])  # same text
    sim1 = cosine_sim(query_vec, vectors[1])  # NRI KYC — should be high
    sim2 = cosine_sim(query_vec, vectors[2])  # housing loan — should be lower

    print("Cosine similarity to query 'KYC documents for NRI account opening':")
    print(f"  [{0}] '{texts[0]}' → {sim0:.4f}")
    print(f"  [{1}] '{texts[1]}' → {sim1:.4f}")
    print(f"  [{2}] '{texts[2]}' → {sim2:.4f}")

    print()
    nri_beats_housing = sim1 > sim2
    print(f"NRI KYC chunk scores higher than housing loan chunk: {nri_beats_housing}")
    assert nri_beats_housing, "FAIL: NRI chunk should score higher than housing loan chunk"
    print("PASS")
