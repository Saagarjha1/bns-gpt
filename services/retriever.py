from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    print(f"🤖 Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("✅ Embedding model loaded")

    return model


async def get_query_embedding_async(text: str) -> list[float]:
    """
    Generate a 384-dimensional embedding locally.
    No Hugging Face API is required.
    """

    if not text or not text.strip():
        raise ValueError(
            "Cannot embed an empty query."
        )

    model = get_model()

    vector = model.encode(
        text,
        normalize_embeddings=True
    )

    vector = vector.tolist()

    if len(vector) != 384:
        raise RuntimeError(
            f"Expected 384 dimensions, "
            f"got {len(vector)}"
        )

    if not any(
        float(value) != 0.0
        for value in vector
    ):
        raise RuntimeError(
            "Embedding model returned an all-zero vector."
        )

    print(
        f"✅ Local embedding generated: "
        f"{len(vector)} dimensions"
    )

    return [
        float(value)
        for value in vector
    ]
