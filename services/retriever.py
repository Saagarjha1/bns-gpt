import os
import httpx

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

async def get_query_embedding_async(text: str) -> list[float]:
    """Retrieves sentence embeddings from HuggingFace with a 3-second hard timeout."""
    if not HUGGINGFACE_API_KEY:
        return [0.0] * 384

    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": text}

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.post(HF_MODEL_URL, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], float):
                    return data
                elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    return data[0]
    except Exception as err:
        print(f"⚠️ HuggingFace API network timeout or error: {err}")

    return [0.0] * 384
