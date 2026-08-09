import os
import httpx

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

HF_MODEL_URL = (
    "https://router.huggingface.co/"
    "hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/"
    "pipeline/feature-extraction"
)


async def get_query_embedding_async(text: str) -> list[float]:
    """Generate a 384-dimensional MiniLM embedding."""

    if not HUGGINGFACE_API_KEY:
        raise RuntimeError(
            "HUGGINGFACE_API_KEY is not configured."
        )

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": text
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:

            response = await client.post(
                HF_MODEL_URL,
                headers=headers,
                json=payload,
            )

            print(
                f"🤗 HuggingFace status: "
                f"{response.status_code}"
            )

            if response.status_code != 200:
                print(
                    f"❌ HuggingFace response: "
                    f"{response.text[:1000]}"
                )

                raise RuntimeError(
                    f"HuggingFace embedding failed "
                    f"with HTTP {response.status_code}"
                )

            data = response.json()

            # Expected response:
            # [[0.123, 0.456, ...]]
            if (
                isinstance(data, list)
                and len(data) > 0
                and isinstance(data[0], list)
            ):
                vector = data[0]

            elif (
                isinstance(data, list)
                and len(data) > 0
                and isinstance(data[0], (float, int))
            ):
                vector = data

            else:
                raise RuntimeError(
                    f"Unexpected HuggingFace response: "
                    f"{str(data)[:1000]}"
                )

            vector = [float(x) for x in vector]

            if len(vector) != 384:
                raise RuntimeError(
                    f"Wrong embedding dimension: "
                    f"expected 384, got {len(vector)}"
                )

            if not any(x != 0.0 for x in vector):
                raise RuntimeError(
                    "HuggingFace returned an all-zero embedding."
                )

            print(
                "✅ HuggingFace embedding generated: "
                f"{len(vector)} dimensions"
            )

            return vector

    except httpx.TimeoutException as err:

        print(
            f"❌ HuggingFace timeout: {err}"
        )

        raise RuntimeError(
            "HuggingFace embedding request timed out."
        ) from err

    except Exception as err:

        print(
            f"❌ HuggingFace embedding error: "
            f"{type(err).__name__}: {err}"
        )

        raise
