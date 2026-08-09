import os
import httpx


# ============================================================
# CONFIGURATION
# ============================================================

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

HF_MODEL_URL = (
    "https://router.huggingface.co/"
    "hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/"
    "pipeline/feature-extraction"
)


# ============================================================
# QUERY EMBEDDING
# ============================================================

async def get_query_embedding_async(
    text: str
) -> list[float]:
    """
    Generate a 384-dimensional embedding for a user query.

    Uses the same model used during Colab ingestion:
        sentence-transformers/all-MiniLM-L6-v2

    Render does NOT load sentence-transformers/PyTorch.
    The embedding is generated through Hugging Face.
    """

    if not text or not text.strip():
        raise ValueError(
            "Cannot generate an embedding for an empty query."
        )

    if not HUGGINGFACE_API_KEY:
        raise RuntimeError(
            "HUGGINGFACE_API_KEY is not configured in Render."
        )

    headers = {
        "Authorization": (
            f"Bearer {HUGGINGFACE_API_KEY}"
        ),
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": text.strip(),
        "normalize": True,
    }

    try:

        async with httpx.AsyncClient(
            timeout=20.0
        ) as client:

            response = await client.post(
                HF_MODEL_URL,
                headers=headers,
                json=payload,
            )

        print(
            f"🤗 HuggingFace status: "
            f"{response.status_code}"
        )

        # ----------------------------------------------------
        # API ERROR
        # ----------------------------------------------------

        if response.status_code != 200:

            error_text = response.text[:2000]

            print(
                "❌ HuggingFace API error:"
            )

            print(error_text)

            raise RuntimeError(
                "HuggingFace embedding request failed: "
                f"HTTP {response.status_code}"
            )

        # ----------------------------------------------------
        # PARSE RESPONSE
        # ----------------------------------------------------

        data = response.json()

        # Most common response:
        #
        # [
        #   [0.123, 0.456, ...]
        # ]
        #
        if (
            isinstance(data, list)
            and len(data) > 0
            and isinstance(data[0], list)
        ):

            vector = data[0]

        # Some responses may already be:
        #
        # [0.123, 0.456, ...]
        #

        elif (
            isinstance(data, list)
            and len(data) > 0
            and isinstance(
                data[0],
                (float, int)
            )
        ):

            vector = data

        else:

            raise RuntimeError(
                "Unexpected HuggingFace response: "
                f"{str(data)[:1000]}"
            )

        # ----------------------------------------------------
        # CONVERT TO FLOAT
        # ----------------------------------------------------

        vector = [
            float(value)
            for value in vector
        ]

        # ----------------------------------------------------
        # VERIFY DIMENSION
        # ----------------------------------------------------

        if len(vector) != 384:

            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected 384, "
                f"received {len(vector)}"
            )

        # ----------------------------------------------------
        # VERIFY VECTOR IS NOT ZERO
        # ----------------------------------------------------

        if not any(
            value != 0.0
            for value in vector
        ):

            raise RuntimeError(
                "HuggingFace returned an "
                "all-zero embedding."
            )

        print(
            "✅ Query embedding generated: "
            "384 dimensions"
        )

        return vector

    except httpx.TimeoutException as err:

        print(
            f"❌ HuggingFace timeout: {err}"
        )

        raise RuntimeError(
            "HuggingFace embedding request "
            "timed out."
        ) from err

    except httpx.HTTPError as err:

        print(
            f"❌ HuggingFace HTTP error: "
            f"{err}"
        )

        raise RuntimeError(
            "Could not connect to "
            "HuggingFace embedding service."
        ) from err

    except Exception as err:

        print(
            "❌ Embedding generation error: "
            f"{type(err).__name__}: {err}"
        )

        raise
