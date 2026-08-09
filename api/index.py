from pathlib import Path

code = r'''import os
import sys
import json
import time
import random
import asyncio
import hashlib
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional

import httpx
import jwt

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ==========================================
# PATH CONFIGURATION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ==========================================
# OPTIONAL PINECONE INTEGRATION
# ==========================================

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except Exception as e:
    PINECONE_AVAILABLE = False
    print(f"WARNING: Pinecone package unavailable: {e}")


# ==========================================
# OPTIONAL SENTRY
# ==========================================

try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except Exception as e:
    SENTRY_AVAILABLE = False
    print(f"WARNING: Sentry unavailable: {e}")


# ==========================================
# ENVIRONMENT & CONFIGURATION
# ==========================================

GROQ_KEYS = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
]

VALID_GROQ_KEYS = [
    key.strip()
    for key in GROQ_KEYS
    if key and key.strip()
]


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Your actual Pinecone index.
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "bns-legal-index",
)

# Your Pinecone records are in the default namespace.
PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "default",
)

# Hugging Face token is optional for the public inference endpoint.
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

JWT_SECRET = (
    os.getenv("JWT_SECRET")
    or os.getenv("JWT_SECRET_KEY")
    or "super-secret-bns-key"
)

UPSTASH_REDIS_REST_URL = os.getenv(
    "UPSTASH_REDIS_REST_URL"
)

UPSTASH_REDIS_REST_TOKEN = os.getenv(
    "UPSTASH_REDIS_REST_TOKEN"
)

SENTRY_DSN = os.getenv("SENTRY_DSN")


# ==========================================
# SENTRY INITIALIZATION
# ==========================================

if SENTRY_AVAILABLE and SENTRY_DSN:
    try:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=1.0,
        )
        print("Sentry initialized.")
    except Exception as e:
        print(f"WARNING: Sentry initialization failed: {e}")


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Enterprise BNS Legal AI API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# PINECONE INITIALIZATION
# ==========================================

pc = None
index = None
pinecone_init_error = None
pinecone_dimension = None
pinecone_record_count = None


if not PINECONE_AVAILABLE:
    pinecone_init_error = "Pinecone package is not available."

elif not PINECONE_API_KEY:
    pinecone_init_error = "PINECONE_API_KEY is missing."

else:
    try:
        print(
            f"Connecting to Pinecone index: "
            f"{PINECONE_INDEX_NAME}"
        )

        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Verify that the named index actually exists.
        index_info = pc.describe_index(
            PINECONE_INDEX_NAME
        )

        pinecone_dimension = getattr(
            index_info,
            "dimension",
            None,
        )

        index = pc.Index(PINECONE_INDEX_NAME)

        # Verify the index is reachable.
        stats = index.describe_index_stats()

        pinecone_record_count = getattr(
            stats,
            "total_vector_count",
            None,
        )

        if pinecone_record_count is None and isinstance(stats, dict):
            pinecone_record_count = stats.get(
                "total_vector_count"
            )

        print(
            f"Pinecone connected: "
            f"{PINECONE_INDEX_NAME}"
        )
        print(
            f"Pinecone dimension: "
            f"{pinecone_dimension}"
        )
        print(
            f"Pinecone records: "
            f"{pinecone_record_count}"
        )

    except Exception as e:
        pinecone_init_error = str(e)
        pc = None
        index = None

        print(
            f"Pinecone initialization error: {e}"
        )


# ==========================================
# SCHEMAS
# ==========================================

class QueryRequest(BaseModel):
    query: str


# ==========================================
# IP / JWT HELPERS
# ==========================================

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get(
        "x-forwarded-for"
    )

    if forwarded:
        return forwarded.split(",")[0].strip()

    return (
        request.client.host
        if request.client
        else "127.0.0.1"
    )


def create_jwt_token(ip: str) -> str:
    payload = {
        "ip": ip,
        "iat": int(time.time()),
        "exp": int(time.time()) + (24 * 3600),
    }

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm="HS256",
    )


def verify_jwt_token(
    token: str,
) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
        )
    except Exception:
        return None


# ==========================================
# EMBEDDING GENERATION
#
# Uses the same model referenced by the
# ingestion pipeline:
# sentence-transformers/all-MiniLM-L6-v2
#
# Expected dimension: 384.
# ==========================================

async def get_embedding(
    text: str,
) -> list[float]:

    text = (text or "").strip()

    if not text:
        raise ValueError(
            "Cannot embed an empty query."
        )

    url = (
        "https://api-inference.huggingface.co/"
        "pipeline/feature-extraction/"
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    headers = {}

    if HF_API_TOKEN:
        headers["Authorization"] = (
            f"Bearer {HF_API_TOKEN}"
        )

    async with httpx.AsyncClient(
        timeout=20.0
    ) as client:

        response = await client.post(
            url,
            headers=headers,
            json={
                "inputs": text,
                "options": {
                    "wait_for_model": True
                },
            },
        )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError(
            "Unexpected Hugging Face embedding response."
        )

    vector = (
        data[0]
        if isinstance(data[0], list)
        else data
    )

    vector = [
        float(value)
        for value in vector
    ]

    if len(vector) != 384:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"expected 384, got {len(vector)}."
        )

    # If Pinecone reported a dimension, verify it.
    if (
        pinecone_dimension is not None
        and int(pinecone_dimension) != len(vector)
    ):
        raise ValueError(
            "Pinecone/embedding dimension mismatch: "
            f"Pinecone={pinecone_dimension}, "
            f"embedding={len(vector)}."
        )

    return vector


# ==========================================
# RATE LIMITING
# ==========================================

MAX_QUERIES_PER_DAY = 20


async def check_rate_limit_async(
    client_ip: str,
) -> tuple[bool, int]:

    if (
        not UPSTASH_REDIS_REST_URL
        or not UPSTASH_REDIS_REST_TOKEN
    ):
        return True, MAX_QUERIES_PER_DAY

    key = (
        f"rate_limit:"
        f"{client_ip}:"
        f"{time.strftime('%Y-%m-%d')}"
    )

    headers = {
        "Authorization":
            f"Bearer {UPSTASH_REDIS_REST_TOKEN}"
    }

    try:
        async with httpx.AsyncClient(
            timeout=2.5
        ) as client:

            inc_res = await client.post(
                f"{UPSTASH_REDIS_REST_URL}/incr/{key}",
                headers=headers,
            )

            res_data = inc_res.json()

            raw_result = (
                res_data.get("result")
                if isinstance(
                    res_data,
                    dict,
                )
                else None
            )

            try:
                current_count = int(raw_result)
            except (TypeError, ValueError):
                current_count = 1

            if current_count == 1:
                await client.post(
                    f"{UPSTASH_REDIS_REST_URL}"
                    f"/expire/{key}/86400",
                    headers=headers,
                )

            remaining = max(
                0,
                MAX_QUERIES_PER_DAY
                - current_count,
            )

            if current_count > MAX_QUERIES_PER_DAY:
                return False, 0

            return True, remaining

    except Exception as err:
        print(
            f"WARNING: Rate limit error: {err}"
        )

        # Fail open so a Redis outage does not
        # take the API completely offline.
        return True, MAX_QUERIES_PER_DAY


# ==========================================
# HEALTH ENDPOINT
# ==========================================

@app.get("/")
@app.get("/health")
@app.get("/api/health")
def health_check():

    connected = False
    error = pinecone_init_error
    record_count = pinecone_record_count

    if index is not None:
        try:
            stats = index.describe_index_stats()

            connected = True
            error = None

            record_count = getattr(
                stats,
                "total_vector_count",
                None,
            )

            if record_count is None and isinstance(
                stats,
                dict,
            ):
                record_count = stats.get(
                    "total_vector_count"
                )

        except Exception as e:
            connected = False
            error = str(e)

    return {
        "status": "healthy",
        "pinecone_connected": connected,
        "pinecone_index": PINECONE_INDEX_NAME,
        "pinecone_namespace": PINECONE_NAMESPACE,
        "pinecone_dimension": pinecone_dimension,
        "pinecone_record_count": record_count,
        "pinecone_error": error,
        "valid_groq_keys_count": len(
            VALID_GROQ_KEYS
        ),
    }


# ==========================================
# AUTH TOKEN
# ==========================================

@app.get("/auth/token")
@app.get("/api/auth/token")
def get_auth_token(
    request: Request,
):
    client_ip = get_client_ip(request)

    token = create_jwt_token(client_ip)

    return {
        "token": token,
        "ip": client_ip,
    }


# ==========================================
# QUERY STREAM
# ==========================================

@app.post("/query/stream")
@app.post("/api/query/stream")
async def query_stream(
    req: QueryRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
):

    try:
        user_query = (
            req.query.strip()
            if req and req.query
            else ""
        )

        if not user_query:
            raise HTTPException(
                status_code=400,
                detail="Query string cannot be empty.",
            )

        client_ip = get_client_ip(request)

        if (
            authorization
            and authorization.startswith("Bearer ")
        ):
            token = authorization.split(
                " ",
                1,
            )[1]

            decoded = verify_jwt_token(
                token
            )

            if (
                decoded
                and "ip" in decoded
            ):
                client_ip = decoded["ip"]

        allowed, remaining = (
            await check_rate_limit_async(
                client_ip
            )
        )

        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Rate limit exceeded. "
                    f"Maximum "
                    f"{MAX_QUERIES_PER_DAY} "
                    "queries allowed per day."
                ),
            )

        async def event_generator(
        ) -> AsyncGenerator[str, None]:

            yield (
                "data: "
                + json.dumps({
                    "status": (
                        "Searching database... | "
                        f"{remaining}/"
                        f"{MAX_QUERIES_PER_DAY} "
                        "queries remaining"
                    )
                })
                + "\n\n"
            )

            await asyncio.sleep(0.01)

            candidate_docs = []

            # ======================================
            # PINECONE SEMANTIC SEARCH
            # ======================================

            if (
                index is not None
                and PINECONE_API_KEY
            ):
                try:
                    print(
                        "Generating query embedding..."
                    )

                    query_vector = (
                        await get_embedding(
                            user_query
                        )
                    )

                    print(
                        "Query embedding dimension: "
                        f"{len(query_vector)}"
                    )

                    loop = (
                        asyncio.get_running_loop()
                    )

                    query_res = (
                        await loop.run_in_executor(
                            None,
                            lambda: index.query(
                                vector=query_vector,
                                top_k=5,
                                include_metadata=True,
                                namespace=(
                                    PINECONE_NAMESPACE
                                ),
                            ),
                        )
                    )

                    if hasattr(
                        query_res,
                        "matches",
                    ):
                        matches = (
                            query_res.matches
                        )
                    elif isinstance(
                        query_res,
                        dict,
                    ):
                        matches = (
                            query_res.get(
                                "matches",
                                [],
                            )
                        )
                    else:
                        matches = []

                    print(
                        "Pinecone matches: "
                        f"{len(matches)}"
                    )

                    for match in matches:

                        if hasattr(
                            match,
                            "metadata",
                        ):
                            metadata = match.metadata
                        elif isinstance(
                            match,
                            dict,
                        ):
                            metadata = match.get(
                                "metadata"
                            )
                        else:
                            metadata = None

                        if (
                            isinstance(
                                metadata,
                                dict,
                            )
                            and metadata.get("text")
                        ):
                            candidate_docs.append({
                                "text": metadata["text"]
                            })

                except Exception as vector_err:
                    print(
                        "WARNING: Pinecone search failed: "
                        f"{vector_err}"
                    )

            else:
                print(
                    "WARNING: Pinecone is unavailable."
                )

            # ======================================
            # CONTEXT
            # ======================================

            contexts = [
                doc["text"]
                for doc in candidate_docs[:5]
            ]

            if contexts:
                context_str = "\n\n".join(
                    contexts
                )
            else:
                context_str = (
                    "No relevant statutory provisions "
                    "were retrieved from the vector index."
                )

            yield (
                "data: "
                + json.dumps({
                    "status": (
                        f"Retrieved {len(contexts)} "
                        "legal documents. "
                        "Generating legal analysis..."
                    )
                })
                + "\n\n"
            )

            # ======================================
            # GROQ
            # ======================================

            selected_groq_key = (
                random.choice(
                    VALID_GROQ_KEYS
                )
                if VALID_GROQ_KEYS
                else None
            )

            if not selected_groq_key:
                fallback_text = (
                    f"### Query: {user_query}\n\n"
                    "**Configuration Required:** "
                    "`GROQ_API_KEY` is not set in "
                    "Environment Variables.\n\n"
                    "Please add `GROQ_API_KEY` in "
                    "your Render dashboard."
                )

                yield (
                    "data: "
                    + json.dumps({
                        "chunk": fallback_text
                    })
                    + "\n\n"
                )

                yield "data: [DONE]\n\n"
                return

            system_prompt = (
                "You are an expert legal assistant "
                "specializing in the Bharatiya Nyaya "
                "Sanhita (BNS), 2023. "
                "Provide precise, structured, and "
                "legally accurate answers based on "
                "the retrieved statutory context. "
                "Cite relevant BNS section numbers "
                "where applicable. "
                "Do not invent statutory sections, "
                "penalties, or facts. "
                "If the retrieved context is "
                "insufficient, clearly say so."
            )

            full_user_prompt = (
                f"Retrieved legal context:\n"
                f"{context_str}\n\n"
                f"User Legal Query:\n"
                f"{user_query}"
            )

            try:
                headers = {
                    "Authorization":
                        f"Bearer {selected_groq_key}",
                    "Content-Type":
                        "application/json",
                }

                payload = {
                    "model":
                        "llama-3.3-70b-versatile",
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": full_user_prompt,
                        },
                    ],
                    "stream": True,
                    "temperature": 0.2,
                }

                async with httpx.AsyncClient(
                    timeout=30.0
                ) as client:

                    async with client.stream(
                        "POST",
                        "https://api.groq.com/"
                        "openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:

                        if response.status_code != 200:

                            err_body = (
                                await response.aread()
                            )

                            error_text = (
                                err_body.decode(
                                    errors="replace"
                                )
                            )

                            yield (
                                "data: "
                                + json.dumps({
                                    "chunk": (
                                        "Groq API Error "
                                        f"({response.status_code}): "
                                        f"{error_text}"
                                    )
                                })
                                + "\n\n"
                            )

                            yield "data: [DONE]\n\n"
                            return

                        async for line in (
                            response.aiter_lines()
                        ):

                            if not line.startswith(
                                "data: "
                            ):
                                continue

                            data_str = (
                                line[6:].strip()
                            )

                            if data_str == "[DONE]":
                                break

                            try:
                                data_json = json.loads(
                                    data_str
                                )

                                choices = data_json.get(
                                    "choices",
                                    [],
                                )

                                if not choices:
                                    continue

                                delta = (
                                    choices[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )

                                if delta:
                                    yield (
                                        "data: "
                                        + json.dumps({
                                            "chunk": delta
                                        })
                                        + "\n\n"
                                    )

                            except Exception:
                                continue

                yield (
                    "data: "
                    + json.dumps({
                        "status": (
                            "Completed | "
                            f"{remaining}/"
                            f"{MAX_QUERIES_PER_DAY} "
                            "queries remaining"
                        )
                    })
                    + "\n\n"
                )

                yield "data: [DONE]\n\n"

            except Exception as groq_err:
                yield (
                    "data: "
                    + json.dumps({
                        "chunk": (
                            "Streaming exception: "
                            f"{str(groq_err)}"
                        )
                    })
                    + "\n\n"
                )

                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except HTTPException as http_ex:
        raise http_ex

    except Exception as top_ex:
        print(
            f"Query endpoint error: {top_ex}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Internal Server Error: "
                f"{str(top_ex)}"
            ),
        )


# ==========================================
# CATCH-ALL
# ==========================================

@app.api_route(
    "/{path_name:path}",
    methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
    ],
)
async def catch_all(
    request: Request,
    path_name: str,
):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Route not found",
            "requested_path": path_name,
        },
    )
'''

path = Path("/mnt/data/api_index_updated.py")
path.write_text(code, encoding="utf-8")
print(f"Created: {path}")
print(f"Lines: {len(code.splitlines())}")
