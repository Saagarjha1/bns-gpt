import os
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
import requests
import numpy as np

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
# OPTIONAL INTEGRATIONS
# ==========================================

try:
    from pinecone import Pinecone

    PINECONE_AVAILABLE = True

except Exception as e:
    PINECONE_AVAILABLE = False
    print(f"⚠️ Pinecone package unavailable: {e}")


try:
    import sentry_sdk

    SENTRY_AVAILABLE = True

except Exception as e:
    SENTRY_AVAILABLE = False
    print(f"⚠️ Sentry unavailable: {e}")


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
    k.strip()
    for k in GROQ_KEYS
    if k and k.strip()
]


# ------------------------------------------------
# PINECONE
# ------------------------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Your actual Pinecone index
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "bns-legal-index"
)

# Your Pinecone records are in this namespace
PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "default"
)


# ------------------------------------------------
# HUGGING FACE
# ------------------------------------------------

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")


# ------------------------------------------------
# JWT
# ------------------------------------------------

JWT_SECRET = (
    os.getenv("JWT_SECRET")
    or os.getenv("JWT_SECRET_KEY")
    or "super-secret-bns-key"
)


# ------------------------------------------------
# UPSTASH REDIS
# ------------------------------------------------

UPSTASH_REDIS_REST_URL = os.getenv(
    "UPSTASH_REDIS_REST_URL"
)

UPSTASH_REDIS_REST_TOKEN = os.getenv(
    "UPSTASH_REDIS_REST_TOKEN"
)


# ------------------------------------------------
# SENTRY
# ------------------------------------------------

SENTRY_DSN = os.getenv("SENTRY_DSN")


# ==========================================
# SENTRY INITIALIZATION
# ==========================================

if SENTRY_AVAILABLE and SENTRY_DSN:

    try:

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=1.0
        )

        print("✅ Sentry initialized")

    except Exception as e:

        print(f"⚠️ Sentry initialization failed: {e}")


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Enterprise BNS Legal AI API",
    version="1.0.0"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# PINECONE CLIENT INITIALIZATION
# ==========================================

pc = None
index = None
pinecone_init_error = None


if not PINECONE_AVAILABLE:

    pinecone_init_error = (
        "Pinecone Python package is not available."
    )

    print(
        "❌ Pinecone initialization skipped: "
        "package unavailable."
    )


elif not PINECONE_API_KEY:

    pinecone_init_error = (
        "PINECONE_API_KEY environment variable is missing."
    )

    print(
        "❌ Pinecone initialization skipped: "
        "PINECONE_API_KEY is missing."
    )


else:

    try:

        print(
            f"🔌 Connecting to Pinecone index: "
            f"{PINECONE_INDEX_NAME}"
        )

        pc = Pinecone(
            api_key=PINECONE_API_KEY
        )

        index = pc.Index(
            PINECONE_INDEX_NAME
        )

        # Actually verify the index connection
        # instead of merely creating the client.
        stats = index.describe_index_stats()

        print(
            f"✅ Pinecone connected successfully: "
            f"{PINECONE_INDEX_NAME}"
        )

        print(
            f"📊 Pinecone stats: {stats}"
        )

    except Exception as e:

        pinecone_init_error = str(e)

        pc = None
        index = None

        print(
            "❌ Pinecone initialization error:"
        )

        print(
            f"   {e}"
        )


# ==========================================
# EMBEDDING GENERATION
#
# Must match the 384-dimensional
# all-MiniLM-L6-v2 vectors used by
# the ingestion pipeline.
# ==========================================

def get_embedding(text: str) -> list[float]:

    text = (text or "").strip()

    if not text:

        raise ValueError(
            "Cannot generate embedding for empty text."
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


    try:

        response = requests.post(
            url,
            headers=headers,
            json={
                "inputs": text,
                "options": {
                    "wait_for_model": True
                }
            },
            timeout=10
        )


        response.raise_for_status()

        data = response.json()


        if isinstance(data, list) and data:

            # Hugging Face may return:
            #
            # [0.1, 0.2, ...]
            #
            # or:
            #
            # [[0.1, 0.2, ...]]

            if isinstance(data[0], list):

                vector = data[0]

            else:

                vector = data


            vector = [
                float(x)
                for x in vector
            ]


            if len(vector) != 384:

                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"expected 384, got {len(vector)}"
                )


            return vector


        raise ValueError(
            "Unexpected embedding response format."
        )


    except Exception as e:

        print(
            f"⚠️ Hugging Face embedding error: {e}"
        )

        print(
            "⚠️ Using deterministic 384-dimensional "
            "fallback vector."
        )


        # ------------------------------------------
        # Deterministic 384-dimensional fallback
        # ------------------------------------------

        seed = (
            int(
                hashlib.md5(
                    text.encode("utf-8")
                ).hexdigest(),
                16
            )
            % (2 ** 32)
        )


        rng = np.random.default_rng(seed)


        vector = rng.normal(
            0,
            1,
            384
        )


        norm = np.linalg.norm(vector)


        if norm != 0:

            vector = vector / norm


        return vector.tolist()


# ==========================================
# SCHEMAS
# ==========================================

class QueryRequest(BaseModel):

    query: str


# ==========================================
# IP HELPER
# ==========================================

def get_client_ip(
    request: Request
) -> str:

    x_forwarded_for = request.headers.get(
        "x-forwarded-for"
    )


    if x_forwarded_for:

        return (
            x_forwarded_for
            .split(",")[0]
            .strip()
        )


    return (
        request.client.host
        if request.client
        else "127.0.0.1"
    )


# ==========================================
# JWT
# ==========================================

def create_jwt_token(
    ip: str
) -> str:

    payload = {

        "ip": ip,

        "iat": int(time.time()),

        "exp": int(time.time()) + (
            24 * 3600
        )
    }


    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm="HS256"
    )


def verify_jwt_token(
    token: str
) -> Optional[Dict[str, Any]]:

    try:

        return jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"]
        )

    except Exception:

        return None


# ==========================================
# RATE LIMITING
# ==========================================

MAX_QUERIES_PER_DAY = 20


async def check_rate_limit_async(
    client_ip: str
) -> tuple[bool, int]:

    # If Redis isn't configured,
    # allow requests without rate limiting.

    if (
        not UPSTASH_REDIS_REST_URL
        or not UPSTASH_REDIS_REST_TOKEN
    ):

        return (
            True,
            MAX_QUERIES_PER_DAY
        )


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
        ) as http_client:

            inc_res = await http_client.post(
                f"{UPSTASH_REDIS_REST_URL}/incr/{key}",
                headers=headers
            )


            res_data = inc_res.json()


            raw_result = (
                res_data.get("result")
                if isinstance(
                    res_data,
                    dict
                )
                else None
            )


            current_count = (
                int(raw_result)
                if (
                    raw_result is not None
                    and str(raw_result).isdigit()
                )
                else 1
            )


            if current_count == 1:

                await http_client.post(
                    f"{UPSTASH_REDIS_REST_URL}"
                    f"/expire/{key}/86400",
                    headers=headers
                )


            remaining = max(
                0,
                MAX_QUERIES_PER_DAY
                - current_count
            )


            if (
                current_count
                > MAX_QUERIES_PER_DAY
            ):

                return False, 0


            return True, remaining


    except Exception as err:

        print(
            f"⚠️ Rate limit warning: {err}"
        )


        # Fail open so Redis outage doesn't
        # completely take down the AI.

        return (
            True,
            MAX_QUERIES_PER_DAY
        )


# ==========================================
# HEALTH ENDPOINT
# ==========================================

@app.get("/")
@app.get("/health")
@app.get("/api/health")
def health_check():

    pinecone_connected = False
    pinecone_error = pinecone_init_error
    record_count = None


    if index is not None:

        try:

            stats = (
                index.describe_index_stats()
            )


            pinecone_connected = True


            # Try to extract total record count
            # from the Pinecone response.

            if hasattr(
                stats,
                "total_vector_count"
            ):

                record_count = (
                    stats.total_vector_count
                )

            elif isinstance(
                stats,
                dict
            ):

                record_count = (
                    stats.get(
                        "total_vector_count"
                    )
                )


            pinecone_error = None


        except Exception as e:

            pinecone_connected = False

            pinecone_error = str(e)


    return {

        "status": "healthy",

        "pinecone_connected":
            pinecone_connected,

        "pinecone_index":
            PINECONE_INDEX_NAME,

        "pinecone_namespace":
            PINECONE_NAMESPACE,

        "pinecone_record_count":
            record_count,

        "pinecone_error":
            pinecone_error,

        "valid_groq_keys_count":
            len(VALID_GROQ_KEYS)
    }


# ==========================================
# AUTH TOKEN
# ==========================================

@app.get("/auth/token")
@app.get("/api/auth/token")
def get_auth_token(
    request: Request
):

    client_ip = get_client_ip(
        request
    )


    token = create_jwt_token(
        client_ip
    )


    return {
        "token": token,
        "ip": client_ip
    }


# ==========================================
# QUERY STREAM
# ==========================================

@app.post("/query/stream")
@app.post("/api/query/stream")
async def query_stream(

    req: QueryRequest,

    request: Request,

    authorization: Optional[str] = Header(None)
):

    try:

        # ------------------------------------------
        # Validate query
        # ------------------------------------------

        user_query = (
            req.query.strip()
            if req and req.query
            else ""
        )


        if not user_query:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Query string cannot be empty."
                )
            )


        # ------------------------------------------
        # Identify client
        # ------------------------------------------

        client_ip = get_client_ip(
            request
        )


        if (
            authorization
            and authorization.startswith(
                "Bearer "
            )
        ):

            token = (
                authorization
                .split(" ", 1)[1]
            )


            decoded = verify_jwt_token(
                token
            )


            if (
                decoded
                and "ip" in decoded
            ):

                client_ip = decoded["ip"]


        # ------------------------------------------
        # Rate limit
        # ------------------------------------------

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
                )
            )


        # ==========================================
        # SSE EVENT GENERATOR
        # ==========================================

        async def event_generator():

            # --------------------------------------
            # Initial status
            # --------------------------------------

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
            # PINECONE SEARCH
            # ======================================

            if (
                index is not None
                and PINECONE_API_KEY
            ):

                try:

                    print(
                        "🔎 Generating embedding "
                        "for query..."
                    )


                    query_vector = (
                        get_embedding(
                            user_query
                        )
                    )


                    print(
                        "✅ Query embedding "
                        f"dimension: "
                        f"{len(query_vector)}"
                    )


                    # ----------------------------------
                    # Pinecone query runs in a thread
                    # because Pinecone SDK operation
                    # is synchronous.
                    # ----------------------------------

                    loop = (
                        asyncio.get_running_loop()
                    )


                    query_res = (
                        await loop.run_in_executor(

                            None,

                            lambda:
                                index.query(

                                    vector=query_vector,

                                    top_k=5,

                                    include_metadata=True,

                                    namespace=(
                                        PINECONE_NAMESPACE
                                    )
                                )
                        )
                    )


                    # ----------------------------------
                    # Extract matches
                    # ----------------------------------

                    matches = []


                    if hasattr(
                        query_res,
                        "matches"
                    ):

                        matches = (
                            query_res.matches
                        )


                    elif isinstance(
                        query_res,
                        dict
                    ):

                        matches = (
                            query_res.get(
                                "matches",
                                []
                            )
                        )


                    print(
                        f"🔎 Pinecone returned "
                        f"{len(matches)} matches"
                    )


                    for match in matches:

                        metadata = None


                        if hasattr(
                            match,
                            "metadata"
                        ):

                            metadata = (
                                match.metadata
                            )


                        elif isinstance(
                            match,
                            dict
                        ):

                            metadata = (
                                match.get(
                                    "metadata"
                                )
                            )


                        if (
                            metadata
                            and isinstance(
                                metadata,
                                dict
                            )
                            and metadata.get(
                                "text"
                            )
                        ):

                            candidate_docs.append({

                                "text":
                                    metadata[
                                        "text"
                                    ]
                            })


                except Exception as vector_err:

                    print(
                        "⚠️ Vector search "
                        f"warning: {vector_err}"
                    )


            else:

                print(
                    "⚠️ Pinecone unavailable. "
                    "Skipping vector search."
                )


            # ======================================
            # BUILD CONTEXT
            # ======================================

            contexts = [

                doc["text"]

                for doc
                in candidate_docs[:5]

            ]


            if contexts:

                context_str = (
                    "\n\n".join(contexts)
                )

            else:

                context_str = (
                    "No relevant statutory "
                    "provisions retrieved "
                    "from vector index."
                )


            print(
                f"📚 Context documents: "
                f"{len(contexts)}"
            )


            yield (
                "data: "
                + json.dumps({
                    "status": (
                        "Retrieved "
                        f"{len(contexts)} "
                        "relevant legal documents. "
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

                    f"### Query: "
                    f"{user_query}\n\n"

                    "**Configuration Required:** "
                    "`GROQ_API_KEY` is not set "
                    "in Environment Variables.\n\n"

                    "Please add `GROQ_API_KEY` "
                    "in your Render dashboard "
                    "environment settings."
                )


                yield (
                    "data: "
                    + json.dumps({
                        "chunk":
                            fallback_text
                    })
                    + "\n\n"
                )


                yield (
                    "data: [DONE]\n\n"
                )


                return


            # ======================================
            # LEGAL SYSTEM PROMPT
            # ======================================

            system_prompt = (

                "You are an expert legal assistant "
                "specializing in the Bharatiya "
                "Nyaya Sanhita (BNS), 2023. "

                "Provide precise, structured, "
                "and legally accurate answers "
                "based primarily on the retrieved "
                "statutory context. "

                "Cite relevant BNS section numbers "
                "where applicable. "

                "Do not invent statutory sections "
                "or penalties. "

                "If the retrieved context does not "
                "contain enough information, "
                "clearly say so."
            )


            full_user_prompt = (

                f"Retrieved legal context:\n"
                f"{context_str}\n\n"

                f"User Legal Query:\n"
                f"{user_query}"
            )


            # ======================================
            # GROQ STREAM
            # ======================================

            try:

                headers = {

                    "Authorization":
                        f"Bearer "
                        f"{selected_groq_key}",

                    "Content-Type":
                        "application/json"
                }


                payload = {

                    "model":
                        "llama-3.3-70b-versatile",

                    "messages": [

                        {
                            "role":
                                "system",

                            "content":
                                system_prompt
                        },

                        {
                            "role":
                                "user",

                            "content":
                                full_user_prompt
                        }

                    ],

                    "stream":
                        True,

                    "temperature":
                        0.2
                }


                async with httpx.AsyncClient(
                    timeout=30.0
                ) as client:

                    async with client.stream(

                        "POST",

                        "https://api.groq.com/"
                        "openai/v1/chat/completions",

                        headers=headers,

                        json=payload

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


                            yield (
                                "data: [DONE]\n\n"
                            )


                            return


                        # ----------------------------------
                        # Process SSE stream
                        # ----------------------------------

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

                                data_json = (
                                    json.loads(
                                        data_str
                                    )
                                )


                                choices = (
                                    data_json.get(
                                        "choices",
                                        []
                                    )
                                )


                                if not choices:

                                    continue


                                delta = (
                                    choices[0]
                                    .get("delta", {})
                                    .get(
                                        "content",
                                        ""
                                    )
                                )


                                if delta:

                                    yield (
                                        "data: "
                                        + json.dumps({
                                            "chunk":
                                                delta
                                        })
                                        + "\n\n"
                                    )


                            except Exception:

                                continue


                # --------------------------------------
                # Completed
                # --------------------------------------

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


                yield (
                    "data: [DONE]\n\n"
                )


            except Exception as groq_err:

                print(
                    "❌ Groq streaming error:"
                )

                print(
                    groq_err
                )


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


                yield (
                    "data: [DONE]\n\n"
                )


        # ==========================================
        # RETURN SSE RESPONSE
        # ==========================================

        return StreamingResponse(

            event_generator(),

            media_type="text/event-stream",

            headers={

                "Cache-Control":
                    "no-cache",

                "Connection":
                    "keep-alive",

                "X-Accel-Buffering":
                    "no"
            }
        )


    except HTTPException as http_ex:

        raise http_ex


    except Exception as top_ex:

        print(
            "❌ Query endpoint error:"
        )

        print(
            top_ex
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "Internal Server Error: "
                f"{str(top_ex)}"
            )
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
        "DELETE"
    ]
)
async def catch_all(
    request: Request,
    path_name: str
):

    return JSONResponse(

        status_code=404,

        content={
            "error":
                "Route not found",

            "requested_path":
                path_name
        }
    )
