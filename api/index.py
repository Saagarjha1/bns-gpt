import os
import sys
import json
import time
import random
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import httpx
import jwt

from services.retriever import get_query_embedding_async


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
    PINECONE_IMPORT_ERROR = None

except Exception as e:
    PINECONE_AVAILABLE = False
    PINECONE_IMPORT_ERROR = str(e)


try:
    import sentry_sdk

    SENTRY_AVAILABLE = True

except Exception:
    SENTRY_AVAILABLE = False


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

# IMPORTANT:
# This must point to the new 563-vector index.
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "bns-legal-index-v2"
)

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
# SENTRY
# ==========================================

if SENTRY_AVAILABLE and SENTRY_DSN:
    try:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=1.0
        )
    except Exception as e:
        print(
            f"⚠️ Sentry initialization warning: {e}"
        )


# ==========================================
# FASTAPI
# ==========================================

app = FastAPI(
    title="Enterprise BNS Legal AI API",
    version="1.0.0"
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

PINECONE_ERROR = None
PINECONE_VECTOR_COUNT = None


def initialize_pinecone():

    global pc
    global index
    global PINECONE_ERROR
    global PINECONE_VECTOR_COUNT

    print("")
    print("=" * 60)
    print("        PINECONE INITIALIZATION")
    print("=" * 60)

    print(
        f"PINECONE_AVAILABLE: "
        f"{PINECONE_AVAILABLE}"
    )

    print(
        f"PINECONE_API_KEY present: "
        f"{bool(PINECONE_API_KEY)}"
    )

    print(
        f"PINECONE_INDEX_NAME: "
        f"{PINECONE_INDEX_NAME}"
    )

    # --------------------------------------
    # Check SDK
    # --------------------------------------

    if not PINECONE_AVAILABLE:

        PINECONE_ERROR = (
            "Pinecone Python package is not available."
        )

        print("❌ " + PINECONE_ERROR)

        if PINECONE_IMPORT_ERROR:
            print(
                f"Import error: "
                f"{PINECONE_IMPORT_ERROR}"
            )

        return

    # --------------------------------------
    # Check API key
    # --------------------------------------

    if not PINECONE_API_KEY:

        PINECONE_ERROR = (
            "PINECONE_API_KEY is missing."
        )

        print("❌ " + PINECONE_ERROR)

        print(
            "Add PINECONE_API_KEY to Render "
            "Environment Variables."
        )

        return

    # --------------------------------------
    # Connect
    # --------------------------------------

    try:

        pc = Pinecone(
            api_key=PINECONE_API_KEY
        )

        print("✅ Pinecone client created")

        # ----------------------------------
        # Verify indexes
        # ----------------------------------

        index_list = pc.list_indexes()

        available_indexes = []

        try:

            for item in index_list:

                if isinstance(item, dict):

                    name = item.get("name")

                else:

                    name = getattr(
                        item,
                        "name",
                        None
                    )

                if name:
                    available_indexes.append(name)

        except Exception as e:

            print(
                f"⚠️ Could not parse index list: "
                f"{e}"
            )

        print(
            f"Available Pinecone indexes: "
            f"{available_indexes}"
        )

        if (
            available_indexes
            and PINECONE_INDEX_NAME
            not in available_indexes
        ):

            raise RuntimeError(
                f"Index "
                f"'{PINECONE_INDEX_NAME}' "
                f"was not found. "
                f"Available indexes: "
                f"{available_indexes}"
            )

        # ----------------------------------
        # Connect to index
        # ----------------------------------

        index = pc.Index(
            PINECONE_INDEX_NAME
        )

        print(
            f"✅ Connected to index: "
            f"{PINECONE_INDEX_NAME}"
        )

        # ----------------------------------
        # Test connection
        # ----------------------------------

        stats = index.describe_index_stats()

        if isinstance(stats, dict):

            PINECONE_VECTOR_COUNT = (
                stats.get(
                    "total_vector_count"
                )
            )

        else:

            PINECONE_VECTOR_COUNT = getattr(
                stats,
                "total_vector_count",
                None
            )

        print(
            "✅ Pinecone connection test "
            "successful"
        )

        print(
            f"📊 Vector count: "
            f"{PINECONE_VECTOR_COUNT}"
        )

        PINECONE_ERROR = None

    except Exception as e:

        PINECONE_ERROR = str(e)

        pc = None
        index = None

        print("")
        print(
            "❌ PINECONE INITIALIZATION FAILED"
        )

        print(
            f"Error type: "
            f"{type(e).__name__}"
        )

        print(
            f"Error: {e}"
        )

    print("=" * 60)
    print("")


# Initialize Pinecone
initialize_pinecone()


# ==========================================
# SCHEMAS
# ==========================================

class QueryRequest(BaseModel):
    query: str


# ==========================================
# HELPERS
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


def create_jwt_token(
    ip: str
) -> str:

    now = int(time.time())

    payload = {
        "ip": ip,
        "iat": now,
        "exp": now + (24 * 3600)
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
            f"Bearer "
            f"{UPSTASH_REDIS_REST_TOKEN}"
    }

    try:

        async with httpx.AsyncClient(
            timeout=2.5
        ) as http_client:

            inc_res = await http_client.post(
                f"{UPSTASH_REDIS_REST_URL}"
                f"/incr/{key}",
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

            try:

                current_count = int(
                    raw_result
                )

            except Exception:

                current_count = 1

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
            f"⚠️ Rate limit warning: "
            f"{err}"
        )

        return True, MAX_QUERIES_PER_DAY


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/")
@app.get("/health")
@app.get("/api/health")
def health_check():

    pinecone_ok = index is not None

    return {

        "status": (
            "healthy"
            if pinecone_ok
            else "degraded"
        ),

        "pinecone_available":
            PINECONE_AVAILABLE,

        "pinecone_api_key_present":
            bool(PINECONE_API_KEY),

        "pinecone_connected":
            pinecone_ok,

        "pinecone_index":
            PINECONE_INDEX_NAME,

        "pinecone_vector_count":
            PINECONE_VECTOR_COUNT,

        "pinecone_error":
            PINECONE_ERROR,

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
# PINECONE SEARCH
# ==========================================

def get_pinecone_matches(
    query_vector
):

    if not index:

        return []

    # --------------------------------------
    # Validate dimension
    # --------------------------------------

    if (
        not isinstance(
            query_vector,
            list
        )
        or len(query_vector) != 384
    ):

        raise ValueError(
            "Invalid query vector "
            "dimension: "
            f"expected 384, got "
            f"{len(query_vector) "
            if isinstance(
                query_vector,
                list
            )
            else 'invalid'}"
        )

    # --------------------------------------
    # Prevent zero-vector search
    # --------------------------------------

    if not any(
        float(value) != 0.0
        for value in query_vector
    ):

        raise ValueError(
            "Query embedding is all zeros."
        )

    try:

        result = index.query(
            vector=query_vector,
            top_k=10,
            include_metadata=True
        )

        if hasattr(
            result,
            "matches"
        ):

            return result.matches

        if isinstance(
            result,
            dict
        ):

            return result.get(
                "matches",
                []
            )

        return []

    except Exception as e:

        print(
            f"❌ Pinecone query error: "
            f"{type(e).__name__}: {e}"
        )

        raise


# ==========================================
# EXTRACT MATCH DATA
# ==========================================

def extract_match_data(
    match
):

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

        metadata = match.get(
            "metadata",
            {}
        )

    else:

        metadata = {}

    if not isinstance(
        metadata,
        dict
    ):

        metadata = {}

    if hasattr(
        match,
        "score"
    ):

        score = match.score

    elif isinstance(
        match,
        dict
    ):

        score = match.get(
            "score"
        )

    else:

        score = None

    if hasattr(
        match,
        "id"
    ):

        match_id = match.id

    elif isinstance(
        match,
        dict
    ):

        match_id = match.get(
            "id"
        )

    else:

        match_id = None

    return (
        match_id,
        score,
        metadata
    )


# ==========================================
# STREAMING QUERY
# ==========================================

@app.post("/query/stream")
@app.post("/api/query/stream")
async def query_stream(
    req: QueryRequest,
    request: Request,
    authorization: Optional[str] = Header(None)
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
                detail=(
                    "Query string "
                    "cannot be empty."
                )
            )

        # ----------------------------------
        # Client IP
        # ----------------------------------

        client_ip = get_client_ip(
            request
        )

        # ----------------------------------
        # JWT
        # ----------------------------------

        if (
            authorization
            and authorization.startswith(
                "Bearer "
            )
        ):

            token = authorization.split(
                " ",
                1
            )[1]

            decoded = verify_jwt_token(
                token
            )

            if (
                decoded
                and "ip" in decoded
            ):

                client_ip = decoded["ip"]

        # ----------------------------------
        # Rate limit
        # ----------------------------------

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

        # ==================================
        # SSE GENERATOR
        # ==================================

        async def event_generator():

            # --------------------------------
            # Initial status
            # --------------------------------

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

            # --------------------------------
            # Candidate documents
            # --------------------------------

            candidate_docs = []

            # =================================
            # PINECONE RETRIEVAL
            # =================================

            if not index:

                yield (
                    "data: "
                    + json.dumps({
                        "chunk":
                            "Pinecone is not "
                            "connected."
                    })
                    + "\n\n"
                )

                yield "data: [DONE]\n\n"

                return

            try:

                # --------------------------------
                # Generate 384-dimensional query
                # embedding using MiniLM.
                # --------------------------------

                print("")
                print("=" * 60)
                print(
                    "🔎 GENERATING QUERY EMBEDDING"
                )
                print("=" * 60)

                query_vector = (
                    await get_query_embedding_async(
                        user_query
                    )
                )

                print(
                    f"✅ Query embedding "
                    f"dimension: "
                    f"{len(query_vector)}"
                )

                # --------------------------------
                # Validate vector
                # --------------------------------

                if (
                    not isinstance(
                        query_vector,
                        list
                    )
                    or len(query_vector) != 384
                ):

                    raise RuntimeError(
                        "Wrong embedding dimension: "
                        f"expected 384, got "
                        f"{len(query_vector)}"
                    )

                if not any(
                    float(value) != 0.0
                    for value in query_vector
                ):

                    raise RuntimeError(
                        "Embedding service "
                        "returned an all-zero "
                        "vector."
                    )

                # --------------------------------
                # Pinecone query
                # --------------------------------

                loop = (
                    asyncio.get_running_loop()
                )

                matches = (
                    await loop.run_in_executor(
                        None,
                        get_pinecone_matches,
                        query_vector
                    )
                )

                print("")
                print("=" * 60)
                print(
                    "🔎 PINECONE SEARCH RESULTS"
                )
                print("=" * 60)

                # --------------------------------
                # Inspect all 10 results
                # --------------------------------

                for rank, match in enumerate(
                    matches[:10],
                    start=1
                ):

                    (
                        match_id,
                        score,
                        metadata
                    ) = extract_match_data(
                        match
                    )

                    text = (
                        metadata.get(
                            "text",
                            ""
                        )
                        if isinstance(
                            metadata,
                            dict
                        )
                        else ""
                    )

                    print(
                        f"\n#{rank} "
                        f"| score={score} "
                        f"| id={match_id}"
                    )

                    print(
                        text[:700]
                    )

                # --------------------------------
                # Build candidate documents
                # --------------------------------

                for match in matches[:10]:

                    (
                        match_id,
                        score,
                        metadata
                    ) = extract_match_data(
                        match
                    )

                    if (
                        isinstance(
                            metadata,
                            dict
                        )
                        and metadata.get("text")
                    ):

                        candidate_docs.append({

                            "id":
                                match_id,

                            "score":
                                score,

                            "text":
                                metadata["text"],

                            "prompt":
                                metadata.get(
                                    "prompt",
                                    ""
                                ),

                            "response":
                                metadata.get(
                                    "response",
                                    ""
                                )
                        })

                print("")
                print(
                    f"📚 Candidate documents: "
                    f"{len(candidate_docs)}"
                )

            except Exception as vector_err:

                print("")
                print(
                    "❌ VECTOR SEARCH FAILED"
                )

                print(
                    f"Error type: "
                    f"{type(vector_err).__name__}"
                )

                print(
                    f"Error: {vector_err}"
                )

                # IMPORTANT:
                # Do not silently send an empty
                # context to Groq.
                yield (
                    "data: "
                    + json.dumps({
                        "chunk":
                            "I could not retrieve "
                            "the relevant BNS "
                            "statutory provisions "
                            "from the legal database. "
                            "Please try the query again."
                    })
                    + "\n\n"
                )

                yield "data: [DONE]\n\n"

                return

            # =================================
            # CONTEXT SELECTION
            # =================================

            # IMPORTANT:
            # Pinecone returns 10.
            # We now pass TOP 5 to Groq.
            contexts = [
                doc["text"]
                for doc
                in candidate_docs[:5]
            ]

            print("")
            print(
                f"📖 Context documents sent "
                f"to Groq: {len(contexts)}"
            )

            # --------------------------------
            # Show context selection
            # --------------------------------

            for i, context in enumerate(
                contexts,
                start=1
            ):

                print(
                    f"\n--- CONTEXT {i} ---"
                )

                print(
                    context[:700]
                )

            yield (
                "data: "
                + json.dumps({
                    "status":
                        "Generating legal analysis..."
                })
                + "\n\n"
            )

            # =================================
            # GROQ KEY
            # =================================

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
                    "`GROQ_API_KEY` is not "
                    "set in Environment Variables."
                )

                yield (
                    "data: "
                    + json.dumps({
                        "chunk":
                            fallback_text
                    })
                    + "\n\n"
                )

                yield "data: [DONE]\n\n"

                return

            # =================================
            # LEGAL SYSTEM PROMPT
            # =================================

            system_prompt = (

                "You are an expert legal "
                "research assistant specializing "
                "in the Bharatiya Nyaya Sanhita "
                "(BNS), 2023.\n\n"

                "Answer the user's question "
                "ONLY using the supplied "
                "retrieved legal context.\n\n"

                "IMPORTANT RULES:\n"

                "1. Prefer the context that "
                "directly answers the user's "
                "question.\n"

                "2. If the context contains an "
                "IPC section and its corresponding "
                "BNS section, use that mapping "
                "for IPC-to-BNS questions.\n"

                "3. Cite the exact BNS section "
                "number appearing in the context.\n"

                "4. State the applicable "
                "punishment, imprisonment, fine, "
                "or community service when it "
                "appears in the retrieved "
                "statutory text.\n"

                "5. Do NOT invent section numbers.\n"

                "6. Do NOT guess statutory "
                "provisions from general knowledge.\n"

                "7. Do NOT say that information "
                "is missing if the supplied "
                "context actually contains "
                "the answer.\n"

                "8. If multiple retrieved "
                "provisions are relevant, "
                "identify the provision that "
                "most directly answers the query.\n"

                "9. Clearly distinguish IPC "
                "sections from BNS sections.\n\n"

                "Give a concise and structured "
                "legal answer."
            )

            # =================================
            # CONTEXT
            # =================================

            context_str = "\n\n".join(
                [
                    (
                        f"Source {i + 1}:\n"
                        f"{context}"
                    )
                    for i, context
                    in enumerate(contexts)
                ]
            )

            if not context_str:

                context_str = (
                    "No relevant statutory "
                    "provisions were retrieved "
                    "from the vector index."
                )

            full_user_prompt = (
                "RETRIEVED LEGAL CONTEXT:\n\n"
                f"{context_str}\n\n"
                "USER LEGAL QUERY:\n"
                f"{user_query}\n\n"
                "Answer the query using the "
                "retrieved context above."
            )

            # =================================
            # GROQ REQUEST
            # =================================

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
                        0.1,

                    "max_tokens":
                        600
                }

                async with httpx.AsyncClient(
                    timeout=60.0
                ) as client:

                    async with client.stream(
                        "POST",
                        (
                            "https://api.groq.com/"
                            "openai/v1/chat/completions"
                        ),
                        headers=headers,
                        json=payload
                    ) as response:

                        # --------------------------------
                        # Groq error
                        # --------------------------------

                        if (
                            response.status_code
                            != 200
                        ):

                            err_body = (
                                await response.aread()
                            )

                            error_text = (
                                err_body.decode(
                                    errors="replace"
                                )
                            )

                            print(
                                "❌ Groq API error:"
                            )

                            print(
                                f"HTTP "
                                f"{response.status_code}"
                            )

                            print(
                                error_text[:2000]
                            )

                            yield (
                                "data: "
                                + json.dumps({
                                    "chunk":
                                        "Groq API Error "
                                        f"({response.status_code}): "
                                        f"{error_text[:1000]}"
                                })
                                + "\n\n"
                            )

                            yield (
                                "data: [DONE]\n\n"
                            )

                            return

                        # --------------------------------
                        # Stream response
                        # --------------------------------

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

                            if (
                                data_str
                                == "[DONE]"
                            ):

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
                                    .get(
                                        "delta",
                                        {}
                                    )
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

                            except Exception as stream_err:

                                print(
                                    "⚠️ Stream "
                                    "parse warning: "
                                    f"{stream_err}"
                                )

                                continue

                # --------------------------------
                # Completed
                # --------------------------------

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

                print(
                    f"❌ Groq streaming error: "
                    f"{type(groq_err).__name__}: "
                    f"{groq_err}"
                )

                yield (
                    "data: "
                    + json.dumps({
                        "chunk":
                            "Streaming exception: "
                            f"{str(groq_err)}"
                    })
                    + "\n\n"
                )

                yield "data: [DONE]\n\n"

        # --------------------------------------
        # StreamingResponse
        # --------------------------------------

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
            f"❌ API error: "
            f"{type(top_ex).__name__}: "
            f"{top_ex}"
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
