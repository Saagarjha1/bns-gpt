import os
import json
import time
import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import jwt

# Optional Pinecone integration
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# Optional Sentry integration
try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False


# ==========================================
# ENVIRONMENT & CONFIGURATION
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bns-legal")
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-bns-key-change-in-prod")
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
SENTRY_DSN = os.getenv("SENTRY_DSN")

if SENTRY_AVAILABLE and SENTRY_DSN:
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=1.0)

app = FastAPI(title="Enterprise BNS Legal AI API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# CLIENT INITIALIZATIONS
# ==========================================
pc = None
index = None

if PINECONE_AVAILABLE and PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"⚠️ Pinecone initialization error: {e}")


# ==========================================
# SCHEMAS & HELPERS
# ==========================================
class QueryRequest(BaseModel):
    query: str


def get_client_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def create_jwt_token(ip: str) -> str:
    payload = {
        "ip": ip,
        "iat": int(time.time()),
        "exp": int(time.time()) + (24 * 3600)  # 24 hours
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


# ==========================================
# RATE LIMITING LOGIC (Upstash / Memory)
# ==========================================
MAX_QUERIES_PER_DAY = 20

def check_rate_limit(client_ip: str) -> tuple[bool, int]:
    """
    Checks query count against Upstash Redis or falls back to allow.
    Returns (is_allowed, remaining_queries).
    """
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        return True, MAX_QUERIES_PER_DAY

    key = f"rate_limit:{client_ip}:{time.strftime('%Y-%m-%d')}"
    headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}
    
    try:
        # Increment counter in Redis
        inc_res = requests.post(
            f"{UPSTASH_REDIS_REST_URL}/incr/{key}",
            headers=headers,
            timeout=3
        ).json()
        
        current_count = inc_res.get("result", 1)
        
        # Set 24h expiration on first request
        if current_count == 1:
            requests.post(
                f"{UPSTASH_REDIS_REST_URL}/expire/{key}/86400",
                headers=headers,
                timeout=3
            )
            
        remaining = max(0, MAX_QUERIES_PER_DAY - current_count)
        if current_count > MAX_QUERIES_PER_DAY:
            return False, 0
        return True, remaining
    except Exception as err:
        print(f"⚠️ Rate limit store error: {err}")
        return True, MAX_QUERIES_PER_DAY


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "pinecone_connected": index is not None,
        "groq_configured": bool(GROQ_API_KEY)
    }


@app.get("/api/auth/token")
def get_auth_token(request: Request):
    client_ip = get_client_ip(request)
    token = create_jwt_token(client_ip)
    return {"token": token, "ip": client_ip}


@app.post("/api/query/stream")
async def query_stream(
    req: QueryRequest,
    request: Request,
    authorization: Optional[str] = Header(None)
):
    user_query = req.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    # 1. Verify Authentication / IP
    client_ip = get_client_ip(request)
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        decoded = verify_jwt_token(token)
        if decoded and "ip" in decoded:
            client_ip = decoded["ip"]

    # 2. Check Rate Limit
    allowed, remaining = check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {MAX_QUERIES_PER_DAY} queries allowed per day."
        )

    # 3. Stream Generator
    async def event_generator() -> AsyncGenerator[str, None]:
        # Emits initial status & remaining quota
        yield f"data: {json.dumps({'status': f'Searching database... | {remaining}/{MAX_QUERIES_PER_DAY} queries remaining'})}\n\n"
        await asyncio.sleep(0.05)

        candidate_docs = []

        # Step A: Pinecone Vector Retrieval (if configured)
        if index and PINECONE_API_KEY:
            try:
                # Dense retrieval or dummy search structure
                # Adjust namespaces/queries according to your embedding pipeline setup
                query_res = index.query(
                    vector=[0.0] * 1536,  # Placeholder if using integrated Pinecone inference
                    top_k=5,
                    include_metadata=True
                )
                for match in query_res.get("matches", []):
                    if "metadata" in match and "text" in match["metadata"]:
                        candidate_docs.append({"text": match["metadata"]["text"]})
            except Exception as vector_err:
                print(f"⚠️ Pinecone vector query error: {vector_err}")

        # Step B: Safe Reranker Execution
        contexts = []
        if candidate_docs:
            try:
                # Explicit runtime check to prevent AttributeError on Pinecone SDK
                if pc and hasattr(pc, "inference") and hasattr(pc.inference, "rerank"):
                    rerank_resp = pc.inference.rerank(
                        model="bge-reranker-v2-m3",
                        query=user_query,
                        documents=candidate_docs,
                        top_n=3,
                        return_documents=True
                    )
                    contexts = [item.document["text"] for item in rerank_resp.data]
                else:
                    contexts = [doc["text"] for doc in candidate_docs[:3]]
            except Exception as rerank_err:
                print(f"⚠️ Reranker failed or not supported: {rerank_err}. Using vector ranking.")
                contexts = [doc["text"] for doc in candidate_docs[:3]]

        yield f"data: {json.dumps({'status': 'Generating legal analysis...'})}\n\n"

        # Step C: Groq LLM Inference & Streaming Response
        if not GROQ_API_KEY:
            # Fallback output if no Groq Key present
            fallback_text = (
                f"### Query: {user_query}\n\n"
                "**Note:** `GROQ_API_KEY` is missing in server environment.\n\n"
                "**Relevant Provisions Found:**\n" + 
                ("\n".join([f"- {c}" for c in contexts]) if contexts else "No documents returned.")
            )
            yield f"data: {json.dumps({'text': fallback_text})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Prepare System & User Prompt
        system_prompt = (
            "You are an expert legal assistant specializing in the Bharatiya Nyaya Sanhita (BNS), 2023. "
            "Provide precise, structured, and legally accurate answers based on statutory provisions. "
            "Cite relevant BNS section numbers where applicable."
        )
        
        context_str = "\n\n".join(contexts) if contexts else "No relevant statutory provisions retrieved."
        full_user_prompt = f"Context:\n{context_str}\n\nUser Legal Query: {user_query}"

        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_prompt}
                ],
                "stream": True,
                "temperature": 0.2
            }

            groq_resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=30
            )

            if groq_resp.status_code != 200:
                err_msg = f"LLM Provider Error ({groq_resp.status_code})"
                yield f"data: {json.dumps({'error': err_msg})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Stream chunks back to client
            for line in groq_resp.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            delta = data_json["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield f"data: {json.dumps({'chunk': delta})}\n\n"
                        except Exception:
                            continue

            yield f"data: {json.dumps({'status': f'Completed | {remaining}/{MAX_QUERIES_PER_DAY} queries remaining'})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as groq_err:
            print(f"⚠️ Groq Streaming Exception: {groq_err}")
            yield f"data: {json.dumps({'error': f'Streaming exception: {str(groq_err)}'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
