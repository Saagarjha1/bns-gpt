import os
import sys
import math
import re
import json
import hashlib
import traceback
from collections import Counter
from typing import List, Optional
from datetime import datetime, timedelta, timezone

# Ensure project root is in sys.path BEFORE any local imports for Vercel
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import jwt
import requests
import numpy as np
from pinecone import Pinecone
from groq import Groq, RateLimitError
import sentry_sdk

# ==============================================================================
# 1. PURE-PYTHON BM25 SPARSE ENCODER
# ==============================================================================
class PureBM25Encoder:
    def __init__(self, filepath: str = "bm25_params.json"):
        self.k1 = 1.2
        self.b = 0.75
        self.doc_freqs = Counter()
        self.total_docs = 1
        self.avgdl = 1.0

        full_path = os.path.join(BASE_DIR, filepath)
        if os.path.exists(full_path):
            try:
                with open(full_path, "r") as f:
                    data = json.load(f)
                    self.k1 = data.get("k1", 1.2)
                    self.b = data.get("b", 0.75)
                    self.total_docs = data.get("total_docs", 1)
                    self.avgdl = data.get("avgdl", 1.0)
                    self.doc_freqs = Counter(data.get("doc_freqs", {}))
            except Exception as e:
                print(f"⚠️ Warning: Failed to load {filepath}: {e}")

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

    @staticmethod
    def _token_to_id(token: str) -> int:
        return int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)

    def encode(self, text: str) -> dict:
        tokens = self.tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        counts = Counter(tokens)
        doc_len = len(tokens)
        indices, values = [], []

        for token, tf in counts.items():
            df = self.doc_freqs.get(token, 1)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            weight = idf * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))))
            if weight > 0:
                indices.append(self._token_to_id(token))
                values.append(round(float(weight), 4))

        return {"indices": indices, "values": values}

bm25 = PureBM25Encoder()

# ==============================================================================
# 2. ENVIRONMENT & SAFE CONFIGURATION
# ==============================================================================
SENTRY_DSN = os.environ.get("SENTRY_DSN")
if SENTRY_DSN and SENTRY_DSN.startswith("http"):
    try:
        sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=1.0)
    except Exception as e:
        print(f"⚠️ Sentry init skipped: {e}")

UPSTASH_REDIS_REST_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "bns-legal-index")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "bns-auto-ip-jwt-secret-key-2026-secure-32bytes")
JWT_ISSUER_URL = os.environ.get("JWT_ISSUER_URL", "")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "")

GROQ_KEYS = [os.environ.get(f"GROQ_API_KEY_{i}") for i in range(1, 6)]
GROQ_KEYS = [k for k in GROQ_KEYS if k]

# ==============================================================================
# 3. FAIL-SAFE UPSTASH REDIS CLIENT
# ==============================================================================
class UpstashRedisREST:
    def __init__(self, url: str, token: str):
        self.url = url.rstrip('/') if url else ""
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.is_placeholder = "your-database.upstash.io" in self.url or not self.url

    def _exec(self, command: List[str]):
        if self.is_placeholder:
            return None
        try:
            resp = requests.post(self.url, headers=self.headers, json=command, timeout=2)
            return resp.json().get("result") if resp.status_code == 200 else None
        except Exception:
            return None

    def get(self, key: str) -> Optional[str]:
        return self._exec(["GET", key])

    def setex(self, key: str, seconds: int, value: str):
        return self._exec(["SET", key, value, "EX", str(seconds)])

    def incr(self, key: str) -> int:
        res = self._exec(["INCR", key])
        return int(res) if res is not None else 1

    def expire(self, key: str, seconds: int):
        return self._exec(["EXPIRE", key, str(seconds)])

    def sadd(self, key: str, member: str):
        return self._exec(["SADD", key, member])

    def scard(self, key: str) -> int:
        res = self._exec(["SCARD", key])
        return int(res) if res is not None else 0

    def sismember(self, key: str, member: str) -> bool:
        res = self._exec(["SISMEMBER", key, member])
        return bool(res) if res is not None else False

    def keys(self, pattern: str) -> List[str]:
        res = self._exec(["KEYS", pattern])
        return res if isinstance(res, list) else []

redis = UpstashRedisREST(UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN)

# ==============================================================================
# 4. SMART KEY MANAGER & JWT AUTHENTICATION
# ==============================================================================
class SmartKeyManager:
    def __init__(self, keys: List[str]):
        self.keys = keys

    def get_client(self) -> tuple[Groq, str]:
        if not self.keys:
            # Fallback mock or check for standard GROQ_API_KEY
            fallback_key = os.environ.get("GROQ_API_KEY")
            if fallback_key:
                return Groq(api_key=fallback_key), fallback_key
            raise HTTPException(status_code=500, detail="No Groq API keys configured in environment variables.")
        
        for _ in range(len(self.keys)):
            idx = redis.incr("bns:key_idx") % len(self.keys)
            key = self.keys[idx]
            khash = hashlib.md5(key.encode()).hexdigest()[:8]
            if not redis.get(f"bns:cooldown:{khash}"):
                return Groq(api_key=key), key
        return Groq(api_key=self.keys[0]), self.keys[0]

    def cooldown(self, key: str, ttl: int = 60):
        khash = hashlib.md5(key.encode()).hexdigest()[:8]
        redis.setex(f"bns:cooldown:{khash}", ttl, "1")

key_mgr = SmartKeyManager(GROQ_KEYS)
security = HTTPBearer(auto_error=False)

def generate_ip_jwt(ip_address: str) -> str:
    anonymized_ip_id = f"ip_{hashlib.sha256(ip_address.encode()).hexdigest()[:16]}"
    payload = {
        "sub": anonymized_ip_id,
        "client_ip": ip_address,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=1)
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")

def verify_jwt_token(
    req: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    client_ip = req.headers.get("x-forwarded-for", req.client.host if req.client else "127.0.0.1").split(",")[0].strip()

    if not credentials:
        return f"ip_{hashlib.sha256(client_ip.encode()).hexdigest()[:16]}"

    token = credentials.credentials
    try:
        use_jwks = JWT_ISSUER_URL and "your-auth-domain" not in JWT_ISSUER_URL
        if use_jwks:
            jwks_client = jwt.PyJWKClient(f"{JWT_ISSUER_URL.rstrip('/')}/.well-known/jwks.json")
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=JWT_AUDIENCE,
                issuer=JWT_ISSUER_URL
            )
        else:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"], options={"verify_signature": False})
            
        return payload.get("sub", f"ip_{hashlib.sha256(client_ip.encode()).hexdigest()[:16]}")
    except Exception:
        return f"ip_{hashlib.sha256(client_ip.encode()).hexdigest()[:16]}"

def enforce_limits(user_id: str) -> str:
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        users_key = f"bns:users:{today}"
        limit_key = f"bns:limit:{today}:{user_id}"

        active_users = redis.scard(users_key) or 0
        is_member = redis.sismember(users_key, user_id)

        if not is_member and active_users >= 500:
            raise HTTPException(status_code=429, detail="Daily system threshold reached (500 active users/day).")

        raw_usage = redis.get(limit_key)
        current_usage = int(raw_usage) if raw_usage is not None else 0
        if current_usage >= 20:
            raise HTTPException(status_code=429, detail="Daily quota reached (20 queries/day max per user).")

        redis.sadd(users_key, user_id)
        redis.expire(users_key, 86400)
        new_count = redis.incr(limit_key) or (current_usage + 1)
        redis.expire(limit_key, 86400)

        return f"{max(0, 20 - new_count)}/20 queries remaining today"
    except HTTPException:
        raise
    except Exception as e:
        print(f"⚠️ Redis Limit Enforcement Warning: {e}")
        return "20/20 queries remaining today"

# ==============================================================================
# 5. DENSE EMBEDDING RETRIEVAL
# ==============================================================================
def get_remote_dense_embedding(text: str) -> List[float]:
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
    
    try:
        resp = requests.post(url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and isinstance(data[0], list):
                return data[0]
            elif isinstance(data, list) and isinstance(data[0], (float, int)):
                return data
    except Exception as e:
        print(f"⚠️ Hugging Face API error: {e}. Using deterministic fallback.")

    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    vector = np.random.normal(0, 1, 384)
    vector /= np.linalg.norm(vector)
    return vector.tolist()

# ==============================================================================
# 6. SEMANTIC CACHING HELPER
# ==============================================================================
def check_semantic_cache(query_vector: List[float], similarity_threshold: float = 0.96):
    try:
        cached_keys = redis.keys("bns:semcache:*")
        if not cached_keys:
            return None, 0.0

        q_vec = np.array(query_vector)

        for key in cached_keys:
            raw_val = redis.get(key)
            if not raw_val:
                continue
            try:
                cached_item = json.loads(raw_val)
                c_vec = np.array(cached_item["vector"])
                
                sim = np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec))
                if sim >= similarity_threshold:
                    return cached_item["response"], float(sim)
            except Exception:
                continue
    except Exception as e:
        print(f"⚠️ Semantic cache error: {e}")
        
    return None, 0.0

# ==============================================================================
# 7. FASTAPI APPLICATION ENGINE & STATIC FRONTEND SERVING
# ==============================================================================
app = FastAPI(title="Enterprise BNS Legal AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PUBLIC_DIR = os.path.join(BASE_DIR, "public")

if os.path.exists(PUBLIC_DIR):
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")

class QueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What is the penalty for extortion under BNS?"})

@app.get("/")
def serve_frontend():
    index_path = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "Backend active. index.html not found in public/"}

@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon():
    favicon_path = os.path.join(PUBLIC_DIR, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return FileResponse(os.path.join(PUBLIC_DIR, "index.html"))

@app.get("/api/auth/token")
def get_auto_ip_token(req: Request):
    client_ip = req.headers.get("x-forwarded-for", req.client.host if req.client else "127.0.0.1").split(",")[0].strip()
    token = generate_ip_jwt(client_ip)
    return {"token": token, "client_ip": client_ip}

@app.post("/api/query/stream")
def handle_query_stream(
    payload: QueryRequest,
    req: Request,
    user_id: str = Depends(verify_jwt_token)
):
    user_query = payload.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        dense_vector = get_remote_dense_embedding(user_query)

        # 1. Check Semantic Cache
        cached_answer, sim_score = check_semantic_cache(dense_vector)
        if cached_answer:
            def cached_stream():
                yield f"data: {json.dumps({'text': cached_answer, 'status': f'🎯 Semantic Cache Hit ({sim_score:.2f} sim)'})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(cached_stream(), media_type="text/event-stream")

        # 2. Check Limits
        quota_status = enforce_limits(user_id)

        # 3. Safe Pinecone Search
        matches = []
        pc = None
        if PINECONE_API_KEY:
            try:
                pc = Pinecone(api_key=PINECONE_API_KEY)
                vector_index = pc.Index(PINECONE_INDEX_NAME)
                sparse_vector = bm25.encode(user_query)

                search_results = vector_index.query(
                    vector=dense_vector,
                    sparse_vector=sparse_vector,
                    top_k=15,
                    include_metadata=True
                )
                if isinstance(search_results, dict):
                    matches = search_results.get("matches", [])
                else:
                    matches = getattr(search_results, "matches", [])
            except Exception as pc_err:
                print(f"⚠️ Pinecone search failed: {pc_err}")

        candidate_docs = []
        for m in matches:
            meta = getattr(m, "metadata", None) if not isinstance(m, dict) else m.get("metadata")
            if meta and isinstance(meta, dict) and "text" in meta:
                doc_id = getattr(m, "id", None) if not isinstance(m, dict) else m.get("id")
                candidate_docs.append({"id": doc_id, "text": meta["text"]})

        contexts = []
        if candidate_docs and pc:
            try:
                rerank_resp = pc.inference.rerank(
                    model="bge-reranker-v2-m3",
                    query=user_query,
                    documents=candidate_docs,
                    top_n=3,
                    return_documents=True
                )
                contexts = [item.document["text"] for item in rerank_resp.data]
            except Exception as rerank_err:
                print(f"⚠️ Reranker failed: {rerank_err}. Using vector ranking.")
                contexts = [doc["text"] for doc in candidate_docs[:3]]
        elif candidate_docs:
            contexts = [doc["text"] for doc in candidate_docs[:3]]

        context_str = "\n\n".join([f"Source {i+1}:\n{c}" for i, c in enumerate(contexts)]) if contexts else "No context retrieved."

    except HTTPException:
        raise
    except Exception as pipeline_err:
        print("\n" + "="*60)
        print("❌ ROUTE ERROR:")
        traceback.print_exc()
        print("="*60 + "\n")
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(pipeline_err)}")

    # 4. SSE Stream Engine
    def sse_generator():
        try:
            sys_prompt = """You are an expert legal AI engine for Bharatiya Nyaya Sanhita (BNS), 2023.

Format your response using these exact section headers:
**Statutory Provision:**
[Section number under BNS/IPC]

**Overview & Scope:**
[Summary of legal applicability]

**Prescribed Penalty / Consequence:**
[Detailed statutory penalties, imprisonment terms, or fines]"""

            full_response_text = ""
            client, active_key = key_mgr.get_client()

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nQUERY: {user_query}"}
                ],
                temperature=0.1,
                max_tokens=400,
                stream=True
            )

            yield f"data: {json.dumps({'meta': True, 'status': f'🟢 Streaming | {quota_status}'})}\n\n"

            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_response_text += content
                    yield f"data: {json.dumps({'chunk': content})}\n\n"

            if full_response_text:
                cache_payload = {
                    "query": user_query,
                    "vector": dense_vector,
                    "response": full_response_text
                }
                q_hash = hashlib.md5(user_query.lower().encode()).hexdigest()[:12]
                redis.setex(f"bns:semcache:{q_hash}", 86400, json.dumps(cache_payload))

            yield "data: [DONE]\n\n"

        except RateLimitError:
            key_mgr.cooldown(active_key)
            yield f"data: {json.dumps({'error': 'Groq rate limit hit. Rotating API keys...'})}\n\n"
        except Exception as e:
            print("\n" + "="*60)
            print("❌ STREAM GENERATOR EXCEPTION:")
            traceback.print_exc()
            print("="*60 + "\n")
            yield f"data: {json.dumps({'error': f'Streaming failure: {str(e)}'})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
