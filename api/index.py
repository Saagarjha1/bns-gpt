import sys
import json
import time
import random
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import jwt

\# ==========================================
\# PATH CONFIGURATION (Fix for Render/Linux imports)
\# ==========================================
ROOT\_DIR = Path(\_\_file\_\_).resolve().parent.parent
if str(ROOT\_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT\_DIR))

\# Optional integrations with fallback
try:
    from pinecone import Pinecone
    PINECONE\_AVAILABLE = True
except Exception:
    PINECONE\_AVAILABLE = False

try:
    import sentry\_sdk
    SENTRY\_AVAILABLE = True
except Exception:
    SENTRY\_AVAILABLE = False


\# ==========================================
\# ENVIRONMENT & CONFIGURATION
\# ==========================================

GROQ\_KEYS = [
    os.getenv("GROQ\_API\_KEY"),
    os.getenv("GROQ\_API\_KEY\_1"),
    os.getenv("GROQ\_API\_KEY\_2"),
    os.getenv("GROQ\_API\_KEY\_3"),
    os.getenv("GROQ\_API\_KEY\_4"),
    os.getenv("GROQ\_API\_KEY\_5"),
]
VALID\_GROQ\_KEYS = [k for k in GROQ\_KEYS if k and k.strip()]

PINECONE\_API\_KEY = os.getenv("PINECONE\_API\_KEY")
PINECONE\_INDEX\_NAME = os.getenv("PINECONE\_INDEX\_NAME", "bns-legal")
JWT\_SECRET = os.getenv("JWT\_SECRET") or os.getenv("JWT\_SECRET\_KEY") or "super-secret-bns-key"
UPSTASH\_REDIS\_REST\_URL = os.getenv("UPSTASH\_REDIS\_REST\_URL")
UPSTASH\_REDIS\_REST\_TOKEN = os.getenv("UPSTASH\_REDIS\_REST\_TOKEN")
SENTRY\_DSN = os.getenv("SENTRY\_DSN")

if SENTRY\_AVAILABLE and SENTRY\_DSN:
    try:
        sentry\_sdk.init(dsn=SENTRY\_DSN, traces\_sample\_rate=1.0)
    except Exception:
        pass

app = FastAPI(title="Enterprise BNS Legal AI API", version="1.0.0")

app.add\_middleware(
    CORSMiddleware,
    allow\_origins=["\*"],
    allow\_credentials=True,
    allow\_methods=["\*"],
    allow\_headers=["\*"],
)


\# ==========================================
\# CLIENT INITIALIZATIONS
\# ==========================================
pc = None
index = None

if PINECONE\_AVAILABLE and PINECONE\_API\_KEY:
    try:
        pc = Pinecone(api\_key=PINECONE\_API\_KEY)
        index = pc.Index(PINECONE\_INDEX\_NAME)
    except Exception as e:
        print(f"⚠️ Pinecone initialization error: {e}")


\# ==========================================
\# SCHEMAS & HELPERS
\# ==========================================
class QueryRequest(BaseModel):
    query: str


def get\_client\_ip(request: Request) -> str:
    x\_forwarded\_for = request.headers.get("x-forwarded-for")
    if x\_forwarded\_for:
        return x\_forwarded\_for.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def create\_jwt\_token(ip: str) -> str:
    payload = {
        "ip": ip,
        "iat": int(time.time()),
        "exp": int(time.time()) + (24 \* 3600)
    }
    return jwt.encode(payload, JWT\_SECRET, algorithm="HS256")


def verify\_jwt\_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(token, JWT\_SECRET, algorithms=["HS256"])
    except Exception:
        return None


\# ==========================================
\# SAFE RATE LIMITING LOGIC
\# ==========================================
MAX\_QUERIES\_PER\_DAY = 20

async def check\_rate\_limit\_async(client\_ip: str) -> tuple[bool, int]:
    if not UPSTASH\_REDIS\_REST\_URL or not UPSTASH\_REDIS\_REST\_TOKEN:
        return True, MAX\_QUERIES\_PER\_DAY

    key = f"rate\_limit:{client\_ip}:{time.strftime('%Y-%m-%d')}"
    headers = {"Authorization": f"Bearer {UPSTASH\_REDIS\_REST\_TOKEN}"}
   &#x20;
    try:
        async with httpx.AsyncClient(timeout=2.5) as http\_client:
            inc\_res = await http\_client.post(f"{UPSTASH\_REDIS\_REST\_URL}/incr/{key}", headers=headers)
            res\_data = inc\_res.json()
           &#x20;
            raw\_result = res\_data.get("result") if isinstance(res\_data, dict) else None
            current\_count = int(raw\_result) if raw\_result is not None and str(raw\_result).isdigit() else 1
           &#x20;
            if current\_count == 1:
                await http\_client.post(f"{UPSTASH\_REDIS\_REST\_URL}/expire/{key}/86400", headers=headers)
               &#x20;
            remaining = max(0, MAX\_QUERIES\_PER\_DAY - current\_count)
            if current\_count > MAX\_QUERIES\_PER\_DAY:
                return False, 0
            return True, remaining
    except Exception as err:
        print(f"⚠️ Rate limit warning: {err}")
        return True, MAX\_QUERIES\_PER\_DAY


\# ==========================================
\# API ENDPOINTS
\# ==========================================

@app.get("/")
@app.get("/health")
@app.get("/api/health")
def health\_check():
    return {
        "status": "healthy",
        "pinecone\_connected": index is not None,
        "valid\_groq\_keys\_count": len(VALID\_GROQ\_KEYS)
    }


@app.get("/auth/token")
@app.get("/api/auth/token")
def get\_auth\_token(request: Request):
    client\_ip = get\_client\_ip(request)
    token = create\_jwt\_token(client\_ip)
    return {"token": token, "ip": client\_ip}


@app.post("/query/stream")
@app.post("/api/query/stream")
async def query\_stream(
    req: QueryRequest,
    request: Request,
    authorization: Optional[str] = Header(None)
):
    try:
        user\_query = req.query.strip() if req and req.query else ""
        if not user\_query:
            raise HTTPException(status\_code=400, detail="Query string cannot be empty.")

        client\_ip = get\_client\_ip(request)
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            decoded = verify\_jwt\_token(token)
            if decoded and "ip" in decoded:
                client\_ip = decoded["ip"]

        allowed, remaining = await check\_rate\_limit\_async(client\_ip)
        if not allowed:
            raise HTTPException(
                status\_code=429,
                detail=f"Rate limit exceeded. Maximum {MAX\_QUERIES\_PER\_DAY} queries allowed per day."
            )

        async def event\_generator() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'status': f'Searching database... | {remaining}/{MAX\_QUERIES\_PER\_DAY} queries remaining'})}\n\n"
            await asyncio.sleep(0.01)

            candidate\_docs = []

            \# Safe Pinecone Query
            if index and PINECONE\_API\_KEY:
                try:
                    loop = asyncio.get\_event\_loop()
                    query\_res = await loop.run\_in\_executor(
                        None,&#x20;
                        lambda: index.query(vector=[0.0] \* 1536, top\_k=3, include\_metadata=True)
                    )
                    for match in query\_res.get("matches", []):
                        if "metadata" in match and "text" in match["metadata"]:
                            candidate\_docs.append({"text": match["metadata"]["text"]})
                except Exception as vector\_err:
                    print(f"⚠️ Vector search warning: {vector\_err}")

            contexts = [doc["text"] for doc in candidate\_docs[:3]] if candidate\_docs else []

            yield f"data: {json.dumps({'status': 'Generating legal analysis...'})}\n\n"

            selected\_groq\_key = random.choice(VALID\_GROQ\_KEYS) if VALID\_GROQ\_KEYS else None

            if not selected\_groq\_key:
                fallback\_text = (
                    f"### Query: {user\_query}\n\n"
                    "\*\*Configuration Required:\*\* \`GROQ\_API\_KEY\` is not set in Environment Variables.\n\n"
                    "Please add \`GROQ\_API\_KEY\` in your Render dashboard environment settings."
                )
                yield f"data: {json.dumps({'chunk': fallback\_text})}\n\n"
                yield "data: [DONE]\n\n"
                return

            system\_prompt = (
                "You are an expert legal assistant specializing in the Bharatiya Nyaya Sanhita (BNS), 2023. "
                "Provide precise, structured, and legally accurate answers based on statutory provisions. "
                "Cite relevant BNS section numbers where applicable."
            )
            context\_str = "\n\n".join(contexts) if contexts else "No relevant statutory provisions retrieved from vector index."
            full\_user\_prompt = f"Context:\n{context\_str}\n\nUser Legal Query: {user\_query}"

            try:
                headers = {
                    "Authorization": f"Bearer {selected\_groq\_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": system\_prompt},
                        {"role": "user", "content": full\_user\_prompt}
                    ],
                    "stream": True,
                    "temperature": 0.2
                }

                async with httpx.AsyncClient(timeout=30.0) as client:
                    async with client.stream(
                        "POST",
                        "[https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)",
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status\_code != 200:
                            err\_body = await response.aread()
                            yield f"data: {json.dumps({'chunk': f'Groq API Error ({response.status\_code}): {err\_body.decode()}'})}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        async for line in response.aiter\_lines():
                            if line.startswith("data: "):
                                data\_str = line[6:].strip()
                                if data\_str == "[DONE]":
                                    break
                                try:
                                    data\_json = json.loads(data\_str)
                                    delta = data\_json["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        yield f"data: {json.dumps({'chunk': delta})}\n\n"
                                except Exception:
                                    continue

                yield f"data: {json.dumps({'status': f'Completed | {remaining}/{MAX\_QUERIES\_PER\_DAY} queries remaining'})}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as groq\_err:
                yield f"data: {json.dumps({'chunk': f'Streaming exception: {str(groq\_err)}'})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event\_generator(),
            media\_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except HTTPException as http\_ex:
        raise http\_ex
    except Exception as top\_ex:
        raise HTTPException(status\_code=500, detail=f"Internal Server Error: {str(top\_ex)}")


@app.api\_route("/{path\_name\:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch\_all(request: Request, path\_name: str):
    return JSONResponse(
        status\_code=404,
        content={"error": "Route not found", "requested\_path": path\_name}
    )
