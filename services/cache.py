import os
import time
import httpx

UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

async def get_cache_async(key: str) -> str | None:
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            res = await client.get(f"{UPSTASH_REDIS_REST_URL}/get/{key}", headers=headers)
            if res.status_code == 200:
                data = res.json()
                return data.get("result")
    except Exception as err:
        print(f"⚠️ Cache get error: {err}")
    return None

async def set_cache_async(key: str, value: str, ttl_seconds: int = 86400) -> bool:
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        return False
    headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            res = await client.post(
                f"{UPSTASH_REDIS_REST_URL}/set/{key}/{value}?EX={ttl_seconds}",
                headers=headers
            )
            return res.status_code == 200
    except Exception as err:
        print(f"⚠️ Cache set error: {err}")
    return False
