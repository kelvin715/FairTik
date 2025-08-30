from typing import List, Dict, Optional
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from datetime import datetime
import sqlite3
import json

app = FastAPI(
    title="TikTok Reward Utilities Service",
    description="Tiktok Reward Utilities Service",
    version="1.0.0"
)

DB_FILENAME = "creator_rewards.db"

class DPOOutput(BaseModel):
    index: int
    recommended_percentage: float
    reasoning: str
    confidence_score: float
    model_version: str
    creator_id: Optional[str] = None
    stream_id: str
    coins: float

class RewardRequest(BaseModel):
    dpo_outputs: List[DPOOutput]

class RewardResult(BaseModel):
    index: int
    stream_id: str
    creator_id: Optional[str]
    recommended_percentage: float
    reasoning: str
    confidence_score: float
    model_version: str
    coins: float
    actual_reward: float
    timestamp: Optional[str] = None

class LiveHistoryRequest(BaseModel):
    creator_id: str
    stream_id: str
    content: dict  # The test_data dict above

class LiveHistoryResult(BaseModel):
    creator_id: str
    stream_id: str
    content: dict
    timestamp: str

def init_db():
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    # Rewards table
    c.execute("""
        CREATE TABLE IF NOT EXISTS rewards (
            creator_id TEXT,
            stream_id TEXT,
            idx INTEGER,
            recommended_percentage REAL,
            reasoning TEXT,
            confidence_score REAL,
            model_version TEXT,
            coins REAL,
            actual_reward REAL,
            timestamp TEXT,
            PRIMARY KEY (creator_id, stream_id)
        )
    """)
    # Live history table
    c.execute("""
        CREATE TABLE IF NOT EXISTS live_history (
            creator_id TEXT,
            stream_id TEXT,
            content_json TEXT,
            timestamp TEXT,
            PRIMARY KEY (creator_id, stream_id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

def write_rewards_to_db(results: List[RewardResult]):
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    for r in results:
        try:
            c.execute("""
                INSERT INTO rewards (
                    creator_id, stream_id, idx, recommended_percentage, reasoning,
                    confidence_score, model_version, coins, actual_reward, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r.creator_id, r.stream_id, r.index, r.recommended_percentage, r.reasoning,
                r.confidence_score, r.model_version, r.coins, r.actual_reward, r.timestamp
            ))
        except sqlite3.IntegrityError:
            # Duplicate, skip
            continue
    conn.commit()
    conn.close()

def read_reward_from_db(creator_id: str, stream_id: str) -> Optional[RewardResult]:
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    c.execute("""
        SELECT idx, stream_id, creator_id, recommended_percentage, reasoning,
               confidence_score, model_version, coins, actual_reward, timestamp
        FROM rewards WHERE creator_id=? AND stream_id=?
    """, (creator_id, stream_id))
    row = c.fetchone()
    conn.close()
    if row:
        return RewardResult(
            index=row[0], stream_id=row[1], creator_id=row[2],
            recommended_percentage=row[3], reasoning=row[4],
            confidence_score=row[5], model_version=row[6],
            coins=row[7], actual_reward=row[8], timestamp=row[9]
        )
    return None

def read_reward_history_from_db(creator_id: str) -> List[RewardResult]:
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    c.execute("""
        SELECT idx, stream_id, creator_id, recommended_percentage, reasoning,
               confidence_score, model_version, coins, actual_reward, timestamp
        FROM rewards WHERE creator_id=?
    """, (creator_id,))
    rows = c.fetchall()
    conn.close()
    return [
        RewardResult(
            index=row[0], stream_id=row[1], creator_id=row[2],
            recommended_percentage=row[3], reasoning=row[4],
            confidence_score=row[5], model_version=row[6],
            coins=row[7], actual_reward=row[8], timestamp=row[9]
        )
        for row in rows
    ]

@app.post("/api/calculate-actual-rewards", response_model=List[RewardResult])
async def calculate_actual_rewards_api(request: RewardRequest):
    dpo_outputs = [o.dict() for o in request.dpo_outputs]
    results = []
    now = datetime.utcnow().isoformat()
    for c in dpo_outputs:
        pct = max(0.0, min(100.0, c.get("recommended_percentage", 0.0)))
        coins = float(c.get("coins", 0.0))
        if pct == 0.0:
            actual_reward = coins * 0.2
        else:
            actual_reward = coins * (pct / 100.0)
        result_dict = {**c, "actual_reward": actual_reward, "timestamp": now}
        results.append(RewardResult(**result_dict))
    return results

@app.post("/api/store-reward-result")
async def store_reward_result(results: List[RewardResult]):
    """Store reward results for each creator_id and stream_id (in SQLite, no duplicates)."""
    write_rewards_to_db(results)
    return {
        "status": "success",
        "stored_count": len(results)
    }

@app.get("/api/get-reward-result/{creator_id}/{stream_id}", response_model=RewardResult)
async def get_reward_result(creator_id: str, stream_id: str):
    result = read_reward_from_db(creator_id, stream_id)
    if not result:
        raise HTTPException(status_code=404, detail="Reward result not found for this creator_id and stream_id")
    return result

@app.get("/api/reward-history/{creator_id}", response_model=List[RewardResult])
async def get_reward_history(creator_id: str):
    return read_reward_history_from_db(creator_id)

@app.delete("/api/delete-reward-result/{creator_id}/{stream_id}")
async def delete_reward_result(creator_id: str, stream_id: str):
    """Delete a reward result for a specific creator_id and stream_id."""
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    c.execute("""
        DELETE FROM rewards WHERE creator_id=? AND stream_id=?
    """, (creator_id, stream_id))
    conn.commit()
    deleted = c.rowcount
    conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Reward result not found for this creator_id and stream_id")
    return {"status": "success", "deleted_count": deleted}

@app.post("/api/store-live-history", response_model=LiveHistoryResult)
async def store_live_history(request: LiveHistoryRequest):
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO live_history (creator_id, stream_id, content_json, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            request.creator_id,
            request.stream_id,
            json.dumps(request.content),
            now
        ))
    except sqlite3.IntegrityError:
        # If already exists, update
        c.execute("""
            UPDATE live_history SET content_json=?, timestamp=?
            WHERE creator_id=? AND stream_id=?
        """, (
            json.dumps(request.content),
            now,
            request.creator_id,
            request.stream_id
        ))
    conn.commit()
    conn.close()
    return LiveHistoryResult(
        creator_id=request.creator_id,
        stream_id=request.stream_id,
        content=request.content,
        timestamp=now
    )

@app.get("/api/get-live-history/{creator_id}/{stream_id}", response_model=LiveHistoryResult)
async def get_live_history(creator_id: str, stream_id: str):
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    c.execute("""
        SELECT content_json, timestamp FROM live_history
        WHERE creator_id=? AND stream_id=?
    """, (creator_id, stream_id))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Live history not found for this creator_id and stream_id")
    return LiveHistoryResult(
        creator_id=creator_id,
        stream_id=stream_id,
        content=json.loads(row[0]),
        timestamp=row[1]
    )

@app.get("/api/live-history-list/{creator_id}", response_model=List[LiveHistoryResult])
async def get_live_history_list(creator_id: str):
    """Get all live history records for a creator (across all streams)."""
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    c.execute("""
        SELECT stream_id, content_json, timestamp FROM live_history
        WHERE creator_id=?
    """, (creator_id,))
    rows = c.fetchall()
    conn.close()
    return [
        LiveHistoryResult(
            creator_id=creator_id,
            stream_id=row[0],
            content=json.loads(row[1]),
            timestamp=row[2]
        )
        for row in rows
    ]

@app.delete("/api/delete-live-history/{creator_id}")
async def delete_live_history(creator_id: str):
    """Delete all live history records for a creator."""
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    c.execute("""
        DELETE FROM live_history WHERE creator_id=?
    """, (creator_id,))
    conn.commit()
    deleted = c.rowcount
    conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="No live history found for this creator_id")
    return {"status": "success", "deleted_count": deleted}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "reward_services:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
