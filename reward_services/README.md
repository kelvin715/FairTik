# TikTok Reward Engine API Guide

This directory contains a budget-preserving creator reward allocation engine and a FastAPI-based RESTful API for DPO-based reward calculation, storage, querying, and deletion. Rewards are stored in SQLite (`creator_rewards.db`).

---

## Quick Start

1. **Install dependencies**

```bash
pip install -r ../requirements.txt
```

2. **Start the API service**

```bash
python reward_services.py
# Or use the provided script
bash reward_service_setup.sh
```

3. **Access API documentation**

Open [http://localhost:8001/docs](http://localhost:8001/docs) in your browser for interactive docs.

---

## API Endpoints

### 1. Calculate Actual Rewards

**POST /api/calculate-actual-rewards**

- **Request Example:**
  ```json
  {
    "dpo_outputs": [
      {
        "index": 0,
        "creator_id": "creatorA",
        "stream_id": "stream001",
        "recommended_percentage": 0.0,
        "reasoning": "High quality content",
        "confidence_score": 0.9,
        "model_version": "DPO-v1.0",
        "coins": 500
      },
      {
        "index": 1,
        "creator_id": "creatorB",
        "stream_id": "stream002",
        "recommended_percentage": 70.0,
        "reasoning": "Strong engagement",
        "confidence_score": 0.8,
        "model_version": "DPO-v1.0",
        "coins": 2000
      }
    ]
  }
  ```
- **Reward Calculation:**  
  - If `recommended_percentage == 0.0`, reward is `coins * 0.2` (20% of coins).
  - Otherwise, reward is `coins * (recommended_percentage / 100.0)`.
- **Response:** List of reward results for each creator, including timestamp.

### 2. Store Reward Results

**POST /api/store-reward-result**

- **Request:** List of `RewardResult` objects (output from previous endpoint).
- **Function:** Stores results in SQLite. Duplicate (creator_id, stream_id) pairs are skipped.

### 3. Get Single Reward Result

**GET /api/get-reward-result/{creator_id}/{stream_id}**

- **Function:** Retrieves the reward result for a specific creator and stream.

### 4. Get Reward History

**GET /api/reward-history/{creator_id}**

- **Function:** Returns all reward results for a creator across all streams.

### 5. Delete Reward Result

**DELETE /api/delete-reward-result/{creator_id}/{stream_id}**

- **Function:** Deletes the reward result for a specific creator and stream.

---

## Features

- **20% Reward for Zero Recommendation:** Creators with `recommended_percentage = 0.0` receive 20% of their coins as reward.
- **Reward Calculation:** Each creator's reward is calculated as above.
- **Duplicate Prevention:** Each (creator_id, stream_id) pair is stored only once.
- **Timestamp:** Each reward result includes a UTC timestamp.
- **Reward History:** Query all rewards for a creator.
- **Delete Support:** Remove specific reward results via API.
- **SQLite Persistence:** Results are stored in `creator_rewards.db` for durability.
- **Interactive API Docs:** Visit `/docs` for live API testing.

---

## Example Usage

```python
import requests

BASE_URL = "http://localhost:8001"

# Calculate rewards
payload = {
    "dpo_outputs": [
        {
            "index": 0,
            "creator_id": "creatorA",
            "stream_id": "stream001",
            "recommended_percentage": 0.0,
            "reasoning": "High quality content",
            "confidence_score": 0.9,
            "model_version": "DPO-v1.0",
            "coins": 500
        },
        {
            "index": 1,
            "creator_id": "creatorB",
            "stream_id": "stream002",
            "recommended_percentage": 70.0,
            "reasoning": "Strong engagement",
            "confidence_score": 0.8,
            "model_version": "DPO-v1.0",
            "coins": 2000
        }
    ]
}
results = requests.post(f"{BASE_URL}/api/calculate-actual-rewards", json=payload).json()

# Store rewards
requests.post(f"{BASE_URL}/api/store-reward-result", json=results)

# Get single reward
r = requests.get(f"{BASE_URL}/api/get-reward-result/creatorA/stream001").json()

# Get reward history
history = requests.get(f"{BASE_URL}/api/reward-history/creatorA").json()

# Delete a reward result
del_resp = requests.delete(f"{BASE_URL}/api/delete-reward-result/creatorA/stream001").json()
```

---

## Database File

- File: `creator_rewards.db`
- Table: `rewards`
- Fields: `creator_id, stream_id, idx, recommended_percentage, reasoning, confidence_score, model_version, coins, actual_reward, timestamp`

---
