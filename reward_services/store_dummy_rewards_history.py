import sqlite3
import random
from datetime import datetime

DB_FILENAME = "creator_rewards.db"

def generate_dummy_reward(live_row):
    creator_id = live_row[0]
    stream_id = live_row[1]
    idx = random.randint(0, 100)
    recommended_percentage = random.choice([0.0, 20.0, 50.0, 70.0, 100.0])
    reasoning = random.choice([
        "High quality content", "Strong engagement", "Excellent performance", "Average engagement", "New creator boost"
    ])
    confidence_score = round(random.uniform(0.6, 1.0), 2)
    model_version = "DPO-v1.0"
    # Simulate coins based on stream length or random
    coins = random.randint(100, 5000)
    # Reward logic: 20% of coins if recommended_percentage == 0, else coins * percentage
    if recommended_percentage == 0.0:
        actual_reward = coins * 0.2
    else:
        actual_reward = coins * (recommended_percentage / 100.0)
    timestamp = datetime.utcnow().isoformat()
    return (
        creator_id, stream_id, idx, recommended_percentage, reasoning,
        confidence_score, model_version, coins, actual_reward, timestamp
    )

def store_dummy_rewards():
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    # Get all live history records
    c.execute("SELECT creator_id, stream_id FROM live_history")
    live_rows = c.fetchall()
    count = 0
    for live_row in live_rows:
        try:
            dummy_reward = generate_dummy_reward(live_row)
            c.execute("""
                INSERT INTO rewards (
                    creator_id, stream_id, idx, recommended_percentage, reasoning,
                    confidence_score, model_version, coins, actual_reward, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, dummy_reward)
            count += 1
        except sqlite3.IntegrityError:
            # Skip if already exists
            continue
    conn.commit()
    conn.close()
    print(f"Inserted {count} dummy rewards based on live history.")

if __name__ == "__main__":
    store_dummy_rewards()