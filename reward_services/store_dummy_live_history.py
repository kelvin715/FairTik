import sqlite3
import json
import random
from uuid import uuid4
from datetime import datetime
import base64

# ----------------------------
# Config
# ----------------------------
DB_FILENAME = "creator_rewards.db"
SEED = 42            # Set to None for non-reproducible randomness
STREAMS_PER_CREATOR = 5
CREATORS = ["creatorA", "creatorB", "creatorC", "creatorD", "creatorE"]

if SEED is not None:
    random.seed(SEED)

def _generate_keyframes_with_images(duration_minutes):
    duration_sec = duration_minutes * 60
    num_keyframes = 3

    # Generate random timestamps
    timestamps = sorted(random.sample(range(30, duration_sec - 30), num_keyframes))

    # Fake image data encoded in Base64
    example_image_b64 = base64.b64encode(b"fake_image_data").decode("utf-8")

    keyframes = [{"timestamp": ts, "image_data": example_image_b64} for ts in timestamps]
    return keyframes

# ----------------------------
# Helper: Generate a random stream content payload
# ----------------------------
def _random_stream_payload(creator_id, base_profile, stream_num):
    # Use fixed stream_id format
    stream_id = f"stream{stream_num:03d}"

    # Duration
    duration_minutes = random.randint(35, 180)                 # 35 min to 3 hours
    duration_hours = round(duration_minutes / 60.0, 2)

    # Generate keyframes with images
    keyframes = _generate_keyframes_with_images(duration_minutes)

    # Viewership, scaled to creator's followers
    follower_count = base_profile["follower_count"]
    peak_viewers = max(50, int(follower_count * random.uniform(0.01, 0.12)))
    average_viewers = max(20, int(peak_viewers * random.uniform(0.6, 0.9)))

    # Interactions proportional to audience and duration
    comments_rate = random.uniform(0.01, 0.05)                 # comments per viewer-minute proxy
    total_comments = int(average_viewers * duration_minutes * comments_rate)
    total_likes = int(total_comments * random.uniform(1.2, 3.0))

    # Interaction metrics
    total_viewers = max(peak_viewers, int(peak_viewers * random.uniform(1.0, 2.5)))
    engagement_rate = round(random.uniform(0.35, 0.9), 2)     # 0..1
    retention_rate = round(random.uniform(0.3, 0.85), 2)      # 0..1
    chat_activity = round(random.uniform(0.45, 0.95), 2)      # 0..1

    # Time placement
    day_of_week = random.randint(0, 6)                         # 0=Mon .. 6=Sun
    is_weekend = 1 if day_of_week in (5, 6) else 0
    time_of_day = random.randint(12, 23)                       # typical streaming hours

    topics = [
        "Q&A chill session",
        "Speedrun challenge",
        "Cooking & chat",
        "Music practice live",
        "Art & doodles",
        "Tech & gadgets",
        "Unboxing & reviews",
        "IRL hangout",
        "Study with me",
        "Fitness & wellness"
    ]
    tags = [
        "#live", "#streaming", "#vibes", "#community", "#AMA",
        "#fun", "#learn", "#chill", "#hype", "#creators"
    ]
    topic = random.choice(topics)
    transcript = (
        f"Hey everyone, welcome back! Today’s stream: {topic}. "
        f"Drop your questions in chat and don’t forget to like. {random.choice(tags)} {random.choice(tags)}"
    )

    content = {
        "stream_id": stream_id,
        "keyframes": keyframes,  # <-- now uses generated keyframes
        "transcript": transcript,
        "basic_metrics": {
            "duration_minutes": duration_minutes,
            "peak_viewers": peak_viewers,
            "average_viewers": average_viewers,
            "total_comments": total_comments,
            "total_likes": total_likes
        },
        "interaction_metrics": {
            "engagement_rate": engagement_rate,
            "retention_rate": retention_rate,
            "total_viewers": total_viewers,
            "chat_activity": chat_activity
        },
        "creator_profile": {
            # Keep the creator profile consistent, but you could add tiny noise per stream if desired
            "historical_avg_score": base_profile["historical_avg_score"],
            "follower_count": base_profile["follower_count"],
            "consistency_score": base_profile["consistency_score"],
            "experience_months": base_profile["experience_months"]
        },
        "duration_hours": duration_hours,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend
    }

    return {
        "creator_id": creator_id,
        "stream_id": stream_id,
        "content": content
    }

# ----------------------------
# Generate dummy_histories for creatorA..creatorE
# ----------------------------
def generate_dummy_histories(creators=CREATORS, streams_per_creator=STREAMS_PER_CREATOR):
    histories = []

    # Build a base profile per creator (consistent across their streams)
    def base_profile_for(creator):
        # Vary bands per creator to get diverse distributions
        creator_idx = creators.index(creator)
        follower_bands = [
            (3_000, 12_000),
            (8_000, 25_000),
            (10_000, 40_000),
            (20_000, 80_000),
            (50_000, 150_000),
        ]
        low, high = follower_bands[min(creator_idx, len(follower_bands)-1)]
        followers = random.randint(low, high)

        return {
            "historical_avg_score": round(random.uniform(0.55, 0.9), 2),
            "follower_count": followers,
            "consistency_score": round(random.uniform(0.65, 0.95), 2),
            "experience_months": random.randint(2, 36)
        }

    for creator in creators:
        base_prof = base_profile_for(creator)
        for stream_num in range(1, streams_per_creator + 1):
            histories.append(_random_stream_payload(creator, base_prof, stream_num))

    return histories

# Create the global dummy_histories variable expected by your existing function
dummy_histories = generate_dummy_histories()

# ----------------------------
# Ensure the table exists with a composite primary key
# so INSERT raises IntegrityError on duplicates (creator_id, stream_id)
# ----------------------------
def ensure_table():
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS live_history (
            creator_id   TEXT NOT NULL,
            stream_id    TEXT NOT NULL,
            content_json TEXT NOT NULL,
            timestamp    TEXT NOT NULL,
            PRIMARY KEY (creator_id, stream_id)
        )
    """)
    conn.commit()
    conn.close()

# ----------------------------
def store_dummy_live_history():
    ensure_table()  # safe to call; creates if not exist
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    for entry in dummy_histories:
        try:
            c.execute("""
                INSERT INTO live_history (creator_id, stream_id, content_json, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                entry["creator_id"],
                entry["stream_id"],
                json.dumps(entry["content"]),
                now
            ))
        except sqlite3.IntegrityError:
            # If already exists, update
            c.execute("""
                UPDATE live_history SET content_json=?, timestamp=?
                WHERE creator_id=? AND stream_id=?
            """, (
                json.dumps(entry["content"]),
                now,
                entry["creator_id"],
                entry["stream_id"]
            ))
    conn.commit()
    conn.close()
    print(f"Dummy live history data stored. Total rows attempted: {len(dummy_histories)}")

# ----------------------------
# Run it
# ----------------------------
if __name__ == "__main__":
    store_dummy_live_history()
