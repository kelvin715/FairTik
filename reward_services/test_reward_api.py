import requests
import base64

BASE_URL = "http://localhost:8001"

def test_calculate_actual_rewards():
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
            },
            {
                "index": 2,
                "creator_id": "creatorC",
                "stream_id": "stream003",
                "recommended_percentage": 100.0,
                "reasoning": "Excellent performance",
                "confidence_score": 1.0,
                "model_version": "DPO-v1.0",
                "coins": 3000
            },
            {
                "index": 3,
                "creator_id": "creatorA",
                "stream_id": "stream004",
                "recommended_percentage": 50.0,
                "reasoning": "Average engagement",
                "confidence_score": 0.7,
                "model_version": "DPO-v1.0",
                "coins": 800
            }
        ]
    }
    resp = requests.post(f"{BASE_URL}/api/calculate-actual-rewards", json=payload)
    print("Calculate Actual Rewards Response:")
    print(resp.json())
    return resp.json()

def test_store_and_get_reward_result(results):
    # Store results in SQLite
    resp = requests.post(f"{BASE_URL}/api/store-reward-result", json=results)
    print("Store Reward Result Response:")
    print(resp.json())

    # Get result for creatorA, stream001
    resp = requests.get(f"{BASE_URL}/api/get-reward-result/creatorA/stream001")
    print("Get Reward Result for creatorA, stream001:")
    print(resp.json())

    # Get result for creatorB, stream002
    resp = requests.get(f"{BASE_URL}/api/get-reward-result/creatorB/stream002")
    print("Get Reward Result for creatorB, stream002:")
    print(resp.json())

    # Get result for creatorC, stream003
    resp = requests.get(f"{BASE_URL}/api/get-reward-result/creatorC/stream003")
    print("Get Reward Result for creatorC, stream003:")
    print(resp.json())

    # Get result for creatorA, stream004
    resp = requests.get(f"{BASE_URL}/api/get-reward-result/creatorA/stream004")
    print("Get Reward Result for creatorA, stream004:")
    print(resp.json())

    # Get reward history for creatorA
    resp = requests.get(f"{BASE_URL}/api/reward-history/creatorA")
    print("Reward History for creatorA:")
    print(resp.json())

    # Get reward history for creatorB
    resp = requests.get(f"{BASE_URL}/api/reward-history/creatorB")
    print("Reward History for creatorB:")
    print(resp.json())

    # Get reward history for creatorC
    resp = requests.get(f"{BASE_URL}/api/reward-history/creatorC")
    print("Reward History for creatorC:")
    print(resp.json())

def test_duplicate_store(results):
    # Try storing the same results again to test duplicate prevention
    resp = requests.post(f"{BASE_URL}/api/store-reward-result", json=results)
    print("Duplicate Store Reward Result Response (should skip all):")
    print(resp.json())

def test_delete_reward_result():
    # Delete a reward result
    resp = requests.delete(f"{BASE_URL}/api/delete-reward-result/creatorA/stream001")
    print("Delete Reward Result for creatorA, stream001:")
    print(resp.json())
    # Try to get the deleted result (should 404)
    resp = requests.get(f"{BASE_URL}/api/get-reward-result/creatorA/stream001")
    print("Get Deleted Reward Result for creatorA, stream001 (should 404):")
    print(resp.status_code, resp.json())

def test_store_and_get_live_history():
    # Example base64 image (shortened for demo)
    example_image_b64 = base64.b64encode(b"fake_image_data").decode("utf-8")
    test_data = {
        "stream_id": "test_stream_001",
        "keyframes": [
            {"timestamp": 300, "image_data": example_image_b64},
            {"timestamp": 600, "image_data": example_image_b64},
            {"timestamp": 900, "image_data": example_image_b64}
        ] if example_image_b64 else [],
        "transcript": "Welcome to my stream. Today I present a toy kitchen. Please like and follow!",
        "basic_metrics": {
            "duration_minutes": 90,
            "peak_viewers": 1500,
            "average_viewers": 1200,
            "total_comments": 3200,
            "total_likes": 8500
        },
        "interaction_metrics": {
            "engagement_rate": 0.75,
            "retention_rate": 0.68,
            "total_viewers": 1500,
            "chat_activity": 0.82
        },
        "creator_profile": {
            "historical_avg_score": 0.78,
            "follower_count": 15000,
            "consistency_score": 0.85,
            "experience_months": 8
        },
        "duration_hours": 1.5,
        "time_of_day": 20,
        "day_of_week": 6,
        "is_weekend": 1
    }
    payload = {
        "creator_id": "creatorA",
        "stream_id": "test_stream_001",
        "content": test_data
    }
    resp = requests.post(f"{BASE_URL}/api/store-live-history", json=payload)
    print("Store Live History Response:")
    print(resp.json())

    # Get live history
    resp = requests.get(f"{BASE_URL}/api/get-live-history/creatorA/test_stream_001")
    print("Get Live History for creatorA, test_stream_001:")
    print(resp.json())

if __name__ == "__main__":
    results = test_calculate_actual_rewards()
    test_store_and_get_reward_result(results)
    test_duplicate_store(results)
    test_delete_reward_result()
    test_store_and_get_live_history()