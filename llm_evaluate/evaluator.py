import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import base64
from typing import Dict, List, Any, Optional
import openai
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API
API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

class MultimodalLLMEvaluator:
    """Multimodal LLM Evaluator"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=API_KEY)
        
    def evaluate_live_stream(self, stream_data: Dict) -> Dict:
        """Evaluate live stream content quality"""
        
        # Build evaluation prompt
        evaluation_prompt = self._build_evaluation_prompt()
        
        # Prepare multimodal input
        messages = self._prepare_multimodal_input(stream_data, evaluation_prompt)
        
        try:
            # Call GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            
            # Parse response
            evaluation_result = self._parse_evaluation_response(response.choices[0].message.content)
            
            logger.info(f"AI evaluation completed for stream {stream_data.get('stream_id', 'unknown')}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"AI evaluation failed: {e}")
            # Return default evaluation
            return self._get_default_evaluation()
    
    def _build_evaluation_prompt(self) -> str:
        """Build evaluation prompt"""
        return """
请作为专业的直播内容评估专家，对这场直播进行多维度评分。

评估维度：
1. **视觉质量** (0-1分)：画面清晰度、光线、构图、稳定性
2. **内容价值** (0-1分)：教育价值、娱乐价值、实用价值、创新性
3. **主播表现** (0-1分)：语言表达、互动质量、专业度、感染力
4. **整体评估** (0-1分)：综合观看体验和推荐价值

请基于提供的视频截图、转录文本和直播数据给出客观评分。

输出格式（严格JSON）：
{
    "reasoning": {
        "visual_quality": "画面清晰稳定，光线良好",
        "content_value": "教育内容丰富，实用性强",
        "host_performance": "表达流畅，互动积极"
    },
    "visual_quality": 0.85,
    "content_value": 0.92,
    "host_performance": 0.88,
    "overall_rating": 0.86,
    "content_category": "educational"
}
"""
    
    def _prepare_multimodal_input(self, stream_data: Dict, prompt: str) -> List:
        """Prepare multimodal input"""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        # Add video screenshots
        if "keyframes" in stream_data:
            for frame in stream_data["keyframes"][:5]:  # Maximum 5 screenshots
                if "image_data" in frame:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame['image_data']}"}
                    })
        
        # Add transcript text
        if "transcript" in stream_data:
            transcript_text = f"\n\nLive transcript text:\n{stream_data['transcript']}"
            messages[0]["content"][0]["text"] += transcript_text
        
        # Add basic statistics
        if "basic_metrics" in stream_data:
            metrics_text = f"\n\nLive data:\n{json.dumps(stream_data['basic_metrics'], ensure_ascii=False, indent=2)}"
            messages[0]["content"][0]["text"] += metrics_text
        
        return messages
    
    def _parse_evaluation_response(self, response_text: str) -> Dict:
        """Parse evaluation response"""
        try:
            # Extract JSON part
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Validate and clean data
                return self._validate_evaluation_result(result)
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse evaluation response: {e}")
            return self._get_default_evaluation()
    
    def _validate_evaluation_result(self, result: Dict) -> Dict:
        """Validate and clean evaluation result"""
        required_fields = ["visual_quality", "content_value", "host_performance", "overall_rating"]
        
        for field in required_fields:
            if field not in result:
                result[field] = 0.5
            else:
                result[field] = max(0.0, min(1.0, float(result[field])))
        
        if "content_category" not in result:
            result["content_category"] = "general"
        
        if "reasoning" not in result:
            result["reasoning"] = {field: "Evaluation completed" for field in required_fields}
        
        return result
    
    def _get_default_evaluation(self) -> Dict:
        """Get default evaluation result"""
        return {
            "visual_quality": 0.5,
            "content_value": 0.5,
            "host_performance": 0.5,
            "overall_rating": 0.5,
            "content_category": "general",
            "reasoning": {
                "visual_quality": "默认评分",
                "content_value": "默认评分", 
                "host_performance": "默认评分"
            }
        }


def simulate_stream_data(stream_id: str) -> Dict:
    """Simulate live stream data (for testing)"""

    # Utility function and image processing
    def load_example_image_as_base64(image_path: str) -> str:
        """Load example image and convert to base64"""
        import base64
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return ""
    
    # Load real example image
    example_image_b64 = load_example_image_as_base64("/root/tiktok_techjam/test_data/live_video_frames/image.png")
    
    return {
        "stream_id": stream_id,
        "keyframes": [
            {"timestamp": 300, "image_data": example_image_b64},
            {"timestamp": 600, "image_data": example_image_b64},  # Use the same image as an example
            {"timestamp": 900, "image_data": example_image_b64}
        ] if example_image_b64 else [],  # If image loading fails, do not add image data
        "transcript": "Hello, everyone. Welcome to my live stream. Today I brought a toy kitchen for you. Please like and follow me. Thank you.",
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
            "experience_months": 8  # 8 months experience (not a newbie)
        },
        "duration_hours": 1.5,
        "time_of_day": 20,  
        "day_of_week": 6,  
        "is_weekend": 1
    }

if __name__ == "__main__":
    multimodal_evaluator = MultimodalLLMEvaluator()

    stream_data = simulate_stream_data("stream_001")
    ai_evaluation = multimodal_evaluator.evaluate_live_stream(stream_data)
    print(ai_evaluation)
    
    
