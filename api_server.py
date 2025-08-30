from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
import logging
import os
import base64
import json
import openai
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API
API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

# Create FastAPI application
app = FastAPI(
    title="TikTok Revenue Allocation AI Assistant",
    description="DPO-trained revenue allocation recommendation system + multimodal live stream content evaluation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model and tokenizer
model = None
tokenizer = None
text_streamer = None
multimodal_evaluator = None

# Request model - Revenue allocation
class RevenueAllocationRequest(BaseModel):
    creator_id: Optional[str] = None
    creator_experience_months: int
    follower_count: int
    historical_performance: float
    consistency_score: float
    content_value: float # evaluated by multimodal_evaluator
    visual_quality: float # evaluated by multimodal_evaluator
    host_performance: float # evaluated by multimodal_evaluator
    overall_rating: float # evaluated by multimodal_evaluator
    viewer_count: int
    engagement_rate: float
    retention_rate: float
    chat_activity: float
    live_duration_minutes: int
    peak_viewers: int
    avg_viewers: int
    comment_count: int
    like_count: int

# Request model - Live stream evaluation
class LiveStreamEvaluationRequest(BaseModel):
    stream_id: str
    keyframes: List[Dict[str, Any]]
    transcript: Optional[str] = ""
    basic_metrics: Dict[str, Any]
    interaction_metrics: Optional[Dict[str, Any]] = None
    creator_profile: Optional[Dict[str, Any]] = None
    duration_hours: Optional[float] = None
    time_of_day: Optional[int] = None
    day_of_week: Optional[int] = None
    is_weekend: Optional[int] = None

# Response model - Revenue allocation
class RevenueAllocationResponse(BaseModel):
    recommended_percentage: float
    reasoning: str
    confidence_score: float
    model_version: str

# Response model - Live stream evaluation
class LiveStreamEvaluationResponse(BaseModel):
    stream_id: str
    evaluation_result: Dict[str, Any]
    evaluation_timestamp: str
    model_version: str

class MultimodalLLMEvaluator:
    """Multimodal LLM evaluator"""
    
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
            # Return default rating
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
            for frame in stream_data["keyframes"][:5]:  # Max 5 screenshots
                if "image_data" in frame:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame['image_data']}"}
                    })
        
        # Add transcript text
        if "transcript" in stream_data:
            transcript_text = f"\n\n直播转录文本：\n{stream_data['transcript']}"
            messages[0]["content"][0]["text"] += transcript_text
        
        # Add basic statistics
        if "basic_metrics" in stream_data:
            metrics_text = f"\n\n直播数据：\n{json.dumps(stream_data['basic_metrics'], ensure_ascii=False, indent=2)}"
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
                "visual_quality": "Default rating",
                "content_value": "Default rating", 
                "host_performance": "Default rating"
            }
        }

def load_model():
    """Load DPO trained model"""
    global model, tokenizer, text_streamer
    
    try:
        model_name = "/root/tiktok_techjam/policy_learning/final_dpo_model"
        
        # Check if model path exists
        if not os.path.exists(model_name):
            raise FileNotFoundError(f"Model path not found: {model_name}")
        
        logger.info("Loading model...")
        
        # Model loading parameters
        max_seq_length = 2048
        dtype = None  # Auto detect
        load_in_4bit = True
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        text_streamer = TextStreamer(tokenizer)
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return False

def create_prompt(request: RevenueAllocationRequest) -> str:
    """根据请求数据创建提示词"""
    prompt = f"""作为TikTok收益分配专家，请为以下创作者推荐一个公平的收益分成比例。

创作者档案：
- 创作经验：{request.creator_experience_months}个月
- 粉丝数量：{request.follower_count}
- 历史表现：{request.historical_performance:.2f}
- 稳定性评分：{request.consistency_score:.2f}

内容质量评估：
- 内容价值：{request.content_value:.2f}
- 视觉质量：{request.visual_quality:.2f}
- 主播表现：{request.host_performance:.2f}
- 整体评分：{request.overall_rating:.2f}

互动表现：
- 观看人数：{request.viewer_count}
- 参与率：{request.engagement_rate:.2f}
- 留存率：{request.retention_rate:.2f}
- 聊天活跃度：{request.chat_activity:.2f}

基础数据：
- 直播时长：{request.live_duration_minutes}分钟
- 峰值观众：{request.peak_viewers}
- 平均观众：{request.avg_viewers}
- 评论数：{request.comment_count}
- 点赞数：{request.like_count}

提供一个公平的创作者收益分成比例(0-100%)和简短的理由"""
    
    return prompt

def extract_percentage_and_reasoning(response_text: str) -> tuple[float, str]:
    """Extract percentage and reasoning from model response"""
    try:
        # Simple text parsing logic
        lines = response_text.strip().split('\n')
        
        # Find sentences containing percentages
        percentage = None
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            if '%' in line or '百分比' in line or '比例' in line:
                # Extract numbers
                import re
                numbers = re.findall(r'\d+(?:\.\d+)?', line)
                if numbers:
                    percentage = float(numbers[0])
                    if percentage > 100:  # If over 100, it might be in decimal form
                        percentage = percentage / 100
                    break
        
        # If no clear percentage found, use the entire response as reasoning
        if percentage is None:
            percentage = 50.0  # Default value
            reasoning = response_text
        else:
            # Extract reasoning part (usually after percentage)
            reasoning = response_text
        
        return percentage, reasoning
        
    except Exception as e:
        logger.warning(f"Parsing response failed: {str(e)}")
        return 50.0, response_text

@app.on_event("startup")
async def startup_event():
    """Load model when application starts"""
    global multimodal_evaluator
    
    logger.info("Starting Revenue Allocation AI Assistant...")
    
    # Load DPO model
    success = load_model()
    if not success:
        logger.error("DPO model loading failed, application cannot start")
        raise RuntimeError("DPO model loading failed")
    
    # Initialize multimodal evaluator
    try:
        multimodal_evaluator = MultimodalLLMEvaluator()
        logger.info("Multimodal evaluator initialized successfully!")
    except Exception as e:
        logger.error(f"Multimodal evaluator initialization failed: {str(e)}")
        raise RuntimeError("Multimodal evaluator initialization failed")

@app.get("/")
async def root():
    """Root path, return service status"""
    return {
        "message": "TikTok Revenue Allocation AI Assistant + Multimodal Live Stream Evaluation",
        "status": "Running",
        "version": "1.0.0",
        "features": ["Revenue Allocation Recommendation", "Live Stream Evaluation"]
    }

@app.get("/health")
async def health_check():
    """Health check interface"""
    return {
        "status": "healthy",
        "dpo_model_loaded": model is not None and tokenizer is not None,
        "multimodal_evaluator_loaded": multimodal_evaluator is not None
    }

@app.post("/api/revenue-allocation", response_model=RevenueAllocationResponse)
async def get_revenue_allocation(request: RevenueAllocationRequest):
    """Get revenue allocation recommendation"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Create prompt
        prompt = create_prompt(request)
        
        # Prepare input
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        
        # Generate response
        logger.info("Generating response...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract part after prompt
        if prompt in response_text:
            response_text = response_text[len(prompt):].strip()
        
        # Parse response
        percentage, reasoning = extract_percentage_and_reasoning(response_text)
        
        # Calculate confidence score (based on response length and quality)
        confidence_score = min(1.0, len(reasoning) / 100.0)
        
        logger.info(f"Recommended percentage: {percentage}%, Confidence: {confidence_score}")
        
        return RevenueAllocationResponse(
            recommended_percentage=percentage,
            reasoning=reasoning,
            confidence_score=confidence_score,
            model_version="DPO-v1.0"
        )
        
    except Exception as e:
        logger.error(f"Error during reasoning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

@app.post("/api/live-stream-evaluation", response_model=LiveStreamEvaluationResponse)
async def evaluate_live_stream(request: LiveStreamEvaluationRequest):
    """Evaluate live stream content quality"""
    try:
        if multimodal_evaluator is None:
            raise HTTPException(status_code=503, detail="Multimodal evaluator not loaded")
        
        # Build stream data
        stream_data = {
            "stream_id": request.stream_id,
            "keyframes": request.keyframes,
            "transcript": request.transcript,
            "basic_metrics": request.basic_metrics,
            "interaction_metrics": request.interaction_metrics or {},
            "creator_profile": request.creator_profile or {},
            "duration_hours": request.duration_hours,
            "time_of_day": request.time_of_day,
            "day_of_week": request.day_of_week,
            "is_weekend": request.is_weekend
        }
        
        logger.info(f"Evaluating live stream: {request.stream_id}")
        
        # 执行评估
        evaluation_result = multimodal_evaluator.evaluate_live_stream(stream_data)
        
        logger.info(f"Live stream evaluation completed: {request.stream_id}")
        
        return LiveStreamEvaluationResponse(
            stream_id=request.stream_id,
            evaluation_result=evaluation_result,
            evaluation_timestamp=datetime.now().isoformat(),
            model_version="GPT-4o-v1.0"
        )
        
    except Exception as e:
        logger.error(f"Live stream evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Live stream evaluation failed: {str(e)}")

@app.post("/api/batch-revenue-allocation")
async def batch_revenue_allocation(requests: list[RevenueAllocationRequest]):
    """Batch get revenue allocation recommendation"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        
        for i, request in enumerate(requests):
            try:
                # Create prompt
                prompt = create_prompt(request)
                
                # Prepare input
                inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
                
                # Generate response
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode response
                response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract part after prompt
                if prompt in response_text:
                    response_text = response_text[len(prompt):].strip()
                
                # Parse response
                percentage, reasoning = extract_percentage_and_reasoning(response_text)
                confidence_score = min(1.0, len(reasoning) / 100.0)
                
                results.append({
                    "index": i,
                    "creator_id": request.creator_id,
                    "recommended_percentage": percentage,
                    "reasoning": reasoning,
                    "confidence_score": confidence_score,
                    "model_version": "DPO-v1.0"
                })
                
            except Exception as e:
                logger.error(f"Error during batch revenue allocation: {str(e)}")
                results.append({
                    "index": i,
                    "creator_id": request.creator_id,
                    "error": str(e),
                    "recommended_percentage": 50.0,
                    "reasoning": "Batch revenue allocation failed",
                    "confidence_score": 0.0,
                    "model_version": "DPO-v1.0"
                })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error during batch revenue allocation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch revenue allocation failed: {str(e)}")

@app.post("/api/batch-live-stream-evaluation")
async def batch_live_stream_evaluation(requests: list[LiveStreamEvaluationRequest]):
    """Batch evaluate live stream content"""
    try:
        if multimodal_evaluator is None:
            raise HTTPException(status_code=503, detail="Multimodal evaluator not loaded")
        
        results = []
        
        for i, request in enumerate(requests):
            try:
                # Build stream data
                stream_data = {
                    "stream_id": request.stream_id,
                    "keyframes": request.keyframes,
                    "transcript": request.transcript,
                    "basic_metrics": request.basic_metrics,
                    "interaction_metrics": request.interaction_metrics or {},
                    "creator_profile": request.creator_profile or {},
                    "duration_hours": request.duration_hours,
                    "time_of_day": request.time_of_day,
                    "day_of_week": request.day_of_week,
                    "is_weekend": request.is_weekend
                }
                
                # Execute evaluation
                evaluation_result = multimodal_evaluator.evaluate_live_stream(stream_data)
                
                results.append({
                    "index": i,
                    "stream_id": request.stream_id,
                    "evaluation_result": evaluation_result,
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "model_version": "GPT-4o-v1.0"
                })
                
            except Exception as e:
                logger.error(f"Error during batch live stream evaluation: {str(e)}")
                results.append({
                    "index": i,
                    "stream_id": request.stream_id,
                    "error": str(e),
                    "evaluation_result": multimodal_evaluator._get_default_evaluation(),
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "model_version": "GPT-4o-v1.0"
                })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error during batch live stream evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch live stream evaluation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Start server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
