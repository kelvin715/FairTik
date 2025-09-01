# TikTok Revenue Allocation AI Assistant

A DPO-trained AI system for fair revenue allocation recommendations and multimodal live stream content evaluation.

## 🎯 Project Overview

This project implements an AI-powered revenue allocation system for TikTok creators using Direct Preference Optimization (DPO) training methodology. The system provides fair revenue distribution recommendations based on multiple fairness principles and includes multimodal evaluation capabilities for live stream content quality assessment.

## ✨ Key Features

### 1. **DPO-Trained Revenue Allocation**
- Fair revenue distribution recommendations based on multiple criteria
- Trained using Direct Preference Optimization for alignment with fairness principles
- Supports both single and batch processing requests

### 2. **Multimodal Live Stream Evaluation**
- Real-time content quality assessment using GPT-4o
- Visual analysis of live stream keyframes
- Text-based content evaluation from transcripts
- Comprehensive engagement metrics analysis

### 3. **Fairness-First Design**
- **Content Quality Priority**: High-quality content receives higher revenue share
- **Traffic Contribution Reward**: High-engagement content gets better allocation
- **Newbie Support**: Preferential treatment for new creators (< 6 months)
- **Niche Protection**: Support for high-quality content with smaller audiences
- **Loyalty Reward**: Better treatment for consistent long-term creators
- **Geographic Fairness**: Equal treatment across different regions
- **Content Diversity**: Encouragement for diverse content creation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- OpenAI API key for multimodal evaluation

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tiktok_techjam
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

4. Start the API server:
```bash
bash start_server.sh
```

The server will start at `http://localhost:8000` with interactive API documentation available at `http://localhost:8000/docs`.

## 📚 API Documentation

### Health Check
```
GET /health
```

### Revenue Allocation Endpoints

#### Single Revenue Allocation
```
POST /api/revenue-allocation
```

**Request Body:**
```json
{
  "creator_experience_months": 1,
  "follower_count": 50000,
  "historical_performance": 0.57,
  "consistency_score": 0.37,
  "content_value": 0.83,
  "visual_quality": 0.59,
  "host_performance": 0.66,
  "overall_rating": 0.69,
  "viewer_count": 14574,
  "engagement_rate": 0.12,
  "retention_rate": 0.58,
  "chat_activity": 0.44,
  "live_duration_minutes": 120,
  "peak_viewers": 14574,
  "avg_viewers": 12515,
  "comment_count": 23523,
  "like_count": 29275
}
```

#### Batch Revenue Allocation
```
POST /api/batch-revenue-allocation
```

### Live Stream Evaluation Endpoints

#### Single Live Stream Evaluation
```
POST /api/live-stream-evaluation
```

**Request Body:**
```json
{
  "stream_id": "stream_001",
  "keyframes": [
    {
      "timestamp": 300,
      "image_data": "base64_encoded_image"
    }
  ],
  "transcript": "Live stream transcript text",
  "basic_metrics": {
    "duration_minutes": 90,
    "peak_viewers": 1500,
    "average_viewers": 1200,
    "total_comments": 3200,
    "total_likes": 8500
  }
}
```

#### Batch Live Stream Evaluation
```
POST /api/batch-live-stream-evaluation
```

## 🏗️ Project Structure

```
tiktok_techjam/
├── api_server.py                 # Main FastAPI server
├── requirements.txt              # Python dependencies
├── start_server.sh              # Server startup script
├── llm_evaluate/                # Multimodal evaluation module
│   └── evaluator.py            # GPT-4o based content evaluator
├── policy_learning/             # DPO training and dataset generation
│   ├── fairness_dataset_generator.py  # Fairness preference dataset generator
│   ├── dpo_data_formatter.py    # DPO training data formatter
│   ├── regenerate_diverse_dataset.py  # Dataset diversity enhancement
│   ├── unsloth_DPO.ipynb       # DPO training notebook
│   ├── dataset/                # Training datasets
│   ├── final_dpo_model/        # Trained DPO model
│   └── outputs/                # Training checkpoints
├── test_data/                  # Test assets
│   ├── audio_samples/
│   └── live_video_frames/
├── test_revenue_allocation_api.py    # Revenue allocation API tests
└── test_live_evaluation_api.py       # Live evaluation API tests
```

## 🔬 Model Training

### Dataset Generation

The project includes sophisticated dataset generation capabilities:

```bash
cd policy_learning
python fairness_dataset_generator.py
```

This generates diverse preference pairs based on multiple fairness principles, creator diversity, and content categories.

### DPO Training

The model is trained using the Unsloth framework with Direct Preference Optimization:

1. Open `policy_learning/unsloth_DPO.ipynb` in Jupyter
2. Follow the notebook to train the model with your preference dataset
3. The trained model will be saved to `policy_learning/final_dpo_model/`

## 🧪 Testing

### Test Revenue Allocation API
```bash
python test_revenue_allocation_api.py
```

### Test Live Stream Evaluation API
```bash
python test_live_evaluation_api.py
```

## 📊 Fairness Principles

The system implements a comprehensive fairness framework:

| Principle | Weight | Description |
|-----------|--------|-------------|
| Content Quality | 25% | Prioritize high-quality content |
| Traffic Contribution | 20% | Reward high-engagement content |
| Newbie Support | 15% | Support new creators |
| Niche Protection | 15% | Protect quality niche content |
| Loyalty Reward | 10% | Reward consistent creators |
| Geographic Fairness | 5% | Ensure regional equity |
| Content Diversity | 5% | Encourage diverse content |
| Peak Performance | 5% | Reward peak-hour performance |

## 🛠️ Technical Stack

- **Backend**: FastAPI with Python 3.8+
- **ML Framework**: PyTorch, Transformers, Unsloth
- **Model**: DPO-trained language model based on Qwen2.5-1.5B
- **Multimodal AI**: OpenAI GPT-4o for vision and text analysis
- **Training**: Direct Preference Optimization (DPO)
- **Deployment**: Uvicorn ASGI server

## 📝 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for multimodal evaluation

### Model Configuration

The DPO model is configured for:
- Base model: Qwen2.5-1.5B-Instruct
- LoRA rank: 16
- Learning rate: 2e-4
- Training steps: 100+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or support, please:
1. Check the API documentation at `/docs` when the server is running
2. Review the test files for usage examples
3. Open an issue for bug reports or feature requests

## 🏆 Features Highlights

- **State-of-the-art DPO Training**: Uses the latest Direct Preference Optimization techniques
- **Multimodal Evaluation**: Combines visual and textual analysis for comprehensive assessment
- **Fairness-First Approach**: Implements multiple fairness principles for equitable revenue distribution
- **Production Ready**: Full API server with comprehensive testing and documentation
- **Scalable Design**: Supports both single and batch processing for high-throughput scenarios

---

# AI-Native Analytics: Cross-Platform UI with Lynx

## 🎯 Problem
LLMs and AI systems are transforming user interaction. Traditional UI paradigms can't handle natural language queries, real-time AI responses, and intelligent data visualization.

## 🤖 Solution
AI-native livestream analytics platform using **Lynx** (TikTok's cross-platform UI framework):

- **Intelligent Chatbot Interface**: Natural language data analysis
- **Natural Language Data Filtering**: Convert queries to structured filters
- **Cross-Platform**: Single codebase for iOS, Android, Web

## 🦊 Why Lynx?
- **Modern Visual Design**: Gradient thinking boxes and modern UI effects with one line of CSS
- **Main Thread Optimization**: Dual-threaded architecture with instant first-frame rendering and non-blocking UI
- **Cross-Device Sync**: Seamless state synchronization across iOS, Android, Web

## 🎨 AI-Era Patterns
- **Natural Language First**: Redefine data interaction
- **Streaming Responses**: Real-time AI analysis
- **Conversational Analytics**: Multi-turn AI conversations
- **Intelligent Filtering**: AI-powered query conversion

## 🏆 Innovation
- **AI-Native Design**: Interfaces built for AI collaboration
- **Cross-Platform AI**: Seamless AI capabilities across platforms
- **Future-Ready**: Foundation for AI-generated UI

## 🚀 Setup
See **[Frontend Guide](https://github.com/Cicici-Shi/lynx-starter#readme)** for details.

---