import json
import torch
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class DPOTrainingExample:
    """DPO training example"""
    prompt: str
    chosen: str  # More fair answer
    rejected: str  # Less fair answer

class DPODataFormatter:
    """Convert preference dataset to DPO training format"""
    
    def __init__(self):
        pass
    
    def format_dataset_for_dpo(self, preference_dataset: Dict) -> List[DPOTrainingExample]:
        """Convert preference dataset to DPO training format"""
        dpo_examples = []
        
        for pair in preference_dataset["preference_pairs"]:
            # Only process high confidence samples
            if pair.get("confidence", 0) < 0.7:
                continue
                
            # Build input prompt
            prompt = self._build_prompt(pair)
            
            # Determine chosen and rejected
            if pair["preferred"] == "a":
                chosen = self._format_response(pair["proposal_a"])
                rejected = self._format_response(pair["proposal_b"])
            else:
                chosen = self._format_response(pair["proposal_b"])
                rejected = self._format_response(pair["proposal_a"])
            
            dpo_examples.append(DPOTrainingExample(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected
            ))
        
        return dpo_examples
    
    def _build_prompt(self, pair: Dict) -> str:
        """Build input prompt"""
        scenario = pair["proposal_a"]["scenario_data"]
        
        prompt = f"""As a TikTok revenue allocation expert, please recommend a fair revenue share percentage for the following creator.

Creator Profile:
- Creation Experience: {scenario['creator_profile']['experience_months']} months
- Follower Count: {scenario['creator_profile']['follower_count']}
- Historical Performance: {scenario['creator_profile']['historical_avg_score']:.2f}
- Consistency Score: {scenario['creator_profile']['consistency_score']:.2f}

Content Quality Assessment:
- Content Value: {scenario['ai_evaluation']['content_value']:.2f}
- Visual Quality: {scenario['ai_evaluation']['visual_quality']:.2f}
- Host Performance: {scenario['ai_evaluation']['host_performance']:.2f}
- Overall Rating: {scenario['ai_evaluation']['overall_rating']:.2f}

Interaction Performance:
- Viewer Count: {scenario['interaction_metrics']['total_viewers']}
- Engagement Rate: {scenario['interaction_metrics']['engagement_rate']:.2f}
- Retention Rate: {scenario['interaction_metrics']['retention_rate']:.2f}
- Chat Activity: {scenario['interaction_metrics']['chat_activity']:.2f}

Basic Data:
- Live Duration: {scenario['basic_metrics']['duration_minutes']} minutes
- Peak Viewers: {scenario['basic_metrics']['peak_viewers']}
- Average Viewers: {scenario['basic_metrics']['average_viewers']}
- Comment Count: {scenario['basic_metrics']['total_comments']}
- Like Count: {scenario['basic_metrics']['total_likes']}

Please consider the following fairness principles:
1. Content Quality Priority (40% weight): High-quality content should receive higher revenue share
2. New Creator Support (25% weight): New creators (<6 months) should receive preferential policies
3. Niche Protection (20% weight): High-quality content with fewer viewers needs protection
4. Loyalty Reward (15% weight): Long-term stable creators should receive better treatment

Please provide a fair creator revenue share percentage (0-100%) and detailed reasoning, first output detailed thinking process, then output final answer:"""
        
        return prompt
    
    def _format_response(self, proposal: Dict) -> str:
        """Format response"""
        return f"Reasoning: {proposal['reasoning']}\n\nRecommended creator revenue share percentage: {proposal['creator_share_percentage']}%"
    
    def save_dpo_dataset(self, dpo_examples: List[DPOTrainingExample], filename: str = "/root/tiktok_techjam/policy_learning/dataset/dpo_training_dataset.json"):
        """Save DPO training dataset"""
        data = []
        for example in dpo_examples:
            data.append({
                "prompt": example.prompt,
                "chosen": example.chosen, 
                "rejected": example.rejected
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"DPO training dataset saved to {filename}")
        print(f"Total samples: {len(data)}")
    
    def save_for_huggingface_trl(self, dpo_examples: List[DPOTrainingExample], filename: str = "/root/tiktok_techjam/policy_learning/dataset/dpo_dataset_hf.json"):
        """Save for Hugging Face TRL"""
        data = []
        for example in dpo_examples:
            data.append({
                "prompt": example.prompt,
                "chosen": example.chosen,
                "rejected": example.rejected,
                "system": "You are a TikTok revenue allocation expert, please recommend a fair revenue share percentage for the following creator."
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"HuggingFace TRL format dataset saved to {filename}")

def load_preference_dataset(filename: str = "/root/tiktok_techjam/policy_learning/dataset/fairness_preference_dataset.json") -> Dict:
    """Load preference dataset"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """Main function: convert preference dataset to DPO training format"""
    print("Starting to convert preference dataset to DPO training format...")
    
    # Load preference dataset
    try:
        preference_dataset = load_preference_dataset()
        print(f"Successfully loaded preference dataset, containing {len(preference_dataset['preference_pairs'])} preference pairs")
    except FileNotFoundError:
        print("Preference dataset file not found, please run fairness_dataset_generator.py first")
        return
    
    # Convert to DPO format
    formatter = DPODataFormatter()
    dpo_examples = formatter.format_dataset_for_dpo(preference_dataset)
    
    print(f"Conversion completed, generated {len(dpo_examples)} DPO training samples")
    
    # Save DPO dataset
    formatter.save_dpo_dataset(dpo_examples)
    formatter.save_for_huggingface_trl(dpo_examples)
    
    # Show sample examples
    if dpo_examples:
        print("\n=== DPO training sample examples ===")
        example = dpo_examples[0]
        print(f"Prompt: {example.prompt[:200]}...")
        print(f"\nChosen: {example.chosen}")
        print(f"\nRejected: {example.rejected}")

if __name__ == "__main__":
    main()
