import openai
import json
import random
import math
import os
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

class FairnessDatasetGenerator:
    """Fairness preference dataset generator"""
    
    def __init__(self, save_dir: str = "/root/tiktok_techjam/policy_learning/dataset"):
        self.client = openai.OpenAI(api_key=API_KEY)
        self.save_dir = save_dir
        
        # Extended fairness principles
        self.fairness_principles = {
            "content_quality": {"weight": 0.25, "description": "Content quality priority: High-quality content should receive higher revenue share"},
            "traffic_contribution": {"weight": 0.2, "description": "Traffic contribution reward: High-exposure, high-engagement content should receive higher revenue share"},
            "newbie_support": {"weight": 0.15, "description": "New creator support: New creators (< 6 months) should receive preferential revenue share"},
            "niche_protection": {"weight": 0.15, "description": "Niche content protection: High-quality content with fewer viewers should be protected"},
            "loyalty_reward": {"weight": 0.1, "description": "Loyalty reward: Long-term stable creators should receive better treatment"},
            "geographic_fairness": {"weight": 0.05, "description": "Geographic fairness: Creators from different regions should receive fair treatment"},
            "content_diversity": {"weight": 0.05, "description": "Content diversity: Encourage diverse content creation"},
            "peak_performance": {"weight": 0.05, "description": "Peak performance reward: Content performing well during peak hours"}
        }
        
        # Add diversity configuration
        self.creator_types = ["Individual Creator", "MCN Agency", "Professional Team", "Student Creator", "Full-time Streamer", 
                             "Part-time Creator", "Celebrity Artist", "Internet Celebrity", "Amateur Streamer", "Corporate Account"]
        
        self.content_categories = ["Entertainment Comedy", "Knowledge Science", "Life Sharing", "Food Making", "Fashion Beauty", 
                                  "Gaming Live", "Music Performance", "Dance Show", "Travel Record", "Pet Cute",
                                  "Fitness Sports", "Tech Digital", "Education Learning", "Business Marketing", "News Information"]
        
        self.regions = ["Tier 1 City", "Tier 2 City", "Tier 3 City", "County", "Rural Area",
                       "Overseas Chinese", "Hong Kong Taiwan", "Ethnic Minority Area", "Economically Developed Area", "Underdeveloped Area"]
        
        self.peak_hours = ["Morning Peak (7-9 AM)", "Lunch Break (12-14 PM)", "Evening Peak (18-20 PM)", 
                          "Prime Time (20-22 PM)", "Late Night (22-24 PM)", "Early Morning (0-6 AM)"]
        
        self.creator_backgrounds = ["College Student", "Office Worker", "Freelancer", "Retiree", "Full-time Mom",
                                   "Professional Streamer", "Artist Celebrity", "Internet Celebrity", "Corporate Employee", "Entrepreneur"]
        
        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize dataset structure
        self.dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_scenarios": 0,
                "num_proposals": 0, 
                "num_preference_pairs": 0,
                "fairness_principles": self.fairness_principles,
                "diversity_factors": {
                    "creator_types": self.creator_types,
                    "content_categories": self.content_categories,
                    "regions": self.regions,
                    "peak_hours": self.peak_hours,
                    "creator_backgrounds": self.creator_backgrounds
                },
                "last_updated": datetime.now().isoformat()
            },
            "scenarios": [],
            "proposals": [],
            "preference_pairs": []
        }
        
        # Try to load existing dataset
        self._load_existing_dataset()

    def _load_existing_dataset(self):
        """Load existing dataset file"""
        dataset_file = os.path.join(self.save_dir, "fairness_preference_dataset.json")
        if os.path.exists(dataset_file):
            try:
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    self.dataset = json.load(f)
                logger.info(f"Loaded existing dataset: {self.dataset['metadata']['num_scenarios']} scenarios")
            except Exception as e:
                logger.warning(f"Failed to load existing dataset: {e}")
    
    def _save_incremental(self, batch_size: int = 10):
        """Incremental save dataset"""
        dataset_file = os.path.join(self.save_dir, "fairness_preference_dataset.json")
        
        # Update metadata
        self.dataset["metadata"]["num_scenarios"] = len(self.dataset["scenarios"])
        self.dataset["metadata"]["num_proposals"] = len(self.dataset["proposals"])
        self.dataset["metadata"]["num_preference_pairs"] = len(self.dataset["preference_pairs"])
        self.dataset["metadata"]["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"Incremental save completed: {len(self.dataset['scenarios'])} scenarios, {len(self.dataset['proposals'])} proposals, {len(self.dataset['preference_pairs'])} preference pairs")
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
    
    def _generate_single_scenario(self, scenario_id: str) -> Dict:
        """Generate single creator scenario"""
        
        # Randomly select diversity factors
        creator_type = random.choice(self.creator_types)
        content_category = random.choice(self.content_categories)
        region = random.choice(self.regions)
        peak_hour = random.choice(self.peak_hours)
        creator_background = random.choice(self.creator_backgrounds)
        
        # Adjust experience level based on creator type (using continuous distribution)
        if creator_type in ["Student Creator", "Amateur Streamer"]:
            experience_months = int(random.uniform(1, 12))  # 1-12 months
        elif creator_type in ["Celebrity Artist", "Internet Celebrity", "Professional Streamer"]:
            experience_months = int(random.uniform(12, 72))  # 12-72 months
        else:
            experience_months = int(random.uniform(1, 60))  # 1-60 months
        
        # Adjust follower count based on creator type and background (using continuous distribution)
        if creator_type in ["Celebrity Artist", "Internet Celebrity"]:
            # Use log-normal distribution to simulate follower count distribution
            follower_count = int(random.lognormvariate(12, 0.8))  # Approximately 50K-5M
        elif creator_type in ["MCN Agency", "Professional Team"]:
            follower_count = int(random.lognormvariate(10, 0.6))  # Approximately 10K-500K
        elif creator_type in ["Student Creator", "Amateur Streamer"]:
            follower_count = int(random.uniform(50, 5000))  # 50-5000
        else:
            # Individual creators use mixed distribution
            if random.random() < 0.7:  # 70% probability of small creators
                follower_count = int(random.uniform(100, 15000))
            else:  # 30% probability of large creators
                follower_count = int(random.uniform(15000, 500000))
        
        # Adjust content quality based on content type
        if content_category in ["Knowledge Science", "Education Learning", "Tech Digital"]:
            content_value = round(random.uniform(0.6, 0.95), 2)
            visual_quality = round(random.uniform(0.5, 0.9), 2)
        elif content_category in ["Entertainment Comedy", "Music Performance", "Dance Show"]:
            content_value = round(random.uniform(0.4, 0.9), 2)
            visual_quality = round(random.uniform(0.6, 0.95), 2)
        else:
            content_value = round(random.uniform(0.2, 0.95), 2)
            visual_quality = round(random.uniform(0.3, 0.9), 2)
        
        host_performance = round(random.uniform(0.4, 0.95), 2)
        
        # Adjust traffic based on region and time period
        if region in ["Tier 1 City", "Economically Developed Area"]:
            viewer_multiplier = random.uniform(1.2, 2.0)
        elif region in ["Rural Area", "Underdeveloped Area"]:
            viewer_multiplier = random.uniform(0.3, 0.8)
        else:
            viewer_multiplier = random.uniform(0.8, 1.5)
        
        if peak_hour in ["Prime Time (20-22 PM)", "Evening Peak (18-20 PM)"]:
            viewer_multiplier *= random.uniform(1.3, 1.8)
        elif peak_hour in ["Early Morning (0-6 AM)"]:
            viewer_multiplier *= random.uniform(0.4, 0.7)
        
        # Generate base traffic (using continuous distribution)
        # Determine base traffic range based on creator type and follower count
        if creator_type in ["Celebrity Artist", "Internet Celebrity"]:
            base_viewers = int(random.lognormvariate(10, 0.7))  # Approximately 10K-1M
        elif creator_type in ["MCN Agency", "Professional Team"]:
            base_viewers = int(random.lognormvariate(8, 0.6))  # Approximately 1K-100K
        elif creator_type in ["Student Creator", "Amateur Streamer"]:
            base_viewers = int(random.uniform(20, 1000))  # 20-1000
        else:
            # Generate traffic based on follower ratio
            follower_to_viewer_ratio = random.uniform(0.01, 0.5)  # 1%-50% of followers will watch
            base_viewers = int(follower_count * follower_to_viewer_ratio)
        
        # Apply regional and time period adjustments
        total_viewers = int(base_viewers * viewer_multiplier)
        
        # Adjust engagement rate based on content type
        if content_category in ["Gaming Live", "Music Performance"]:
            engagement_rate = round(random.uniform(0.3, 0.9), 2)
            retention_rate = round(random.uniform(0.4, 0.9), 2)
        elif content_category in ["Knowledge Science", "Education Learning"]:
            engagement_rate = round(random.uniform(0.2, 0.7), 2)
            retention_rate = round(random.uniform(0.3, 0.8), 2)
        else:
            engagement_rate = round(random.uniform(0.1, 0.9), 2)
            retention_rate = round(random.uniform(0.2, 0.85), 2)
        
        # Generate more interaction metrics
        chat_activity = round(random.uniform(0.05, 0.95), 2)
        share_rate = round(random.uniform(0.005, 0.25), 3)
        # Gift value based on viewer count and creator type
        if creator_type in ["Celebrity Artist", "Internet Celebrity"]:
            gift_value = int(total_viewers * random.uniform(0.5, 5.0))
        elif creator_type in ["Student Creator", "Amateur Streamer"]:
            gift_value = int(total_viewers * random.uniform(0.1, 1.0))
        else:
            gift_value = int(total_viewers * random.uniform(0.2, 3.0))
        
        # Adjust consistency score based on creator type
        if creator_type in ["Professional Streamer", "Celebrity Artist"]:
            consistency_score = round(random.uniform(0.7, 0.95), 2)
        elif creator_type in ["Student Creator", "Amateur Streamer"]:
            consistency_score = round(random.uniform(0.3, 0.8), 2)
        else:
            consistency_score = round(random.uniform(0.3, 0.95), 2)
        
        # Generate time-related metrics
        current_time = datetime.now()
        broadcast_date = current_time - timedelta(days=random.randint(0, 30))
        broadcast_hour = random.randint(0, 23)
        
        # Generate seasonal factors
        season = self._get_season(broadcast_date)
        seasonal_boost = self._get_seasonal_boost(season, content_category)
        
        return {
            "scenario_id": scenario_id,
            "diversity_factors": {
                "creator_type": creator_type,
                "content_category": content_category,
                "region": region,
                "peak_hour": peak_hour,
                "creator_background": creator_background,
                "season": season,
                "seasonal_boost": seasonal_boost
            },
            "ai_evaluation": {
                "content_value": content_value,
                "visual_quality": visual_quality,
                "host_performance": host_performance,
                "overall_rating": round((content_value + visual_quality + host_performance) / 3, 2),
                "content_originality": round(random.uniform(0.3, 0.95), 2),
                "production_quality": round(random.uniform(0.4, 0.9), 2)
            },
            "interaction_metrics": {
                "total_viewers": total_viewers,
                "engagement_rate": engagement_rate,
                "retention_rate": retention_rate,
                "chat_activity": chat_activity,
                "share_rate": share_rate,
                "gift_value": gift_value,
                "unique_viewers": int(total_viewers * random.uniform(0.7, 0.95)),
                "peak_concurrent_viewers": int(total_viewers * random.uniform(0.8, 1.2))
            },
            "creator_profile": {
                "experience_months": experience_months,
                "follower_count": follower_count,
                "consistency_score": consistency_score,
                "historical_avg_score": round(random.uniform(0.3, 0.85), 2),
                "total_broadcasts": int(random.uniform(1, 2000)), 
                "avg_session_duration": int(random.uniform(30, 300)), 
                "creator_level": self._get_creator_level(follower_count, experience_months)
            },
            "basic_metrics": {
                "duration_minutes": int(random.uniform(30, 240)), 
                "peak_viewers": total_viewers,
                "average_viewers": int(total_viewers * random.uniform(0.5, 0.95)),
                "total_comments": int(total_viewers * random.uniform(0.1, 5.0)),
                "total_likes": int(total_viewers * random.uniform(0.5, 15.0)),
                "broadcast_date": broadcast_date.strftime("%Y-%m-%d"),
                "broadcast_hour": broadcast_hour
            },
            "market_factors": {
                "competition_level": random.choice(["Low", "Medium", "High"]),
                "trending_topic": random.choice([True, False]),
                "platform_promotion": random.choice([True, False]),
                "cross_platform_presence": random.randint(0, 5)
            }
        }
    
    def _get_season(self, date: datetime) -> str:
        """Get season"""
        month = date.month
        if month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Autumn"
        else:
            return "Winter"
    
    def _get_seasonal_boost(self, season: str, content_category: str) -> float:
        """Get seasonal boost"""
        seasonal_boosts = {
            "Spring": {"Travel Record": 1.3, "Fashion Beauty": 1.2, "Fitness Sports": 1.1},
            "Summer": {"Food Making": 1.2, "Travel Record": 1.4, "Fitness Sports": 1.3},
            "Autumn": {"Food Making": 1.3, "Fashion Beauty": 1.2, "Education Learning": 1.1},
            "Winter": {"Food Making": 1.4, "Life Sharing": 1.2, "Entertainment Comedy": 1.1}
        }
        return seasonal_boosts.get(season, {}).get(content_category, 1.0)
    
    def _get_creator_level(self, follower_count: int, experience_months: int) -> str:
        """Determine creator level based on follower count and experience"""
        if follower_count >= 1000000:
            return "Super Influencer"
        elif follower_count >= 100000:
            return "Major Influencer"
        elif follower_count >= 10000:
            return "Medium Influencer"
        elif follower_count >= 1000:
            return "Small Influencer"
        else:
            return "New Creator"

    def generate_diverse_scenarios(self, num_scenarios: int = 100) -> List[Dict]:
        """Generate diverse creator scenarios"""
        scenarios = []
        
        # Ensure all types are represented
        type_distribution = {
            "Individual Creator": int(num_scenarios * 0.2),
            "MCN Agency": int(num_scenarios * 0.1),
            "Professional Team": int(num_scenarios * 0.1),
            "Student Creator": int(num_scenarios * 0.15),
            "Full-time Streamer": int(num_scenarios * 0.1),
            "Part-time Creator": int(num_scenarios * 0.1),
            "Celebrity Artist": int(num_scenarios * 0.05),
            "Internet Celebrity": int(num_scenarios * 0.1),
            "Amateur Streamer": int(num_scenarios * 0.05),
            "Corporate Account": int(num_scenarios * 0.05)
        }
        
        scenario_count = 0
        for creator_type, count in type_distribution.items():
            for _ in range(count):
                if scenario_count >= num_scenarios:
                    break
                    
                # Select appropriate parameters based on creator type
                scenario = self._generate_scenario_with_constraints(
                    f"scenario_{scenario_count:03d}", 
                    creator_type=creator_type
                )
                scenarios.append(scenario)
                scenario_count += 1
        
        # If not enough, randomly generate the remaining ones
        while len(scenarios) < num_scenarios:
            scenario = self._generate_single_scenario(f"scenario_{len(scenarios):03d}")
            scenarios.append(scenario)
            
        return scenarios
    
    def _generate_scenario_with_constraints(self, scenario_id: str, **constraints) -> Dict:
        """Generate scenario based on constraints"""
        # First generate base scenario
        scenario = self._generate_single_scenario(scenario_id)
        
        # Apply constraints
        if 'creator_type' in constraints:
            scenario['diversity_factors']['creator_type'] = constraints['creator_type']
            
            # Adjust related parameters based on creator type
            if constraints['creator_type'] == "Student Creator":
                scenario['creator_profile']['experience_months'] = int(random.uniform(1, 12))
                scenario['creator_profile']['follower_count'] = int(random.uniform(100, 5000))
                scenario['diversity_factors']['creator_background'] = "College Student"
            elif constraints['creator_type'] == "Celebrity Artist":
                scenario['creator_profile']['experience_months'] = int(random.uniform(24, 72))
                scenario['creator_profile']['follower_count'] = int(random.uniform(100000, 5000000))
                scenario['diversity_factors']['creator_background'] = "Artist Celebrity"
        
        return scenario
    
    def generate_extreme_scenarios(self, num_scenarios: int = 20) -> List[Dict]:
        """Generate extreme scenarios for testing boundary conditions"""
        extreme_scenarios = []
        
        # 1. Ultra-high traffic celebrity scenarios
        for i in range(num_scenarios // 4):
            scenario = self._generate_single_scenario(f"extreme_star_{i:02d}")
            scenario['diversity_factors']['creator_type'] = "Celebrity Artist"
            scenario['creator_profile']['follower_count'] = int(random.uniform(1000000, 10000000))
            scenario['interaction_metrics']['total_viewers'] = int(random.uniform(500000, 2000000))
            scenario['interaction_metrics']['engagement_rate'] = round(random.uniform(0.8, 0.95), 2)
            scenario['ai_evaluation']['content_value'] = round(random.uniform(0.8, 0.95), 2)
            extreme_scenarios.append(scenario)
        
        # 2. New creator small traffic scenarios
        for i in range(num_scenarios // 4):
            scenario = self._generate_single_scenario(f"extreme_newbie_{i:02d}")
            scenario['diversity_factors']['creator_type'] = "Student Creator"
            scenario['creator_profile']['experience_months'] = int(random.uniform(1, 6))
            scenario['creator_profile']['follower_count'] = int(random.uniform(30, 300))
            scenario['interaction_metrics']['total_viewers'] = int(random.uniform(15, 150))
            scenario['ai_evaluation']['content_value'] = round(random.uniform(0.6, 0.9), 2)  # High quality but low traffic
            extreme_scenarios.append(scenario)
        
        # 3. Niche high-quality scenarios
        for i in range(num_scenarios // 4):
            scenario = self._generate_single_scenario(f"extreme_niche_{i:02d}")
            scenario['diversity_factors']['content_category'] = random.choice(["Knowledge Science", "Tech Digital", "Education Learning"])
            scenario['diversity_factors']['region'] = random.choice(["Rural Area", "Underdeveloped Area"])
            scenario['interaction_metrics']['total_viewers'] = int(random.uniform(80, 800))
            scenario['ai_evaluation']['content_value'] = round(random.uniform(0.8, 0.95), 2)
            scenario['ai_evaluation']['content_originality'] = round(random.uniform(0.9, 0.95), 2)
            extreme_scenarios.append(scenario)
        
        # 4. High competition low quality scenarios
        for i in range(num_scenarios // 4):
            scenario = self._generate_single_scenario(f"extreme_competition_{i:02d}")
            scenario['diversity_factors']['content_category'] = "Entertainment Comedy"
            scenario['diversity_factors']['region'] = "Tier 1 City"
            scenario['market_factors']['competition_level'] = "High"
            scenario['interaction_metrics']['total_viewers'] = int(random.uniform(3000, 25000))
            scenario['ai_evaluation']['content_value'] = round(random.uniform(0.3, 0.6), 2)
            scenario['ai_evaluation']['content_originality'] = round(random.uniform(0.2, 0.5), 2)
            extreme_scenarios.append(scenario)
        
        return extreme_scenarios
    
    def _generate_realistic_distribution(self, distribution_type: str, **params) -> float:
        """Generate more realistic distribution data"""
        if distribution_type == "follower_count":
            # Follower count follows power law distribution (long tail)
            if params.get('creator_type') in ["Celebrity Artist", "Internet Celebrity"]:
                return int(random.lognormvariate(12, 0.8))
            elif params.get('creator_type') in ["MCN Agency", "Professional Team"]:
                return int(random.lognormvariate(10, 0.6))
            else:
                # Most creators have fewer followers, few have many
                if random.random() < 0.8:  # 80% are small creators
                    return int(random.uniform(50, 5000))
                else:  # 20% are large creators
                    return int(random.uniform(5000, 100000))
        
        elif distribution_type == "viewer_count":
            # Viewer count is related to follower count but has randomness
            follower_count = params.get('follower_count', 1000)
            viewer_ratio = random.uniform(0.01, 0.8)  # 1%-80% of followers watch
            return int(follower_count * viewer_ratio)
        
        elif distribution_type == "engagement_rate":
            # Engagement rate is related to content type and creator type
            base_rate = 0.3
            if params.get('content_category') in ["Gaming Live", "Music Performance"]:
                base_rate += random.uniform(0.1, 0.3)
            elif params.get('content_category') in ["Knowledge Science", "Education Learning"]:
                base_rate += random.uniform(-0.1, 0.1)
            
            if params.get('creator_type') in ["Celebrity Artist", "Internet Celebrity"]:
                base_rate += random.uniform(0.1, 0.2)
            
            return round(max(0.05, min(0.95, base_rate + random.uniform(-0.1, 0.1))), 2)
        
        elif distribution_type == "content_quality":
            # Content quality is related to creator experience
            experience_months = params.get('experience_months', 12)
            base_quality = min(0.9, 0.3 + (experience_months / 60) * 0.6)
            return round(base_quality + random.uniform(-0.2, 0.2), 2)
        
        else:
            return random.uniform(0, 1)
    
    def analyze_diversity_distribution(self, dataset: Dict) -> Dict:
        """Analyze diversity distribution of dataset"""
        analysis = {
            "creator_types": {},
            "content_categories": {},
            "regions": {},
            "creator_levels": {},
            "experience_ranges": {},
            "viewer_ranges": {},
            "quality_ranges": {},
            "seasons": {},
            "peak_hours": {}
        }
        
        for scenario in dataset['scenarios']:
            diversity = scenario.get('diversity_factors', {})
            creator_profile = scenario['creator_profile']
            interaction_metrics = scenario['interaction_metrics']
            ai_evaluation = scenario['ai_evaluation']
            
            # Creator type distribution
            creator_type = diversity.get('creator_type', 'Unknown')
            analysis['creator_types'][creator_type] = analysis['creator_types'].get(creator_type, 0) + 1
            
            # Content type distribution
            content_category = diversity.get('content_category', 'Unknown')
            analysis['content_categories'][content_category] = analysis['content_categories'].get(content_category, 0) + 1
            
            # Regional distribution
            region = diversity.get('region', 'Unknown')
            analysis['regions'][region] = analysis['regions'].get(region, 0) + 1
            
            # Creator level distribution
            creator_level = creator_profile.get('creator_level', 'Unknown')
            analysis['creator_levels'][creator_level] = analysis['creator_levels'].get(creator_level, 0) + 1
            
            # Experience range distribution
            experience_months = creator_profile.get('experience_months', 0)
            if experience_months <= 6:
                exp_range = "New Creator (≤6 months)"
            elif experience_months <= 24:
                exp_range = "Growing (7-24 months)"
            else:
                exp_range = "Mature (>24 months)"
            analysis['experience_ranges'][exp_range] = analysis['experience_ranges'].get(exp_range, 0) + 1
            
            # Viewer count range distribution
            viewers = interaction_metrics.get('total_viewers', 0)
            if viewers <= 1000:
                viewer_range = "Small Traffic (≤1K)"
            elif viewers <= 10000:
                viewer_range = "Medium Traffic (1K-10K)"
            elif viewers <= 100000:
                viewer_range = "Large Traffic (10K-100K)"
            else:
                viewer_range = "Ultra Large Traffic (>100K)"
            analysis['viewer_ranges'][viewer_range] = analysis['viewer_ranges'].get(viewer_range, 0) + 1
            
            # Content quality range distribution
            content_value = ai_evaluation.get('content_value', 0)
            if content_value <= 0.5:
                quality_range = "Low Quality (≤0.5)"
            elif content_value <= 0.8:
                quality_range = "Medium Quality (0.5-0.8)"
            else:
                quality_range = "High Quality (>0.8)"
            analysis['quality_ranges'][quality_range] = analysis['quality_ranges'].get(quality_range, 0) + 1
            
            # Season distribution
            season = diversity.get('season', 'Unknown')
            analysis['seasons'][season] = analysis['seasons'].get(season, 0) + 1
            
            # Time period distribution
            peak_hour = diversity.get('peak_hour', 'Unknown')
            analysis['peak_hours'][peak_hour] = analysis['peak_hours'].get(peak_hour, 0) + 1
        
        return analysis

    def generate_revenue_split_proposals(self, scenario: Dict) -> List[Dict]:
        """Generate multiple revenue share proposals for given scenario"""
        
        # Extract diversity factors
        diversity = scenario.get('diversity_factors', {})
        creator_type = diversity.get('creator_type', 'Individual Creator')
        content_category = diversity.get('content_category', 'Life Sharing')
        region = diversity.get('region', 'Tier 2 City')
        peak_hour = diversity.get('peak_hour', 'Prime Time (20-22 PM)')
        creator_background = diversity.get('creator_background', 'Office Worker')
        season = diversity.get('season', 'Spring')
        seasonal_boost = diversity.get('seasonal_boost', 1.0)
        
        # Extract market factors
        market = scenario.get('market_factors', {})
        competition_level = market.get('competition_level', 'Medium')
        trending_topic = market.get('trending_topic', False)
        platform_promotion = market.get('platform_promotion', False)
        
        prompt = f"""
Based on the following TikTok creator scenario, please generate 3 different revenue share proposals.

=== Creator Profile ===
- Creator Type: {creator_type}
- Creator Background: {creator_background}
- Experience: {scenario['creator_profile']['experience_months']} months
- Follower Count: {scenario['creator_profile']['follower_count']}
- Creator Level: {scenario['creator_profile']['creator_level']}
- Consistency Score: {scenario['creator_profile']['consistency_score']}
- Historical Average Score: {scenario['creator_profile']['historical_avg_score']}
- Total Broadcasts: {scenario['creator_profile']['total_broadcasts']}

=== Content Performance ===
- Content Category: {content_category}
- Content Value: {scenario['ai_evaluation']['content_value']}
- Visual Quality: {scenario['ai_evaluation']['visual_quality']}
- Host Performance: {scenario['ai_evaluation']['host_performance']}
- Content Originality: {scenario['ai_evaluation']['content_originality']}
- Production Quality: {scenario['ai_evaluation']['production_quality']}
- Overall Rating: {scenario['ai_evaluation']['overall_rating']}

=== Traffic Performance ===
- Total Viewers: {scenario['interaction_metrics']['total_viewers']}
- Engagement Rate: {scenario['interaction_metrics']['engagement_rate']}
- Retention Rate: {scenario['interaction_metrics']['retention_rate']}
- Share Rate: {scenario['interaction_metrics']['share_rate']}
- Gift Value: {scenario['interaction_metrics']['gift_value']}
- Total Comments: {scenario['basic_metrics']['total_comments']}
- Total Likes: {scenario['basic_metrics']['total_likes']}

=== Environmental Factors ===
- Region: {region}
- Broadcast Time: {peak_hour}
- Season: {season} (Seasonal Boost: {seasonal_boost})
- Competition Level: {competition_level}
- Trending Topic: {'Yes' if trending_topic else 'No'}
- Platform Promotion: {'Yes' if platform_promotion else 'No'}

=== Fairness Principles (Consider Comprehensively) ===
1. Content Quality Priority (25% weight): High-quality content should receive higher revenue share
2. Traffic Contribution Reward (20% weight): High-exposure, high-engagement content should receive higher revenue share
3. Newbie Support (15% weight): New creators (<6 months) should receive preferential policies
4. Niche Protection (15% weight): High-quality content with fewer viewers needs protection
5. Loyalty Reward (10% weight): Long-term stable creators should receive better treatment
6. Geographic Fairness (5% weight): Creators from different regions should receive fair treatment
7. Content Diversity (5% weight): Encourage diverse content creation
8. Peak Performance Reward (5% weight): Content performing well during peak hours

Please generate 3 revenue share proposals for this creator, each proposal including:
1. Creator share percentage (0-100%)
2. Proposal reasoning (detailed explanation of considered factors)

Output format (strict JSON, ensure all strings are wrapped with double quotes):
{{
    "proposals": [
        {{
            "creator_share_percentage": number,
            "reasoning": "string"
        }},
        {{
            "creator_share_percentage": number,
            "reasoning": "string"
        }},
        {{
            "creator_share_percentage": number,
            "reasoning": "string"
        }}
    ]
}}

Note: Please ensure the JSON format is completely correct, all strings are wrapped with double quotes, do not use single quotes.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0
            )
            
            result = self._parse_json_response(response.choices[0].message.content)
            
            # Add scenario information to each proposal
            for proposal in result.get("proposals", []):
                proposal["scenario_id"] = scenario["scenario_id"]
                proposal["scenario_data"] = scenario
            
            return result.get("proposals", [])
            
        except Exception as e:
            logger.error(f"Failed to generate proposals: {e}")
            return self._get_default_proposals(scenario)
    
    def create_preference_pairs(self, proposals: List[Dict]) -> List[Dict]:
        """Create preference comparison data"""
        preference_pairs = []
        
        # Only select the two proposals with the largest difference for comparison, avoid repetition
        if len(proposals) >= 2:
            # Sort by revenue share percentage
            sorted_proposals = sorted(proposals, key=lambda x: x['creator_share_percentage'])
            
            # Select the two proposals with the largest difference (highest and lowest)
            proposal_a = sorted_proposals[0]  # Lowest revenue share
            proposal_b = sorted_proposals[-1]  # Highest revenue share
            
            # Let GPT judge which is more fair
            comparison_result = self._compare_fairness(proposal_a, proposal_b)
            
            # Safety check: ensure comparison_result contains necessary keys
            if (comparison_result and 
                "preference" in comparison_result and 
                comparison_result["preference"] != "unclear"):
                
                preference_pairs.append({
                    "scenario_id": proposal_a["scenario_id"],
                    "proposal_a": proposal_a,
                    "proposal_b": proposal_b,
                    "preferred": comparison_result.get("preference", "unclear"), # "a" or "b"
                    "reasoning": comparison_result.get("reasoning", "Comparison failed"),
                    "confidence": comparison_result.get("confidence", 0.0)
                })
        
        return preference_pairs
    
    def _compare_fairness(self, proposal_a: Dict, proposal_b: Dict) -> Dict:
        """Let GPT compare the fairness of two proposals"""
        
        scenario = proposal_a["scenario_data"]
        diversity = scenario.get('diversity_factors', {})
        market = scenario.get('market_factors', {})
        
        prompt = f"""
As a fairness expert, please compare the fairness of the following two TikTok revenue share proposals.

=== Creator Background ===
- Creator Type: {diversity.get('creator_type', 'Individual Creator')}
- Creator Background: {diversity.get('creator_background', 'Office Worker')}
- Experience: {scenario['creator_profile']['experience_months']} months
- Follower Count: {scenario['creator_profile']['follower_count']}
- Creator Level: {scenario['creator_profile']['creator_level']}
- Content Quality: {scenario['ai_evaluation']['content_value']}
- Content Category: {diversity.get('content_category', 'Life Sharing')}

=== Traffic Performance ===
- Total Viewers: {scenario['interaction_metrics']['total_viewers']}
- Engagement Rate: {scenario['interaction_metrics']['engagement_rate']}
- Retention Rate: {scenario['interaction_metrics']['retention_rate']}
- Share Rate: {scenario['interaction_metrics']['share_rate']}
- Gift Value: {scenario['interaction_metrics']['gift_value']}
- Total Comments: {scenario['basic_metrics']['total_comments']}
- Total Likes: {scenario['basic_metrics']['total_likes']}

=== Environmental Factors ===
- Region: {diversity.get('region', 'Tier 2 City')}
- Broadcast Time: {diversity.get('peak_hour', 'Prime Time (20-22 PM)')}
- Season: {diversity.get('season', 'Spring')}
- Competition Level: {market.get('competition_level', 'Medium')}
- Trending Topic: {'Yes' if market.get('trending_topic', False) else 'No'}
- Platform Promotion: {'Yes' if market.get('platform_promotion', False) else 'No'}

Proposal A: Creator receives {proposal_a['creator_share_percentage']}%
Reasoning: {proposal_a['reasoning']}

Proposal B: Creator receives {proposal_b['creator_share_percentage']}%  
Reasoning: {proposal_b['reasoning']}

=== Fairness Assessment Principles ===
1. Content Quality Priority (25% weight): High-quality content should receive higher revenue share
2. Traffic Contribution Reward (20% weight): High-exposure, high-engagement content should receive higher revenue share
3. Newbie Support (15% weight): New creators (<6 months) should receive preferential policies
4. Niche Protection (15% weight): High-quality content with fewer viewers needs protection
5. Loyalty Reward (10% weight): Long-term stable creators should receive better treatment
6. Geographic Fairness (5% weight): Creators from different regions should receive fair treatment
7. Content Diversity (5% weight): Encourage diverse content creation
8. Peak Performance Reward (5% weight): Content performing well during peak hours

Please determine which proposal is more fair and provide detailed reasoning.

Output format (strict JSON, ensure all strings are wrapped with double quotes):
{{
    "reasoning": "string",
    "fairness_analysis": {{
        "content_quality_factor": "string",
        "traffic_contribution_factor": "string",
        "newbie_support_factor": "string",
        "niche_protection_factor": "string",
        "loyalty_reward_factor": "string",
        "geographic_fairness_factor": "string",
        "content_diversity_factor": "string",
        "peak_performance_factor": "string"
    }},
    "confidence": number,
    "preference": "a" or "b" or "unclear"
}}

Note: Please ensure the JSON format is completely correct, all strings are wrapped with double quotes, do not use single quotes.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # Lower temperature ensures consistency
            )
            
            result = self._parse_json_response(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Failed to compare fairness: {e}")
            return {
                "preference": "unclear",
                "reasoning": "Comparison failed",
                "confidence": 0.0
            }
    
    def generate_full_dataset(self, num_scenarios: int = 50, save_batch_size: int = 10) -> Dict:
        """Generate complete preference dataset, save while generating"""
        logger.info(f"Starting to generate preference dataset with {num_scenarios} scenarios...")
        
        # Get number of existing scenarios
        existing_scenarios = len(self.dataset["scenarios"])
        remaining_scenarios = num_scenarios - existing_scenarios
        
        if remaining_scenarios <= 0:
            logger.info(f"Already have {existing_scenarios} scenarios, no need to regenerate")
            return self.dataset
        
        logger.info(f"Need to generate {remaining_scenarios} new scenarios")
        
        # Use diverse scenario generation
        diverse_scenarios = self.generate_diverse_scenarios(remaining_scenarios)
        
        # Generate scenarios in batches
        for batch_start in range(0, len(diverse_scenarios), save_batch_size):
            batch_end = min(batch_start + save_batch_size, len(diverse_scenarios))
            batch_scenarios = diverse_scenarios[batch_start:batch_end]
            
            logger.info(f"Generating batch {batch_start//save_batch_size + 1}: {len(batch_scenarios)} scenarios")
            
            # Generate scenarios for current batch
            for i, scenario in enumerate(batch_scenarios):
                scenario_id = f"scenario_{(existing_scenarios + batch_start + i):03d}"
                scenario["scenario_id"] = scenario_id
                self.dataset["scenarios"].append(scenario)
                
                # Generate proposals for current scenario
                proposals = self.generate_revenue_split_proposals(scenario)
                self.dataset["proposals"].extend(proposals)
                
                # Create preference pairs
                if len(proposals) >= 2:
                    pairs = self.create_preference_pairs(proposals)
                    self.dataset["preference_pairs"].extend(pairs)
                    logger.info(f"Scenario {scenario_id} generated {len(pairs)} preference pairs")
            
            # Save current batch
            self._save_incremental(save_batch_size)
        
        logger.info(f"Dataset generation completed: {len(self.dataset['scenarios'])} scenarios, {len(self.dataset['proposals'])} proposals, {len(self.dataset['preference_pairs'])} preference pairs")
        return self.dataset
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON response"""
        try:
            # Try to find JSON start and end positions
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                
                # Try to fix common JSON format issues
                json_str = self._fix_json_format(json_str)
                
                # Try to parse JSON
                result = json.loads(json_str)
                return result
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.warning(f"Response text: {response_text[:200]}...")  # Log first 200 characters
            return {}
    
    def _fix_json_format(self, json_str: str) -> str:
        """Fix common JSON format issues"""
        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')
        
        # Remove newlines and carriage returns
        json_str = json_str.replace('\n', ' ')
        json_str = json_str.replace('\r', ' ')
        
        # Fix possible trailing commas
        json_str = json_str.replace(',}', '}')
        json_str = json_str.replace(',]', ']')
        
        # Fix possible missing quotes
        import re
        # Fix property names missing quotes
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        return json_str
    
    def _get_default_proposals(self, scenario: Dict) -> List[Dict]:
        """Get default proposals"""
        return [
            {
                "creator_share_percentage": 70,
                "reasoning": "Default proposal",
                "scenario_id": scenario["scenario_id"],
                "scenario_data": scenario
            }
        ]

def save_dataset(dataset: Dict, filename: str = "/root/tiktok_techjam/policy_learning/dataset/fairness_preference_dataset.json"):
    """Save dataset to file"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"Dataset saved to {filename}")

def main():
    """Main function: Generate preference dataset"""
    generator = FairnessDatasetGenerator()
    
    # Generate small-scale test dataset, save every 3 scenarios
    dataset = generator.generate_full_dataset(num_scenarios=100, save_batch_size=3)
    
    # Generate extreme scenarios and add to dataset
    print("\n=== Generating Extreme Scenarios ===")
    extreme_scenarios = generator.generate_extreme_scenarios(num_scenarios=20)
    
    for scenario in extreme_scenarios:
        dataset['scenarios'].append(scenario)
        
        # Generate proposals for extreme scenarios
        proposals = generator.generate_revenue_split_proposals(scenario)
        dataset['proposals'].extend(proposals)
        
        # Create preference pairs
        if len(proposals) >= 2:
            pairs = generator.create_preference_pairs(proposals)
            dataset['preference_pairs'].extend(pairs)
            print(f"Extreme scenario {scenario['scenario_id']} generated {len(pairs)} preference pairs")
    
    # Save complete dataset including extreme scenarios
    generator._save_incremental(10)
    
    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    print(f"Number of scenarios: {dataset['metadata']['num_scenarios']}")
    print(f"Number of proposals: {dataset['metadata']['num_proposals']}")
    print(f"Number of preference pairs: {dataset['metadata']['num_preference_pairs']}")
    print(f"Last updated: {dataset['metadata']['last_updated']}")
    
    # Show diversity statistics
    print("\n=== Diversity Statistics ===")
    if dataset['scenarios']:
        # Use diversity analysis functionality
        diversity_analysis = generator.analyze_diversity_distribution(dataset)
        
        print("Creator Type Distribution:")
        for creator_type, count in sorted(diversity_analysis['creator_types'].items()):
            percentage = (count / len(dataset['scenarios'])) * 100
            print(f"  {creator_type}: {count} ({percentage:.1f}%)")
        
        print("\nContent Type Distribution:")
        for category, count in sorted(diversity_analysis['content_categories'].items()):
            percentage = (count / len(dataset['scenarios'])) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print("\nRegional Distribution:")
        for region, count in sorted(diversity_analysis['regions'].items()):
            percentage = (count / len(dataset['scenarios'])) * 100
            print(f"  {region}: {count} ({percentage:.1f}%)")
        
        print("\nCreator Level Distribution:")
        for level, count in sorted(diversity_analysis['creator_levels'].items()):
            percentage = (count / len(dataset['scenarios'])) * 100
            print(f"  {level}: {count} ({percentage:.1f}%)")
        
        print("\nExperience Range Distribution:")
        for exp_range, count in sorted(diversity_analysis['experience_ranges'].items()):
            percentage = (count / len(dataset['scenarios'])) * 100
            print(f"  {exp_range}: {count} ({percentage:.1f}%)")
        
        print("\nViewer Range Distribution:")
        for viewer_range, count in sorted(diversity_analysis['viewer_ranges'].items()):
            percentage = (count / len(dataset['scenarios'])) * 100
            print(f"  {viewer_range}: {count} ({percentage:.1f}%)")
        
        print("\nContent Quality Distribution:")
        for quality_range, count in sorted(diversity_analysis['quality_ranges'].items()):
            percentage = (count / len(dataset['scenarios'])) * 100
            print(f"  {quality_range}: {count} ({percentage:.1f}%)")
        
        print("\nSeason Distribution:")
        for season, count in sorted(diversity_analysis['seasons'].items()):
            percentage = (count / len(dataset['scenarios'])) * 100
            print(f"  {season}: {count} ({percentage:.1f}%)")
        
        print("\nTime Period Distribution:")
        for peak_hour, count in sorted(diversity_analysis['peak_hours'].items()):
            percentage = (count / len(dataset['scenarios'])) * 100
            print(f"  {peak_hour}: {count} ({percentage:.1f}%)")
    
    # Show a detailed preference pair example
    if dataset['preference_pairs']:
        print("\n=== Detailed Preference Pair Example ===")
        example_pair = dataset['preference_pairs'][0]
        scenario = example_pair['proposal_a']['scenario_data']
        diversity = scenario.get('diversity_factors', {})
        
        print(f"Scenario ID: {example_pair['scenario_id']}")
        print(f"Creator Type: {diversity.get('creator_type', 'Unknown')}")
        print(f"Content Type: {diversity.get('content_category', 'Unknown')}")
        print(f"Region: {diversity.get('region', 'Unknown')}")
        print(f"Creator Level: {scenario['creator_profile'].get('creator_level', 'Unknown')}")
        print(f"Viewer Count: {scenario['interaction_metrics']['total_viewers']}")
        print(f"Proposal A: {example_pair['proposal_a']['creator_share_percentage']}%")
        print(f"Proposal B: {example_pair['proposal_b']['creator_share_percentage']}%")
        print(f"Preference: {example_pair['preferred']}")
        print(f"Reasoning: {example_pair['reasoning']}")
    
    # Show dataset metadata
    print("\n=== Dataset Metadata ===")
    print(f"Number of fairness principles: {len(dataset['metadata']['fairness_principles'])}")
    print(f"Diversity factors: {list(dataset['metadata']['diversity_factors'].keys())}")
    
    # Show fairness principles
    print("\nFairness Principles:")
    for principle, details in dataset['metadata']['fairness_principles'].items():
        print(f"  {principle}: {details['weight']*100}% - {details['description']}")
    
    # Show data distribution characteristics
    print("\n=== Data Distribution Characteristics ===")
    print("✓ Use continuous distributions instead of discrete values")
    print("✓ Follower count uses log-normal distribution to simulate real distribution")
    print("✓ Viewer count dynamically generated based on follower ratio")
    print("✓ Interaction metrics related to content type and creator type")
    print("✓ Content quality related to creator experience")
    print("✓ Regional and time factors affect traffic performance")
    print("✓ Seasonal factors affect content performance")
    print("✓ Market factors affect competitive environment")

if __name__ == "__main__":
    main()
