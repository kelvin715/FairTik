#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from fairness_dataset_generator import FairnessDatasetGenerator
from dpo_data_formatter import DPODataFormatter

def regenerate_diverse_dataset():
    """Regenerate diverse dataset"""
    print("=== Regenerating Diverse Dataset ===")
    
    # Delete old dataset files
    old_files = [
        "/root/tiktok_techjam/policy_learning/dataset/fairness_preference_dataset.json",
        "/root/tiktok_techjam/policy_learning/dataset/dpo_dataset_hf.json",
        "/root/tiktok_techjam/policy_learning/dataset/dpo_training_dataset.json"
    ]
    
    for file_path in old_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted old file: {file_path}")
    
    # Create new dataset generator
    generator = FairnessDatasetGenerator()
    
    # Generate diverse scenarios
    print("\n=== Generating Diverse Scenarios ===")
    diverse_scenarios = generator.generate_diverse_scenarios(num_scenarios=1000)
    
    print(f"Generated {len(diverse_scenarios)} diverse scenarios")
    
    # Initialize dataset structure
    dataset = {
        "metadata": {
            "generated_at": generator.dataset["metadata"]["generated_at"],
            "num_scenarios": 0,
            "num_proposals": 0,
            "num_preference_pairs": 0,
            "fairness_principles": generator.fairness_principles,
            "diversity_factors": generator.dataset["metadata"]["diversity_factors"],
            "last_updated": generator.dataset["metadata"]["last_updated"]
        },
        "scenarios": [],
        "proposals": [],
        "preference_pairs": []
    }
    
    # Process scenarios in batches
    batch_size = 10  # Process 10 scenarios per batch
    total_scenarios = len(diverse_scenarios)
    total_batches = (total_scenarios + batch_size - 1) // batch_size
    
    print(f"\n=== Starting Batch Processing ===")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    
    start_time = time.time()
    
    for batch_start in range(0, total_scenarios, batch_size):
        batch_end = min(batch_start + batch_size, total_scenarios)
        batch_scenarios = diverse_scenarios[batch_start:batch_end]
        current_batch = batch_start // batch_size + 1
        
        print(f"\n=== Processing Batch {current_batch}/{total_batches} ===")
        print(f"Progress: {batch_start + 1} - {batch_end} / {total_scenarios} ({(batch_end/total_scenarios)*100:.1f}%)")
        
        batch_proposals = []
        batch_preference_pairs = []
        
        # Process scenarios in current batch
        for i, scenario in enumerate(batch_scenarios):
            scenario_id = f"scenario_{batch_start + i:03d}"
            scenario["scenario_id"] = scenario_id
            
            print(f"  Processing scenario {scenario_id}: {scenario['diversity_factors']['creator_type']} - {scenario['diversity_factors']['content_category']}")
            
            # Generate proposals
            proposals = generator.generate_revenue_split_proposals(scenario)
            batch_proposals.extend(proposals)
            
            # Create preference pairs
            if len(proposals) >= 2:
                pairs = generator.create_preference_pairs(proposals)
                batch_preference_pairs.extend(pairs)
                print(f"    Generated {len(pairs)} preference pairs")
        
        # Add current batch to dataset
        dataset["scenarios"].extend(batch_scenarios)
        dataset["proposals"].extend(batch_proposals)
        dataset["preference_pairs"].extend(batch_preference_pairs)
        
        # Update metadata
        dataset["metadata"]["num_scenarios"] = len(dataset["scenarios"])
        dataset["metadata"]["num_proposals"] = len(dataset["proposals"])
        dataset["metadata"]["num_preference_pairs"] = len(dataset["preference_pairs"])
        
        # Save current batch immediately
        print(f"\n  Saving batch {batch_start//batch_size + 1}...")
        preference_file = "/root/tiktok_techjam/policy_learning/dataset/fairness_preference_dataset.json"
        with open(preference_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        elapsed_time = time.time() - start_time
        avg_time_per_batch = elapsed_time / current_batch
        remaining_batches = total_batches - current_batch
        estimated_remaining_time = remaining_batches * avg_time_per_batch
        
        print(f"  ✓ Saved: {len(dataset['scenarios'])} scenarios, {len(dataset['proposals'])} proposals, {len(dataset['preference_pairs'])} preference pairs")
        print(f"  ⏱️  Time used: {elapsed_time/60:.1f} minutes, Estimated remaining: {estimated_remaining_time/60:.1f} minutes")
        
        # Convert to DPO format every 5 batches
        if current_batch % 5 == 0:
            print(f"\n  === Converting to DPO Format (Batch {current_batch}) ===")
            formatter = DPODataFormatter()
            dpo_examples = formatter.format_dataset_for_dpo(dataset)
            
            print(f"  Generated {len(dpo_examples)} DPO training samples")
            
            # Save DPO dataset
            formatter.save_dpo_dataset(dpo_examples)
            formatter.save_for_huggingface_trl(dpo_examples)
    
    total_time = time.time() - start_time
    print(f"\n=== Dataset Generation Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Number of scenarios: {len(dataset['scenarios'])}")
    print(f"Number of proposals: {len(dataset['proposals'])}")
    print(f"Number of preference pairs: {len(dataset['preference_pairs'])}")
    print(f"Average time per scenario: {total_time/len(dataset['scenarios']):.1f} seconds")
    
    # Final conversion to DPO format
    print("\n=== Final Conversion to DPO Format ===")
    formatter = DPODataFormatter()
    dpo_examples = formatter.format_dataset_for_dpo(dataset)
    
    print(f"Generated {len(dpo_examples)} DPO training samples")
    
    # Save DPO dataset
    formatter.save_dpo_dataset(dpo_examples)
    formatter.save_for_huggingface_trl(dpo_examples)
    
    # Analyze diversity
    print("\n=== Diversity Analysis ===")
    diversity_analysis = generator.analyze_diversity_distribution(dataset)
    
    print("Creator Type Distribution:")
    for creator_type, count in sorted(diversity_analysis['creator_types'].items()):
        percentage = (count / len(diverse_scenarios)) * 100
        print(f"  {creator_type}: {count} ({percentage:.1f}%)")
    
    print("\nContent Type Distribution:")
    for category, count in sorted(diversity_analysis['content_categories'].items()):
        percentage = (count / len(diverse_scenarios)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print("\nRegional Distribution:")
    for region, count in sorted(diversity_analysis['regions'].items()):
        percentage = (count / len(diverse_scenarios)) * 100
        print(f"  {region}: {count} ({percentage:.1f}%)")
    
    print("\nViewer Range Distribution:")
    for viewer_range, count in sorted(diversity_analysis['viewer_ranges'].items()):
        percentage = (count / len(diverse_scenarios)) * 100
        print(f"  {viewer_range}: {count} ({percentage:.1f}%)")
    
    # Verify no duplicate scenarios
    print("\n=== Duplicate Check ===")
    scenario_ids = [s["scenario_id"] for s in diverse_scenarios]
    unique_ids = set(scenario_ids)
    print(f"Total scenarios: {len(scenario_ids)}")
    print(f"Unique scenarios: {len(unique_ids)}")
    print(f"Duplicate scenarios: {len(scenario_ids) - len(unique_ids)}")
    
    if len(scenario_ids) == len(unique_ids):
        print("✓ No duplicate scenarios")
    else:
        print("⚠ Duplicate scenarios found")
    
    # Check preference pair diversity
    preference_scenario_ids = [p["scenario_id"] for p in dataset["preference_pairs"]]
    unique_preference_ids = set(preference_scenario_ids)
    print(f"\nPreference pair scenarios: {len(preference_scenario_ids)}")
    print(f"Unique preference pair scenarios: {len(unique_preference_ids)}")
    
    return dataset, dpo_examples

if __name__ == "__main__":
    regenerate_diverse_dataset()
