#!/usr/bin/env python3
"""
Explore available literary/dramatic datasets for emotion-rich text.
"""

import sys
from datasets import load_dataset, list_datasets
import pandas as pd

def explore_dataset(dataset_name, config=None, split='train', max_examples=5):
    """Explore a dataset and show examples."""
    try:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        if config:
            print(f"Config: {config}")
        print(f"{'='*80}\n")
        
        # Load dataset
        if config:
            dataset = load_dataset(dataset_name, config, split=split, streaming=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        # Show examples
        print(f"First {max_examples} examples:\n")
        for i, example in enumerate(dataset):
            if i >= max_examples:
                break
            print(f"Example {i+1}:")
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 200:
                    print(f"  {key}: {value[:200]}...")
                else:
                    print(f"  {key}: {value}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("              探索文艺/戏剧数据集")
    print("="*80)
    
    # Datasets to explore
    datasets_to_try = [
        # Story datasets
        ("roneneldan/TinyStories", None),
        ("WritingPrompts/writingprompts", None),
        
        # Dialogue datasets  
        ("daily_dialog", None),
        
        # Literary texts
        ("bigscience/P3", "story_cloze_2016_get_story_ending"),
    ]
    
    available_datasets = []
    
    for dataset_name, config in datasets_to_try:
        print(f"\n{'='*80}")
        print(f"Testing: {dataset_name}")
        print(f"{'='*80}")
        
        success = explore_dataset(dataset_name, config, max_examples=3)
        
        if success:
            available_datasets.append((dataset_name, config))
            print(f"✅ {dataset_name} is available")
        else:
            print(f"❌ {dataset_name} failed")
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"\nAvailable datasets ({len(available_datasets)}):")
    for name, config in available_datasets:
        print(f"  • {name}" + (f" ({config})" if config else ""))
    print()

if __name__ == '__main__':
    main()
