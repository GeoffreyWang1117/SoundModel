#!/usr/bin/env python3
"""
Add 'text' field to RAVDESS metadata for training compatibility.
"""

import json
from pathlib import Path

def fix_metadata(split_dir: Path):
    """Add 'text' field to metadata.json."""
    metadata_file = split_dir / "metadata.json"

    with open(metadata_file, 'r') as f:
        samples = json.load(f)

    # Add 'text' field to each sample
    for sample in samples:
        # Use the statement as the text
        sample['text'] = sample['statement']

    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"✅ Fixed {split_dir.name}: {len(samples)} samples")


def main():
    ravdess_dir = Path("./audio_augmented_llm/data/ravdess_processed")

    for split_name in ["train", "val", "test"]:
        split_dir = ravdess_dir / split_name
        if split_dir.exists():
            fix_metadata(split_dir)

    print("\n✅ All metadata files updated!")


if __name__ == "__main__":
    main()
