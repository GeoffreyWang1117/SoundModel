#!/usr/bin/env python3
"""
Analyze WavLM embeddings to quantify speaker information content.

This script performs several analyses:
1. Speaker classification accuracy (how much speaker info is encoded?)
2. Emotion classification accuracy (how much emotion info is encoded?)
3. Within-speaker vs between-speaker distance analysis
4. Visualization with t-SNE/UMAP
5. Emotion vs speaker separability comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_dataset_with_embeddings(data_dir: str, embedding_type: str = "wavlm"):
    """
    Load embeddings and metadata from a dataset directory.

    Returns:
        embeddings: numpy array (N, D)
        speakers: list of speaker names
        emotions: list of emotion labels
        sample_ids: list of sample IDs
    """
    data_path = Path(data_dir)

    # Load metadata
    metadata_path = data_path / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load embeddings
    if embedding_type == "wavlm":
        # Try both possible names
        embeddings_path = data_path / "wavlm_embeddings.npz"
        if not embeddings_path.exists():
            embeddings_path = data_path / "emotion_embeddings.npz"
    elif embedding_type == "acoustic":
        embeddings_path = data_path / "acoustic_embeddings.npz"
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    embeddings_dict = np.load(embeddings_path)

    # Extract data in order
    embeddings = []
    speakers = []
    emotions = []
    sample_ids = []

    for item in metadata:
        sample_id = item['id']
        speaker = item['speaker']
        emotion = item['emotion']

        if sample_id in embeddings_dict:
            embeddings.append(embeddings_dict[sample_id])
            speakers.append(speaker)
            emotions.append(emotion)
            sample_ids.append(sample_id)

    return np.array(embeddings), speakers, emotions, sample_ids

def train_speaker_classifier(X_train, y_train, X_test, y_test):
    """Train a logistic regression classifier to predict speaker from embeddings."""
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return clf, accuracy, y_pred

def train_emotion_classifier(X_train, y_train, X_test, y_test):
    """Train a logistic regression classifier to predict emotion from embeddings."""
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return clf, accuracy, y_pred

def compute_distance_statistics(embeddings, speakers, emotions):
    """
    Compute within-speaker vs between-speaker distances,
    and within-emotion vs between-emotion distances.
    """
    # Speaker-based distances
    within_speaker_dists = []
    between_speaker_dists = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            if speakers[i] == speakers[j]:
                within_speaker_dists.append(dist)
            else:
                between_speaker_dists.append(dist)

    # Emotion-based distances
    within_emotion_dists = []
    between_emotion_dists = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            if emotions[i] == emotions[j]:
                within_emotion_dists.append(dist)
            else:
                between_emotion_dists.append(dist)

    return {
        'within_speaker': np.array(within_speaker_dists),
        'between_speaker': np.array(between_speaker_dists),
        'within_emotion': np.array(within_emotion_dists),
        'between_emotion': np.array(between_emotion_dists)
    }

def visualize_embeddings_tsne(embeddings, speakers, emotions, output_dir: Path):
    """Create t-SNE visualizations colored by speaker and emotion."""
    print("\n🔬 Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Color by speaker
    unique_speakers = sorted(list(set(speakers)))
    speaker_colors = {s: i for i, s in enumerate(unique_speakers)}
    colors_speaker = [speaker_colors[s] for s in speakers]

    scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=colors_speaker, cmap='tab10', alpha=0.6, s=30)
    ax1.set_title('WavLM Embeddings (colored by Speaker)', fontsize=14)
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')

    # Add legend for speakers
    handles1 = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=plt.cm.tab10(speaker_colors[s] / 10),
                          markersize=8, label=s) for s in unique_speakers]
    ax1.legend(handles=handles1, loc='best', fontsize=9)

    # Color by emotion
    unique_emotions = sorted(list(set(emotions)))
    emotion_colors = {e: i for i, e in enumerate(unique_emotions)}
    colors_emotion = [emotion_colors[e] for e in emotions]

    scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=colors_emotion, cmap='viridis', alpha=0.6, s=30)
    ax2.set_title('WavLM Embeddings (colored by Emotion)', fontsize=14)
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')

    # Add legend for emotions
    handles2 = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=plt.cm.viridis(emotion_colors[e] / len(unique_emotions)),
                          markersize=8, label=e) for e in unique_emotions]
    ax2.legend(handles=handles2, loc='best', fontsize=9)

    plt.tight_layout()

    output_path = output_dir / 'wavlm_tsne_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved t-SNE visualization to: {output_path}")
    plt.close()

def plot_distance_distributions(distance_stats, output_dir: Path):
    """Plot distributions of within vs between distances."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Speaker distances
    ax1.hist(distance_stats['within_speaker'], bins=50, alpha=0.6, label='Within Speaker', color='blue')
    ax1.hist(distance_stats['between_speaker'], bins=50, alpha=0.6, label='Between Speaker', color='red')
    ax1.set_xlabel('Euclidean Distance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Speaker-based Distance Distribution')
    ax1.legend()
    ax1.axvline(distance_stats['within_speaker'].mean(), color='blue', linestyle='--',
                label=f"Within mean: {distance_stats['within_speaker'].mean():.2f}")
    ax1.axvline(distance_stats['between_speaker'].mean(), color='red', linestyle='--',
                label=f"Between mean: {distance_stats['between_speaker'].mean():.2f}")

    # Emotion distances
    ax2.hist(distance_stats['within_emotion'], bins=50, alpha=0.6, label='Within Emotion', color='green')
    ax2.hist(distance_stats['between_emotion'], bins=50, alpha=0.6, label='Between Emotion', color='orange')
    ax2.set_xlabel('Euclidean Distance')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Emotion-based Distance Distribution')
    ax2.legend()
    ax2.axvline(distance_stats['within_emotion'].mean(), color='green', linestyle='--',
                label=f"Within mean: {distance_stats['within_emotion'].mean():.2f}")
    ax2.axvline(distance_stats['between_emotion'].mean(), color='orange', linestyle='--',
                label=f"Between mean: {distance_stats['between_emotion'].mean():.2f}")

    plt.tight_layout()

    output_path = output_dir / 'distance_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved distance distributions to: {output_path}")
    plt.close()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze WavLM embeddings for speaker information")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory with embeddings")
    parser.add_argument("--test_dir", type=str, default=None,
                       help="Optional test directory (e.g., cross-speaker test)")
    parser.add_argument("--embedding_type", type=str, default="wavlm",
                       choices=["wavlm", "acoustic"], help="Embedding type to analyze")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for results")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("WavLM Embedding Analysis")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Embedding type: {args.embedding_type}")
    print(f"Output directory: {args.output_dir}")

    # Load training data
    print("\n📂 Loading embeddings...")
    embeddings, speakers, emotions, sample_ids = load_dataset_with_embeddings(
        args.data_dir, args.embedding_type
    )

    print(f"✓ Loaded {len(embeddings)} samples")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Unique speakers: {len(set(speakers))} - {sorted(set(speakers))}")
    print(f"  Unique emotions: {len(set(emotions))} - {sorted(set(emotions))}")

    # Split data for classification
    X_train, X_test, y_speaker_train, y_speaker_test, y_emotion_train, y_emotion_test = \
        train_test_split(embeddings, speakers, emotions, test_size=0.2, random_state=42, stratify=speakers)

    # 1. Speaker Classification
    print("\n" + "=" * 80)
    print("1. SPEAKER CLASSIFICATION")
    print("=" * 80)
    print("Training logistic regression classifier to predict speaker from embeddings...")

    speaker_clf, speaker_acc, speaker_pred = train_speaker_classifier(
        X_train, y_speaker_train, X_test, y_speaker_test
    )

    print(f"\n✓ Speaker Classification Accuracy: {speaker_acc * 100:.2f}%")
    print(f"  Baseline (random): {100.0 / len(set(speakers)):.2f}%")
    print(f"  Information gain: {(speaker_acc - 1.0/len(set(speakers))) * 100:.2f}%")

    print("\nClassification Report (Speaker):")
    print(classification_report(y_speaker_test, speaker_pred))

    # 2. Emotion Classification
    print("\n" + "=" * 80)
    print("2. EMOTION CLASSIFICATION")
    print("=" * 80)
    print("Training logistic regression classifier to predict emotion from embeddings...")

    emotion_clf, emotion_acc, emotion_pred = train_emotion_classifier(
        X_train, y_emotion_train, X_test, y_emotion_test
    )

    print(f"\n✓ Emotion Classification Accuracy: {emotion_acc * 100:.2f}%")
    print(f"  Baseline (random): {100.0 / len(set(emotions)):.2f}%")
    print(f"  Information gain: {(emotion_acc - 1.0/len(set(emotions))) * 100:.2f}%")

    print("\nClassification Report (Emotion):")
    print(classification_report(y_emotion_test, emotion_pred))

    # 3. Distance Analysis
    print("\n" + "=" * 80)
    print("3. DISTANCE ANALYSIS")
    print("=" * 80)
    print("Computing within vs between distances...")

    distance_stats = compute_distance_statistics(embeddings, speakers, emotions)

    print(f"\n📊 Speaker-based distances:")
    print(f"  Within-speaker mean: {distance_stats['within_speaker'].mean():.4f}")
    print(f"  Between-speaker mean: {distance_stats['between_speaker'].mean():.4f}")
    print(f"  Separation ratio: {distance_stats['between_speaker'].mean() / distance_stats['within_speaker'].mean():.4f}×")

    print(f"\n📊 Emotion-based distances:")
    print(f"  Within-emotion mean: {distance_stats['within_emotion'].mean():.4f}")
    print(f"  Between-emotion mean: {distance_stats['between_emotion'].mean():.4f}")
    print(f"  Separation ratio: {distance_stats['between_emotion'].mean() / distance_stats['within_emotion'].mean():.4f}×")

    # Compare speaker vs emotion separation
    speaker_separation = distance_stats['between_speaker'].mean() / distance_stats['within_speaker'].mean()
    emotion_separation = distance_stats['between_emotion'].mean() / distance_stats['within_emotion'].mean()

    print(f"\n🔍 Relative Information Content:")
    print(f"  Speaker separation: {speaker_separation:.4f}×")
    print(f"  Emotion separation: {emotion_separation:.4f}×")
    if speaker_separation > emotion_separation:
        ratio = speaker_separation / emotion_separation
        print(f"  ⚠️  Embeddings encode {ratio:.2f}× MORE speaker info than emotion info")
    else:
        ratio = emotion_separation / speaker_separation
        print(f"  ✓ Embeddings encode {ratio:.2f}× MORE emotion info than speaker info")

    # 4. Visualization
    print("\n" + "=" * 80)
    print("4. VISUALIZATION")
    print("=" * 80)

    visualize_embeddings_tsne(embeddings, speakers, emotions, output_dir)
    plot_distance_distributions(distance_stats, output_dir)

    # 5. Cross-speaker test (if provided)
    if args.test_dir:
        print("\n" + "=" * 80)
        print("5. CROSS-SPEAKER TEST")
        print("=" * 80)

        test_embeddings, test_speakers, test_emotions, test_ids = load_dataset_with_embeddings(
            args.test_dir, args.embedding_type
        )

        print(f"\n📂 Loaded test data: {len(test_embeddings)} samples")
        print(f"  Test speakers: {sorted(set(test_speakers))}")

        # Try to predict speaker (should fail for unseen speakers)
        test_speaker_pred = speaker_clf.predict(test_embeddings)

        # Try to predict emotion
        test_emotion_pred = emotion_clf.predict(test_embeddings)
        test_emotion_acc = accuracy_score(test_emotions, test_emotion_pred)

        print(f"\n✓ Emotion classification on test set: {test_emotion_acc * 100:.2f}%")
        print(f"  Emotion baseline: {100.0 / len(set(emotions)):.2f}%")

        if test_emotion_acc < emotion_acc:
            degradation = (emotion_acc - test_emotion_acc) / emotion_acc * 100
            print(f"  ⚠️  Performance degraded by {degradation:.1f}%")
        else:
            improvement = (test_emotion_acc - emotion_acc) / emotion_acc * 100
            print(f"  ✓ Performance improved by {improvement:.1f}%")

        print("\nTest Emotion Classification Report:")
        print(classification_report(test_emotions, test_emotion_pred))

    # Summary
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)

    print(f"\n📊 Summary:")
    print(f"  Speaker classification accuracy: {speaker_acc * 100:.2f}%")
    print(f"  Emotion classification accuracy: {emotion_acc * 100:.2f}%")
    print(f"  Speaker separation ratio: {speaker_separation:.4f}×")
    print(f"  Emotion separation ratio: {emotion_separation:.4f}×")

    if speaker_acc > 0.9:
        print(f"\n⚠️  HIGH SPEAKER INFORMATION CONTENT:")
        print(f"  Embeddings encode strong speaker identity ({speaker_acc * 100:.1f}% accuracy)")
        print(f"  This may hinder cross-speaker generalization")

    if speaker_separation > emotion_separation:
        print(f"\n⚠️  SPEAKER DOMINANCE:")
        print(f"  Speaker information dominates emotion information in embedding space")
        print(f"  Ratio: {speaker_separation / emotion_separation:.2f}×")

    # Save results to JSON
    results = {
        'dataset': args.data_dir,
        'embedding_type': args.embedding_type,
        'num_samples': len(embeddings),
        'embedding_dim': int(embeddings.shape[1]),
        'num_speakers': len(set(speakers)),
        'num_emotions': len(set(emotions)),
        'speaker_classification_accuracy': float(speaker_acc),
        'emotion_classification_accuracy': float(emotion_acc),
        'speaker_separation_ratio': float(speaker_separation),
        'emotion_separation_ratio': float(emotion_separation),
        'within_speaker_mean_dist': float(distance_stats['within_speaker'].mean()),
        'between_speaker_mean_dist': float(distance_stats['between_speaker'].mean()),
        'within_emotion_mean_dist': float(distance_stats['within_emotion'].mean()),
        'between_emotion_mean_dist': float(distance_stats['between_emotion'].mean()),
    }

    if args.test_dir:
        results['test_emotion_accuracy'] = float(test_emotion_acc)

    results_path = output_dir / f'{args.embedding_type}_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Results saved to: {results_path}")
    print()

if __name__ == '__main__':
    main()
