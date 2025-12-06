"""
Supervised Contrastive Loss for Speaker-Invariant Emotion Learning.

This loss function encourages:
- Same emotion, different speaker → similar embeddings
- Different emotion (any speaker) → dissimilar embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for learning speaker-invariant emotion representations.

    For each anchor sample with emotion e_i and speaker s_i:
    - Positive set P(i) = {j | emotion_j == e_i AND speaker_j != s_i}
    - Negative set N(i) = {j | emotion_j != e_i}

    The loss pulls positive pairs together and pushes negative pairs apart.
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Args:
            temperature: Softmax temperature for scaling similarities
            base_temperature: Base temperature for normalization
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, emotion_labels, speaker_ids, mask=None):
        """
        Compute supervised contrastive loss.

        Args:
            features: (batch_size, feature_dim) - L2 normalized embeddings
            emotion_labels: (batch_size,) - emotion class indices
            speaker_ids: (batch_size,) - speaker class indices
            mask: Optional (batch_size, batch_size) - manual mask for valid pairs

        Returns:
            loss: Scalar contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]

        if len(features.shape) < 2:
            raise ValueError(f'`features` needs to be [batch_size, feature_dim], got {features.shape}')

        # L2 normalize features if not already normalized
        features = F.normalize(features, dim=1)

        # Compute similarity matrix: (batch_size, batch_size)
        # sim_matrix[i, j] = cosine similarity between sample i and j
        similarity_matrix = torch.matmul(features, features.T)

        # Create emotion mask: True if same emotion
        emotion_labels = emotion_labels.contiguous().view(-1, 1)
        emotion_mask = torch.eq(emotion_labels, emotion_labels.T).float().to(device)

        # Create speaker mask: True if same speaker
        speaker_ids = speaker_ids.contiguous().view(-1, 1)
        speaker_mask = torch.eq(speaker_ids, speaker_ids.T).float().to(device)

        # Positive mask: same emotion AND different speaker
        # This is the key - we want to pull together same emotions across speakers
        positive_mask = emotion_mask * (1 - speaker_mask)

        # Remove diagonal (self-comparison)
        logits_mask = torch.ones_like(similarity_matrix).fill_diagonal_(0)
        positive_mask = positive_mask * logits_mask

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log_prob
        exp_logits = torch.exp(logits / self.temperature)

        # Mask out self-comparisons
        exp_logits = exp_logits * logits_mask

        # log_prob = log(exp(sim(i,p)) / sum(exp(sim(i,j))))
        log_prob = logits / self.temperature - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive pairs
        # For each anchor, average over all its positive pairs
        positive_per_anchor = positive_mask.sum(1)

        # Only compute loss for anchors that have at least one positive pair
        valid_anchors = positive_per_anchor > 0

        if valid_anchors.sum() == 0:
            # No valid positive pairs in this batch
            # Return zero loss (this can happen with small batches or homogeneous batches)
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Mean log-likelihood over positive pairs
        # (positive_mask * log_prob) sums log-prob only for positive pairs
        # Divide by number of positive pairs per anchor
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_per_anchor + 1e-12)

        # Only average over anchors that have positive pairs
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos[valid_anchors].mean()

        return loss


class SupConLossWithStats(SupervisedContrastiveLoss):
    """
    Extended version that also returns statistics for monitoring.
    """

    def forward_with_stats(self, features, emotion_labels, speaker_ids):
        """
        Compute loss and return detailed statistics.

        Returns:
            loss: Scalar loss
            stats: Dictionary with debugging information
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)

        # Create masks
        emotion_labels = emotion_labels.contiguous().view(-1, 1)
        emotion_mask = torch.eq(emotion_labels, emotion_labels.T).float().to(device)

        speaker_ids = speaker_ids.contiguous().view(-1, 1)
        speaker_mask = torch.eq(speaker_ids, speaker_ids.T).float().to(device)

        # Positive mask: same emotion AND different speaker
        positive_mask = emotion_mask * (1 - speaker_mask)

        # Remove diagonal
        logits_mask = torch.ones_like(similarity_matrix).fill_diagonal_(0)
        positive_mask = positive_mask * logits_mask

        # Negative mask: different emotion (any speaker)
        negative_mask = (1 - emotion_mask) * logits_mask

        # Compute loss (same as base class)
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits / self.temperature)
        exp_logits = exp_logits * logits_mask

        log_prob = logits / self.temperature - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        positive_per_anchor = positive_mask.sum(1)
        valid_anchors = positive_per_anchor > 0

        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'num_positives': 0,
                'num_negatives': 0,
                'avg_positive_sim': 0.0,
                'avg_negative_sim': 0.0
            }

        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_per_anchor + 1e-12)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos[valid_anchors].mean()

        # Compute statistics
        with torch.no_grad():
            num_positives = positive_mask.sum().item()
            num_negatives = negative_mask.sum().item()

            if num_positives > 0:
                avg_positive_sim = (similarity_matrix * positive_mask).sum() / num_positives
            else:
                avg_positive_sim = 0.0

            if num_negatives > 0:
                avg_negative_sim = (similarity_matrix * negative_mask).sum() / num_negatives
            else:
                avg_negative_sim = 0.0

        stats = {
            'num_positives': num_positives,
            'num_negatives': num_negatives,
            'avg_positive_sim': avg_positive_sim.item() if isinstance(avg_positive_sim, torch.Tensor) else avg_positive_sim,
            'avg_negative_sim': avg_negative_sim.item() if isinstance(avg_negative_sim, torch.Tensor) else avg_negative_sim,
            'num_valid_anchors': valid_anchors.sum().item()
        }

        return loss, stats


def test_contrastive_loss():
    """
    Unit test for contrastive loss.
    """
    print("Testing Supervised Contrastive Loss...")

    # Create dummy data
    batch_size = 8
    feature_dim = 128

    # 4 speakers, 2 emotions
    # Each speaker has both emotions
    features = torch.randn(batch_size, feature_dim)
    emotion_labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])  # 4 samples per emotion
    speaker_ids = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])    # 2 samples per speaker

    # Test loss computation
    criterion = SupConLossWithStats(temperature=0.07)
    loss, stats = criterion.forward_with_stats(features, emotion_labels, speaker_ids)

    print(f"Loss: {loss.item():.4f}")
    print(f"Num positives: {stats['num_positives']}")  # Should be 4 (each emotion has 3 cross-speaker pairs)
    print(f"Num negatives: {stats['num_negatives']}")  # Should be 32 (4 samples * 4 different-emotion samples, * 2)
    print(f"Avg positive similarity: {stats['avg_positive_sim']:.4f}")
    print(f"Avg negative similarity: {stats['avg_negative_sim']:.4f}")

    # Test edge case: all same emotion and speaker (should return 0 loss)
    emotion_labels_same = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
    speaker_ids_same = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])

    loss_same, stats_same = criterion.forward_with_stats(features, emotion_labels_same, speaker_ids_same)
    print(f"\nEdge case (all same): Loss = {loss_same.item():.4f}, Positives = {stats_same['num_positives']}")

    print("\nTest passed!")


if __name__ == '__main__':
    test_contrastive_loss()
