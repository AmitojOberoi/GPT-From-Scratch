import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """
    A simple character-level Bigram Language Model.

    Each token is mapped directly to a set of logits
    representing the predicted next-token distribution.
    """

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=vocab_size
        )

    def forward(self, index, targets=None):
        """
        Perform a forward pass.

        Args:
            index: Input token IDs.
            targets: Expected next-token IDs.

        Returns:
            logits: Model predictions.
            loss: Cross-entropy loss if targets are provided.
        """

        logits = self.token_embedding_table(index)

        loss = None

        if targets is not None:

            batch_size, sequence_length, vocab_size = logits.shape

            logits = logits.view(
                batch_size * sequence_length,
                vocab_size
            )

            targets = targets.view(
                batch_size * sequence_length
            )

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss