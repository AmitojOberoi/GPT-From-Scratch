import torch
import torch.nn as nn


class BigramLanguageModel(nn.Module):
    """
    A simple character-level Bigram Language Model.
    """

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=vocab_size
        )

    def forward(self, index):

        logits = self.token_embedding_table(index)

        return logits