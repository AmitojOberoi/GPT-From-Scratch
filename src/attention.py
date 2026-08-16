import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BLOCK_SIZE


class SelfAttentionHead(nn.Module):
    """
    A single causal self-attention head.

    Implements:

        Q = XWq
        K = XWk
        V = XWv

        Attention = softmax(QK^T / sqrt(d_k))V
    """

    def __init__(self, embedding_size, head_size):
        super().__init__()

        self.query = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        self.key = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    BLOCK_SIZE,
                    BLOCK_SIZE
                )
            )
        )

        self.head_size = head_size

    def forward(self, x):

        _, sequence_length, _ = x.shape

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scores = query @ key.transpose(-2, -1)

        scores = scores / (self.head_size ** 0.5)

        mask = self.tril[
            :sequence_length,
            :sequence_length
        ]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        output = attention_weights @ value

        return output


class MultiHeadAttention(nn.Module):
    """
    Multiple self-attention heads running in parallel.
    """

    def __init__(
        self,
        embedding_size,
        num_heads
    ):
        super().__init__()

        assert embedding_size % num_heads == 0

        head_size = embedding_size // num_heads

        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    embedding_size,
                    head_size
                )
                for _ in range(num_heads)
            ]
        )

        self.projection = nn.Linear(
            embedding_size,
            embedding_size
        )

    def forward(self, x):

        outputs = [
            head(x)
            for head in self.heads
        ]

        output = torch.cat(
            outputs,
            dim=-1
        )

        output = self.projection(output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward neural network.

    Expands the embedding dimension, applies a
    non-linear activation, then projects back.
    """

    def __init__(self, embedding_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_size,
                4 * embedding_size
            ),

            nn.ReLU(),

            nn.Linear(
                4 * embedding_size,
                embedding_size
            )
        )

    def forward(self, x):

        return self.network(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer block.

    Architecture:

        Input
          ↓
        LayerNorm
          ↓
        Multi-Head Attention
          ↓
        Residual Connection
          ↓
        LayerNorm
          ↓
        Feed Forward
          ↓
        Residual Connection
          ↓
        Output
    """

    def __init__(
        self,
        embedding_size,
        num_heads
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_size,
            num_heads
        )

        self.feed_forward = FeedForward(
            embedding_size
        )

        self.layer_norm_1 = nn.LayerNorm(
            embedding_size
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_size
        )

    def forward(self, x):

        # ----------------------------------------------
        # Attention + Residual Connection
        # ----------------------------------------------

        x = x + self.attention(
            self.layer_norm_1(x)
        )

        # ----------------------------------------------
        # Feed Forward + Residual Connection
        # ----------------------------------------------

        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x


def main():

    # ==================================================
    # Configuration
    # ==================================================

    batch_size = 4
    sequence_length = 8
    embedding_size = 32
    num_heads = 4

    # ==================================================
    # Example Input
    # ==================================================

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_size
    )

    print("=" * 60)
    print("Transformer Block")
    print("=" * 60)

    print("\nInput Shape:")
    print(x.shape)

    # ==================================================
    # Transformer Block
    # ==================================================

    transformer_block = TransformerBlock(
        embedding_size=embedding_size,
        num_heads=num_heads
    )

    print("\nArchitecture")
    print("=" * 60)

    print(transformer_block)

    # ==================================================
    # Forward Pass
    # ==================================================

    output = transformer_block(x)

    print("\nOutput Shape:")
    print(output.shape)

    # ==================================================
    # Parameter Count
    # ==================================================

    parameter_count = sum(
        parameter.numel()
        for parameter in transformer_block.parameters()
    )

    print("\nTotal Parameters:")
    print(parameter_count)


if __name__ == "__main__":
    main()