import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BLOCK_SIZE


class SelfAttentionHead(nn.Module):
    """
    A single self-attention head.

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

        batch_size, sequence_length, _ = x.shape

        # Query, Key and Value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # QK^T
        scores = query @ key.transpose(-2, -1)

        # Scale by sqrt(d_k)
        scores = scores / (self.head_size ** 0.5)

        # Causal masking
        mask = self.tril[
            :sequence_length,
            :sequence_length
        ]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Softmax
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # Weighted values
        output = attention_weights @ value

        return output


class MultiHeadAttention(nn.Module):
    """
    Multiple self-attention heads running in parallel.

    Each head learns a different representation of
    relationships between tokens.
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

        # Mix information from all attention heads.
        self.projection = nn.Linear(
            embedding_size,
            embedding_size
        )

    def forward(self, x):

        # Run every attention head.
        outputs = [
            head(x)
            for head in self.heads
        ]

        # Concatenate along the embedding dimension.
        output = torch.cat(
            outputs,
            dim=-1
        )

        # Final linear projection.
        output = self.projection(output)

        return output


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
    print("Multi-Head Self-Attention")
    print("=" * 60)

    print("\nInput Shape:")
    print(x.shape)

    # ==================================================
    # Create Multi-Head Attention
    # ==================================================

    multi_head_attention = MultiHeadAttention(
        embedding_size=embedding_size,
        num_heads=num_heads
    )

    print("\nArchitecture")
    print("=" * 60)

    print(multi_head_attention)

    # ==================================================
    # Forward Pass
    # ==================================================

    output = multi_head_attention(x)

    print("\nOutput Shape:")
    print(output.shape)

    # ==================================================
    # Head Information
    # ==================================================

    head_size = embedding_size // num_heads

    print("\nAttention Configuration")
    print("=" * 60)

    print(
        "Embedding Size :",
        embedding_size
    )

    print(
        "Number of Heads:",
        num_heads
    )

    print(
        "Head Size      :",
        head_size
    )

    # ==================================================
    # Parameter Count
    # ==================================================

    parameter_count = sum(
        parameter.numel()
        for parameter in multi_head_attention.parameters()
    )

    print(
        "\nTotal Parameters:",
        parameter_count
    )


if __name__ == "__main__":
    main()