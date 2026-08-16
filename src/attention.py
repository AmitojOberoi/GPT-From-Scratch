import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BLOCK_SIZE


class SelfAttentionHead(nn.Module):
    """
    Single self-attention head.

    Implements:

        Q = XWq
        K = XWk
        V = XWv

        Attention = softmax(QK^T / sqrt(d_k))V
    """

    def __init__(self, embedding_size, head_size):
        super().__init__()

        # --------------------------------------------------
        # Learned Query, Key and Value projections
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Causal mask
        #
        # A token should not be able to see future tokens.
        # --------------------------------------------------

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
        """
        Perform scaled dot-product self-attention.

        Args:
            x: Input tensor of shape
               (batch, sequence_length, embedding_size)

        Returns:
            Attention output.
        """

        # --------------------------------------------------
        # Create Query, Key and Value
        # --------------------------------------------------

        query = self.query(x)

        key = self.key(x)

        value = self.value(x)

        # --------------------------------------------------
        # Calculate attention scores
        #
        # QK^T
        # --------------------------------------------------

        scores = query @ key.transpose(-2, -1)

        # --------------------------------------------------
        # Scale attention scores
        #
        # QK^T / sqrt(d_k)
        # --------------------------------------------------

        scores = scores / (self.head_size ** 0.5)

        # --------------------------------------------------
        # Apply causal mask
        #
        # Future tokens are hidden.
        # --------------------------------------------------

        sequence_length = x.shape[1]

        mask = self.tril[
            :sequence_length,
            :sequence_length
        ]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # --------------------------------------------------
        # Convert scores into probabilities
        # --------------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # --------------------------------------------------
        # Weighted sum of Values
        # --------------------------------------------------

        output = attention_weights @ value

        return output


def main():

    # ==================================================
    # Configuration
    # ==================================================

    batch_size = 4
    sequence_length = 8
    embedding_size = 32
    head_size = 16

    # ==================================================
    # Create Example Input
    # ==================================================

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_size
    )

    print("=" * 60)
    print("Scaled Dot-Product Self-Attention")
    print("=" * 60)

    print("\nInput Shape:")
    print(x.shape)

    # ==================================================
    # Create Attention Head
    # ==================================================

    attention_head = SelfAttentionHead(
        embedding_size=embedding_size,
        head_size=head_size
    )

    print("\nAttention Head")
    print("=" * 60)

    print(attention_head)

    # ==================================================
    # Forward Pass
    # ==================================================

    output = attention_head(x)

    print("\nOutput Shape:")
    print(output.shape)

    # ==================================================
    # Parameter Information
    # ==================================================

    print("\nParameter Information")
    print("=" * 60)

    print(
        "Query Parameters :",
        sum(
            p.numel()
            for p in attention_head.query.parameters()
        )
    )

    print(
        "Key Parameters   :",
        sum(
            p.numel()
            for p in attention_head.key.parameters()
        )
    )

    print(
        "Value Parameters :",
        sum(
            p.numel()
            for p in attention_head.value.parameters()
        )
    )

    # ==================================================
    # Verify Causal Attention
    # ==================================================

    print("\nCausal Mask")
    print("=" * 60)

    print(
        attention_head.tril[
            :sequence_length,
            :sequence_length
        ]
    )


if __name__ == "__main__":
    main()