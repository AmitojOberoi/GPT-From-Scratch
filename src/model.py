import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import TransformerBlock
from config import BLOCK_SIZE


class GPTLanguageModel(nn.Module):
    """
    GPT-style character-level language model.

    Architecture:

        Token IDs
            ↓
        Token Embeddings
            +
        Positional Embeddings
            ↓
        Transformer Blocks
            ↓
        Final LayerNorm
            ↓
        Language Model Head
            ↓
        Logits
    """

    def __init__(
        self,
        vocab_size,
        embedding_size=32,
        num_heads=4,
        num_layers=2
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # ==================================================
        # Token Embeddings
        # ==================================================

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # ==================================================
        # Positional Embeddings
        # ==================================================

        self.position_embedding_table = nn.Embedding(
            BLOCK_SIZE,
            embedding_size
        )

        # ==================================================
        # Transformer Blocks
        # ==================================================

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_size,
                    num_heads
                )
                for _ in range(num_layers)
            ]
        )

        # ==================================================
        # Final Layer Normalization
        # ==================================================

        self.final_layer_norm = nn.LayerNorm(
            embedding_size
        )

        # ==================================================
        # Language Model Head
        # ==================================================

        self.language_model_head = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, index, targets=None):
        """
        Perform a forward pass.

        Args:
            index:
                Input token IDs with shape:
                (batch_size, sequence_length)

            targets:
                Expected next-token IDs.

        Returns:
            logits:
                Shape:
                (batch_size, sequence_length, vocab_size)

            loss:
                Cross-entropy loss if targets are provided.
        """

        batch_size, sequence_length = index.shape

        # ==================================================
        # Token Embeddings
        # ==================================================

        token_embeddings = self.token_embedding_table(
            index
        )

        # ==================================================
        # Positional Embeddings
        # ==================================================

        positions = torch.arange(
            sequence_length,
            device=index.device
        )

        position_embeddings = self.position_embedding_table(
            positions
        )

        # ==================================================
        # Combine Token + Position Information
        # ==================================================

        x = token_embeddings + position_embeddings

        # ==================================================
        # Transformer Blocks
        # ==================================================

        x = self.transformer_blocks(x)

        # ==================================================
        # Final Normalization
        # ==================================================

        x = self.final_layer_norm(x)

        # ==================================================
        # Language Model Head
        # ==================================================

        logits = self.language_model_head(x)

        # ==================================================
        # Loss
        # ==================================================

        loss = None

        if targets is not None:

            batch_size, sequence_length, vocab_size = (
                logits.shape
            )

            # Keep original logits unchanged.
            # Create a flattened copy only for loss.

            logits_for_loss = logits.view(
                batch_size * sequence_length,
                vocab_size
            )

            targets_for_loss = targets.view(
                batch_size * sequence_length
            )

            loss = F.cross_entropy(
                logits_for_loss,
                targets_for_loss
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        index,
        max_new_tokens
    ):
        """
        Generate text autoregressively.

        Args:
            index:
                Starting token IDs.

            max_new_tokens:
                Number of new tokens to generate.

        Returns:
            Original tokens plus generated tokens.
        """

        for _ in range(max_new_tokens):

            # ==================================================
            # Crop context
            # ==================================================

            index_conditioned = index[
                :, -BLOCK_SIZE:
            ]

            # ==================================================
            # Get Predictions
            # ==================================================

            logits, _ = self(
                index_conditioned
            )

            # ==================================================
            # Use Final Position
            # ==================================================

            logits = logits[:, -1, :]

            # ==================================================
            # Convert Logits → Probabilities
            # ==================================================

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # ==================================================
            # Sample Next Token
            # ==================================================

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # ==================================================
            # Append Token
            # ==================================================

            index = torch.cat(
                (index, next_token),
                dim=1
            )

        return index


def main():

    # ==================================================
    # Configuration
    # ==================================================

    batch_size = 4
    sequence_length = 8
    vocab_size = 89

    embedding_size = 32
    num_heads = 4
    num_layers = 2

    # ==================================================
    # Example Input
    # ==================================================

    x = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            sequence_length
        )
    )

    y = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            sequence_length
        )
    )

    print("=" * 60)
    print("GPT Language Model")
    print("=" * 60)

    print("\nInput Shape:")
    print(x.shape)

    # ==================================================
    # Create Model
    # ==================================================

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        num_heads=num_heads,
        num_layers=num_layers
    )

    print("\nModel Architecture")
    print("=" * 60)

    print(model)

    # ==================================================
    # Forward Pass
    # ==================================================

    logits, loss = model(
        x,
        y
    )

    print("\nOutput Information")
    print("=" * 60)

    print(
        "Logits Shape:",
        logits.shape
    )

    print(
        "Loss:",
        loss.item()
    )

    # ==================================================
    # Parameter Count
    # ==================================================

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "\nTotal Parameters:",
        parameter_count
    )


if __name__ == "__main__":
    main()