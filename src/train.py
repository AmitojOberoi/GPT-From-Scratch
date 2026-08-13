import torch

from config import (
    RANDOM_SEED,
    LEARNING_RATE,
    MAX_ITERS,
)
from dataset import TextDataset
from model import BigramLanguageModel
from tokenizer import CharacterTokenizer


# ======================================================
# Reproducibility
# ======================================================

torch.manual_seed(RANDOM_SEED)


def load_text(path):
    """
    Load the training corpus.

    Args:
        path: Path to the text file.

    Returns:
        The complete text corpus as a string.
    """

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def main():

    # ==================================================
    # Load Dataset
    # ==================================================

    text = load_text("data/wizard_of_oz.txt")

    print("=" * 60)
    print("Preview")
    print("=" * 60)

    print(text[:300])

    # ==================================================
    # Tokenizer
    # ==================================================

    tokenizer = CharacterTokenizer(text)

    print("\nVocabulary Size:", tokenizer.vocab_size)

    sample = "hello"

    encoded = tokenizer.encode(sample)

    print("\nOriginal :", sample)
    print("Encoded  :", encoded)
    print("Decoded  :", tokenizer.decode(encoded))

    # ==================================================
    # Dataset
    # ==================================================

    encoded_text = tokenizer.encode(text)

    dataset = TextDataset(encoded_text)

    train_data = dataset.get_train_data()
    val_data = dataset.get_validation_data()

    print("\nDataset Information")
    print("=" * 60)

    print(f"Training Tokens   : {len(train_data)}")
    print(f"Validation Tokens : {len(val_data)}")

    print("\nTraining Preview")
    print(train_data[:100])

    print(f"\nShape : {train_data.shape}")
    print(f"Type  : {train_data.dtype}")

    # ==================================================
    # Initial Batch
    # ==================================================

    x, y = dataset.get_batch()

    print("\nBatch Information")
    print("=" * 60)

    print("\nInputs")
    print(x)

    print("\nTargets")
    print(y)

    # ==================================================
    # Bigram Language Model
    # ==================================================

    model = BigramLanguageModel(
        tokenizer.vocab_size
    )

    print("\nModel Architecture")
    print("=" * 60)

    print(model)

    # ==================================================
    # Initial Forward Pass
    # ==================================================

    logits, loss = model(x, y)

    print("\nInitial Model Information")
    print("=" * 60)

    print("Input Shape :", x.shape)
    print("Logits Shape:", logits.shape)
    print("Initial Loss:", loss.item())

    # ==================================================
    # Optimizer
    # ==================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ==================================================
    # Training Loop
    # ==================================================

    print("\nStarting Training")
    print("=" * 60)

    for step in range(MAX_ITERS):

        # ------------------------------------------------
        # Get a new batch
        # ------------------------------------------------

        x, y = dataset.get_batch("train")

        # ------------------------------------------------
        # Forward pass
        # ------------------------------------------------

        logits, loss = model(x, y)

        # ------------------------------------------------
        # Clear previous gradients
        # ------------------------------------------------

        optimizer.zero_grad(set_to_none=True)

        # ------------------------------------------------
        # Backpropagation
        # ------------------------------------------------

        loss.backward()

        # ------------------------------------------------
        # Update model parameters
        # ------------------------------------------------

        optimizer.step()

        # ------------------------------------------------
        # Report progress
        # ------------------------------------------------

        if step % 100 == 0:

            print(
                f"Step {step:4d} | "
                f"Loss: {loss.item():.4f}"
            )

    # ==================================================
    # Final Loss
    # ==================================================

    print("\nTraining Complete")
    print("=" * 60)

    print(f"Final Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()