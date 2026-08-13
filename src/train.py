import torch

from config import (
    RANDOM_SEED,
    LEARNING_RATE,
    MAX_ITERS,
    EVAL_INTERVAL,
    GENERATE_TOKENS,
)

from dataset import TextDataset
from model import BigramLanguageModel
from tokenizer import CharacterTokenizer


# ======================================================
# Reproducibility
# ======================================================

torch.manual_seed(RANDOM_SEED)


# ======================================================
# Loss Evaluation
# ======================================================

@torch.no_grad()
def estimate_loss(model, dataset, eval_iters=100):
    """
    Estimate average loss on the training and
    validation datasets.
    """

    model.eval()

    losses = {}

    for split in ["train", "val"]:

        split_losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            x, y = dataset.get_batch(split)

            _, loss = model(x, y)

            split_losses[k] = loss.item()

        losses[split] = split_losses.mean().item()

    model.train()

    return losses


# ======================================================
# Data Loading
# ======================================================

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


# ======================================================
# Main Training Pipeline
# ======================================================

def main():

    # ==================================================
    # Load Dataset
    # ==================================================

    text = load_text(
        "data/wizard_of_oz.txt"
    )

    print("=" * 60)
    print("Preview")
    print("=" * 60)

    print(text[:300])

    # ==================================================
    # Tokenizer
    # ==================================================

    tokenizer = CharacterTokenizer(text)

    print(
        "\nVocabulary Size:",
        tokenizer.vocab_size
    )

    sample = "hello"

    encoded = tokenizer.encode(sample)

    print("\nOriginal :", sample)
    print("Encoded  :", encoded)
    print(
        "Decoded  :",
        tokenizer.decode(encoded)
    )

    # ==================================================
    # Dataset
    # ==================================================

    encoded_text = tokenizer.encode(text)

    dataset = TextDataset(encoded_text)

    train_data = dataset.get_train_data()
    val_data = dataset.get_validation_data()

    print("\nDataset Information")
    print("=" * 60)

    print(
        f"Training Tokens   : "
        f"{len(train_data)}"
    )

    print(
        f"Validation Tokens : "
        f"{len(val_data)}"
    )

    print("\nTraining Preview")

    print(
        train_data[:100]
    )

    print(
        f"\nShape : "
        f"{train_data.shape}"
    )

    print(
        f"Type  : "
        f"{train_data.dtype}"
    )

    # ==================================================
    # Batch Generation
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

    logits, initial_loss = model(
        x,
        y
    )

    print("\nInitial Model Information")
    print("=" * 60)

    print(
        "Input Shape :",
        x.shape
    )

    print(
        "Logits Shape:",
        logits.shape
    )

    print(
        "Initial Loss:",
        initial_loss.item()
    )

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
        # Evaluate periodically
        # ------------------------------------------------

        if step % EVAL_INTERVAL == 0:

            losses = estimate_loss(
                model,
                dataset
            )

            print(
                f"Step {step:4d} | "
                f"Train Loss: "
                f"{losses['train']:.4f} | "
                f"Val Loss: "
                f"{losses['val']:.4f}"
            )

        # ------------------------------------------------
        # Get training batch
        # ------------------------------------------------

        x, y = dataset.get_batch(
            "train"
        )

        # ------------------------------------------------
        # Forward Pass
        # ------------------------------------------------

        logits, loss = model(
            x,
            y
        )

        # ------------------------------------------------
        # Clear Gradients
        # ------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ------------------------------------------------
        # Backpropagation
        # ------------------------------------------------

        loss.backward()

        # ------------------------------------------------
        # Update Parameters
        # ------------------------------------------------

        optimizer.step()

    # ==================================================
    # Final Evaluation
    # ==================================================

    final_losses = estimate_loss(
        model,
        dataset
    )

    print("\nTraining Complete")
    print("=" * 60)

    print(
        f"Final Train Loss: "
        f"{final_losses['train']:.4f}"
    )

    print(
        f"Final Val Loss  : "
        f"{final_losses['val']:.4f}"
    )

    # ==================================================
    # Text Generation
    # ==================================================

    print("\nGenerated Text")
    print("=" * 60)

    # Starting character.
    start_text = "D"

    # Convert starting character to token ID.
    start_tokens = tokenizer.encode(
        start_text
    )

    # Convert token IDs to a PyTorch tensor.
    context = torch.tensor(
        [start_tokens],
        dtype=torch.long
    )

    # Generate new tokens.
    generated_tokens = model.generate(
        context,
        max_new_tokens=GENERATE_TOKENS
    )

    # Convert generated token IDs back to text.
    generated_text = tokenizer.decode(
        generated_tokens[0].tolist()
    )

    print(generated_text)


# ======================================================
# Program Entry Point
# ======================================================

if __name__ == "__main__":
    main()