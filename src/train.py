import os

import torch

from config import (
    RANDOM_SEED,
    LEARNING_RATE,
    MAX_ITERS,
    EVAL_INTERVAL,
    EVAL_ITERS,
    GENERATE_TOKENS,
    EMBEDDING_SIZE,
    NUM_HEADS,
    NUM_LAYERS,
    BLOCK_SIZE,
    DEVICE,
    CHECKPOINT_PATH,
)

from dataset import TextDataset
from model import GPTLanguageModel
from tokenizer import CharacterTokenizer


# ======================================================
# Reproducibility
# ======================================================

torch.manual_seed(RANDOM_SEED)


# ======================================================
# Loss Evaluation
# ======================================================

@torch.no_grad()
def estimate_loss(model, dataset):
    """
    Estimate average training and validation loss.
    """

    model.eval()

    losses = {}

    for split in ["train", "val"]:

        split_losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):

            x, y = dataset.get_batch(split)

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            _, loss = model(x, y)

            split_losses[k] = loss.item()

        losses[split] = split_losses.mean().item()

    model.train()

    return losses


# ======================================================
# Load Text
# ======================================================

def load_text(path):
    """
    Load the training corpus.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ======================================================
# Main Training Pipeline
# ======================================================

def main():

    # ==================================================
    # Load Corpus
    # ==================================================

    text = load_text(
        "data/wizard_of_oz.txt"
    )

    print("=" * 60)
    print("GPT Training Pipeline")
    print("=" * 60)

    print("\nCorpus Preview")
    print("-" * 60)

    print(text[:300])

    # ==================================================
    # Tokenizer
    # ==================================================

    tokenizer = CharacterTokenizer(text)

    print(
        "\nVocabulary Size:",
        tokenizer.vocab_size
    )

    # ==================================================
    # Encode Dataset
    # ==================================================

    encoded_text = tokenizer.encode(text)

    dataset = TextDataset(encoded_text)

    train_data = dataset.get_train_data()
    validation_data = dataset.get_validation_data()

    print("\nDataset Information")
    print("=" * 60)

    print(
        f"Training Tokens   : "
        f"{len(train_data)}"
    )

    print(
        f"Validation Tokens : "
        f"{len(validation_data)}"
    )

    # ==================================================
    # Device
    # ==================================================

    print(
        "\nDevice:",
        DEVICE
    )

    # ==================================================
    # Model
    # ==================================================

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_size=EMBEDDING_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    )

    model = model.to(DEVICE)

    # ==================================================
    # Model Information
    # ==================================================

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nModel Information")
    print("=" * 60)

    print(
        "Embedding Size:",
        EMBEDDING_SIZE
    )

    print(
        "Attention Heads:",
        NUM_HEADS
    )

    print(
        "Transformer Layers:",
        NUM_LAYERS
    )

    print(
        "Context Length:",
        BLOCK_SIZE
    )

    print(
        "Total Parameters:",
        parameter_count
    )

    # ==================================================
    # Optimizer
    # ==================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ==================================================
    # Training
    # ==================================================

    print("\nStarting Training")
    print("=" * 60)

    for step in range(MAX_ITERS):

        # ------------------------------------------------
        # Periodic Evaluation
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
        # Get Training Batch
        # ------------------------------------------------

        x, y = dataset.get_batch("train")

        x = x.to(DEVICE)
        y = y.to(DEVICE)

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
    # Save Model Checkpoint
    # ==================================================

    checkpoint_directory = os.path.dirname(
        CHECKPOINT_PATH
    )

    if checkpoint_directory:
        os.makedirs(
            checkpoint_directory,
            exist_ok=True
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),

        "vocab_size": tokenizer.vocab_size,

        "embedding_size": EMBEDDING_SIZE,

        "num_heads": NUM_HEADS,

        "num_layers": NUM_LAYERS,

        "block_size": BLOCK_SIZE,

        "train_loss": final_losses["train"],

        "validation_loss": final_losses["val"],
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )

    print("\nModel Checkpoint Saved")
    print("=" * 60)

    print(
        f"Saved to: {CHECKPOINT_PATH}"
    )

    # ==================================================
    # Text Generation
    # ==================================================

    print("\nGenerated Text")
    print("=" * 60)

    start_text = "D"

    start_tokens = tokenizer.encode(
        start_text
    )

    context = torch.tensor(
        [start_tokens],
        dtype=torch.long,
        device=DEVICE
    )

    generated_tokens = model.generate(
        context,
        max_new_tokens=GENERATE_TOKENS
    )

    generated_text = tokenizer.decode(
        generated_tokens[0]
        .cpu()
        .tolist()
    )

    print(generated_text)


# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    main()