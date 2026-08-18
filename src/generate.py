import argparse

import torch

from config import (
    BLOCK_SIZE,
    DEVICE,
    CHECKPOINT_PATH,
)

from model import GPTLanguageModel
from tokenizer import CharacterTokenizer


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
# Load Checkpoint
# ======================================================

def load_checkpoint(path):
    """
    Load a trained GPT checkpoint.
    """

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    return checkpoint


# ======================================================
# Create Model
# ======================================================

def create_model(
    checkpoint,
    vocab_size
):
    """
    Reconstruct the GPT architecture
    from checkpoint configuration.
    """

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        embedding_size=checkpoint[
            "embedding_size"
        ],
        num_heads=checkpoint[
            "num_heads"
        ],
        num_layers=checkpoint[
            "num_layers"
        ]
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ======================================================
# Generate Text
# ======================================================

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens
):
    """
    Generate text from a prompt.
    """

    encoded_prompt = tokenizer.encode(
        prompt
    )

    context = torch.tensor(
        [encoded_prompt],
        dtype=torch.long,
        device=DEVICE
    )

    generated = model.generate(
        context,
        max_new_tokens=max_new_tokens
    )

    generated_tokens = (
        generated[0]
        .cpu()
        .tolist()
    )

    return tokenizer.decode(
        generated_tokens
    )


# ======================================================
# Main
# ======================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate text using the trained GPT model."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="D",
        help="Starting text for generation."
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=300,
        help="Number of new tokens to generate."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT_PATH,
        help="Path to the model checkpoint."
    )

    args = parser.parse_args()

    # ==================================================
    # Load Corpus
    # ==================================================

    text = load_text(
        "data/wizard_of_oz.txt"
    )

    # ==================================================
    # Tokenizer
    # ==================================================

    tokenizer = CharacterTokenizer(
        text
    )

    # ==================================================
    # Load Checkpoint
    # ==================================================

    checkpoint = load_checkpoint(
        args.checkpoint
    )

    # ==================================================
    # Create Model
    # ==================================================

    model = create_model(
        checkpoint,
        tokenizer.vocab_size
    )

    # ==================================================
    # Information
    # ==================================================

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("=" * 60)
    print("GPT Text Generation")
    print("=" * 60)

    print(
        f"\nDevice       : {DEVICE}"
    )

    print(
        f"Vocabulary   : {tokenizer.vocab_size}"
    )

    print(
        f"Parameters   : {parameter_count}"
    )

    print(
        f"Context Size : {BLOCK_SIZE}"
    )

    print(
        f"Prompt       : {args.prompt}"
    )

    print(
        f"New Tokens   : {args.tokens}"
    )

    print("\nGenerated Text")
    print("=" * 60)

    # ==================================================
    # Generate
    # ==================================================

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.tokens
    )

    print(generated_text)


if __name__ == "__main__":
    main()