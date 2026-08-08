from pyexpat import model

import torch
from model import BigramLanguageModel
from config import RANDOM_SEED
from dataset import TextDataset
from tokenizer import CharacterTokenizer

# ------------------------------------------------------
# Reproducibility
# ------------------------------------------------------

torch.manual_seed(RANDOM_SEED)


def load_text(path):
    """
    Load the training corpus.
    """

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def main():

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    text = load_text("data/wizard_of_oz.txt")

    print("=" * 60)
    print("Preview")
    print("=" * 60)

    print(text[:300])

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = CharacterTokenizer(text)

    print("\nVocabulary Size:", tokenizer.vocab_size)

    sample = "hello"

    encoded = tokenizer.encode(sample)

    print("\nOriginal :", sample)
    print("Encoded  :", encoded)
    print("Decoded  :", tokenizer.decode(encoded))

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Batch Generation
    # --------------------------------------------------

    x, y = dataset.get_batch()

   
# Bigram Language Model


    model = BigramLanguageModel(tokenizer.vocab_size)

    logits = model(x)

    print("\nModel Information")
    print("=" * 60)

    print("Input Shape :", x.shape)
    print("Logits Shape:", logits.shape)

    print("\nBatch Information")
    print("=" * 60)

    print("\nInputs")
    print(x)

    print("\nTargets")
    print(y)


if __name__ == "__main__":
    main()