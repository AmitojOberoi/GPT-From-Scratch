import torch
from dataset import TextDataset
from tokenizer import CharacterTokenizer


def load_text(path):
    """
    Load training corpus.
    """
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def main():

    text = load_text("data/wizard_of_oz.txt")

    print("=" * 60)
    print("Preview")
    print("=" * 60)

    print(text[:300])

    tokenizer = CharacterTokenizer(text)

    print("\nVocabulary Size:", tokenizer.vocab_size)

    sample = "hello"

    encoded = tokenizer.encode(sample)

    print("\nOriginal :", sample)
    print("Encoded  :", encoded)
    print("Decoded  :", tokenizer.decode(encoded))

    # ---------------------------------------------------
    # Convert the complete dataset into a PyTorch tensor
    # ---------------------------------------------------

    encoded_text = tokenizer.encode(text)

    dataset = TextDataset(encoded_text)
    x, y = dataset.get_batch()
    print("\nBatch Information")
    print("=" * 60)

    print("Inputs")
    print(x)

    print("\nTargets")
    print(y)

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


if __name__ == "__main__":
    main()