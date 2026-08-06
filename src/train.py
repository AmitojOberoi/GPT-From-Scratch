import torch
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

    data = torch.tensor(
        encoded_text,
        dtype=torch.long
    )

    print("\nTensor Information")
    print("=" * 60)

    print(data[:100])
    print(f"\nShape : {data.shape}")
    print(f"Type  : {data.dtype}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()