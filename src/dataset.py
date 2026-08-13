import torch

from config import TRAIN_RATIO, BLOCK_SIZE, BATCH_SIZE


class TextDataset:
    """
    Handles dataset preparation and batch generation
    for language model training.
    """

    def __init__(
        self,
        encoded_text,
        train_ratio=TRAIN_RATIO,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
    ):
        """
        Initialize the dataset.

        Args:
            encoded_text: List of integer token IDs.
            train_ratio: Fraction of data used for training.
            block_size: Number of tokens in each input sequence.
            batch_size: Number of sequences per batch.
        """

        self.data = torch.tensor(
            encoded_text,
            dtype=torch.long
        )

        split_index = int(
            len(self.data) * train_ratio
        )

        self.train_data = self.data[:split_index]
        self.val_data = self.data[split_index:]

        self.block_size = block_size
        self.batch_size = batch_size

    def get_train_data(self):
        """
        Return the training data.
        """
        return self.train_data

    def get_validation_data(self):
        """
        Return the validation data.
        """
        return self.val_data

    def get_batch(self, split="train"):
        """
        Generate a random batch of input and target sequences.

        Args:
            split: Either 'train' or 'val'.

        Returns:
            x: Input token sequences.
            y: Target token sequences.
        """

        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        else:
            raise ValueError(
                "split must be either 'train' or 'val'"
            )

        indices = torch.randint(
            len(data) - self.block_size,
            (self.batch_size,)
        )

        x = torch.stack([
            data[i:i + self.block_size]
            for i in indices
        ])

        y = torch.stack([
            data[i + 1:i + self.block_size + 1]
            for i in indices
        ])

        return x, y