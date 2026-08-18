import torch

from config import (
    BLOCK_SIZE,
    BATCH_SIZE,
    TRAIN_RATIO,
)


class TextDataset:
    """
    Dataset for character-level language modeling.

    The dataset creates input/target pairs where the target
    is the input sequence shifted one character to the right.
    """

    def __init__(self, data):

        self.data = torch.tensor(
            data,
            dtype=torch.long
        )

        split_index = int(
            TRAIN_RATIO * len(self.data)
        )

        self.train_data = self.data[
            :split_index
        ]

        self.validation_data = self.data[
            split_index:
        ]

    # ==================================================
    # Dataset Access
    # ==================================================

    def get_train_data(self):
        """
        Return the training tokens.
        """

        return self.train_data

    def get_validation_data(self):
        """
        Return the validation tokens.
        """

        return self.validation_data

    # ==================================================
    # Batch Generation
    # ==================================================

    def get_batch(
        self,
        split="train"
    ):
        """
        Generate a random batch.

        Args:
            split:
                "train" or "val"

        Returns:
            x:
                Input token sequences.

            y:
                Target token sequences.
        """

        if split == "train":

            data = self.train_data

        elif split in ["val", "validation"]:

            data = self.validation_data

        else:

            raise ValueError(
                "split must be 'train' or 'val'"
            )

        # ------------------------------------------------
        # Random starting positions
        # ------------------------------------------------

        max_start = (
            len(data)
            - BLOCK_SIZE
            - 1
        )

        starts = torch.randint(
            0,
            max_start,
            (
                BATCH_SIZE,
            )
        )

        # ------------------------------------------------
        # Input sequences
        # ------------------------------------------------

        x = torch.stack(
            [
                data[
                    start:
                    start + BLOCK_SIZE
                ]
                for start in starts
            ]
        )

        # ------------------------------------------------
        # Target sequences
        # ------------------------------------------------

        y = torch.stack(
            [
                data[
                    start + 1:
                    start + BLOCK_SIZE + 1
                ]
                for start in starts
            ]
        )

        return x, y