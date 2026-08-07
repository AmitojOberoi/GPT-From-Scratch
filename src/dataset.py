import torch


class TextDataset:
    """
    Handles dataset preparation for language model training.
    """

    def __init__(
        self,
        encoded_text,
        train_ratio=0.9,
        block_size=8,
        batch_size=4,
    ):

        self.data = torch.tensor(
            encoded_text,
            dtype=torch.long
        )

        split_index = int(len(self.data) * train_ratio)

        self.train_data = self.data[:split_index]
        self.val_data = self.data[split_index:]

        self.block_size = block_size
        self.batch_size = batch_size

    def get_train_data(self):
        return self.train_data

    def get_validation_data(self):
        return self.val_data

    def get_batch(self, split="train"):

        data = self.train_data if split == "train" else self.val_data

        ix = torch.randint(
            len(data) - self.block_size,
            (self.batch_size,)
        )

        x = torch.stack([
            data[i:i + self.block_size]
            for i in ix
        ])

        y = torch.stack([
            data[i + 1:i + self.block_size + 1]
            for i in ix
        ])

        return x, y