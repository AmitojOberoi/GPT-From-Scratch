import torch


class TextDataset:
    """
    Handles dataset preparation for language model training.
    """

    def __init__(self, encoded_text, train_ratio=0.9):

        self.data = torch.tensor(
            encoded_text,
            dtype=torch.long
        )

        split_index = int(len(self.data) * train_ratio)

        self.train_data = self.data[:split_index]
        self.val_data = self.data[split_index:]

    def get_train_data(self):
        return self.train_data

    def get_validation_data(self):
        return self.val_data