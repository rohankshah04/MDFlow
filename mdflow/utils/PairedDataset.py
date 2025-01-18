import torch
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset) - 1

    def __getitem__(self, index):
        first = self.original_dataset[index]
        second = self.original_dataset[index + 1]
        return (first, second)

