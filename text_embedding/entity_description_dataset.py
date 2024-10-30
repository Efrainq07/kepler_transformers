import torch
from torch.utils.data import Dataset, DataLoader

class EntityDescriptionDataset(Dataset):
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.data = []
        self._load_data()

    def _load_data(self):
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                entity_id, entity_description = line.strip().split('\t', 1)
                self.data.append(entity_description)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_description_dataloader(input_file: str, batch_size: int, shuffle: bool = False):
    dataset = EntityDescriptionDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Example usage:
# dataloader = create_description_dataloader(input_file="entity_descriptions.txt", batch_size=32)
