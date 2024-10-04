import torch
from torch.utils.data import Dataset
from diplom.functions import time_tracker, generate_spectre


class CNNDataset(Dataset):
    def __init__(self, count: int, device: str, load: bool = False):
        self.heatmaps = None
        self.coefficients = None
        self.device = device
        if load:
            self.load_data()
        else:
            self.get_data(count)

    def save_data(self):
        torch.save(self.heatmaps, "heatmaps.ts")
        torch.save(self.coefficients, "coefficients.ts")

    def load_data(self):
        self.heatmaps = torch.load("heatmaps.ts")
        self.coefficients = torch.load("coefficients.ts")

    @time_tracker
    def get_data(self, count: int):
        data = [generate_spectre() for _ in range(count)]

        self.heatmaps = torch.tensor([a[0] for a in data], dtype=torch.float32, device=self.device)
        self.coefficients = torch.tensor([a[1] for a in data], dtype=torch.float32, device=self.device)

        self.save_data()

    def __len__(self):
        return len(self.heatmaps)

    def __getitem__(self, index):
        return self.heatmaps[index], self.coefficients[index]

