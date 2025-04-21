import torch
from torch.utils.data import Dataset
from diplom.functions import time_tracker, generate_spectre


class CNNDataset(Dataset):
    def __init__(self, count: int, device: str, load: bool = False, name_heat="heatmaps.ts", name_coeff="coefficients.ts"):
        self.heatmaps = None
        self.coefficients = None
        self.device = device
        self.name_heatmap = name_heat
        self.name_coefficients = name_coeff
        if load:
            self.load_data()
        else:
            self.get_data(count)

    def save_data(self):
        torch.save(self.heatmaps, self.name_heatmap)
        torch.save(self.coefficients, self.name_coefficients)

    def load_data(self):
        self.heatmaps = torch.load(self.name_heatmap)
        self.coefficients = torch.load(self.name_coefficients)

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

