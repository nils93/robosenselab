from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch

class PoseDataset(Dataset):
    def __init__(self, df=None, csv_path=None, root_dir="", transform=None):
        if df is not None:
            self.data = df
        elif csv_path is not None:
            self.data = pd.read_csv(csv_path)
        else:
            raise ValueError("Entweder df oder csv_path muss angegeben werden")

        self.root_dir = root_dir
        self.transform = transform

        labels = sorted(self.data["label"].unique())
        self.label2idx = {name: idx for idx, name in enumerate(labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["filepath"])
        image = Image.open(img_path).convert("RGB")

        label_name = row["label"]
        label = self.label2idx.get(label_name, -1)

        if label_name == "background":
            # Dummy-Werte für Pose
            translation = torch.zeros(3, dtype=torch.float32)
            quaternion = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        else:
            translation = torch.tensor([row["tx"], row["ty"], row["tz"]], dtype=torch.float32)
            quaternion = torch.tensor([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=torch.float32)


        if self.transform:
            image = self.transform(image)

        return image, label, translation, quaternion
    
    @staticmethod
    def load_dataframe(csv_path):
        import pandas as pd
        return pd.read_csv(csv_path)

