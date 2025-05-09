import torch
import torch.nn as nn

class MultiTaskCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 → 112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 → 56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 → 28
        )

        # Dummy-Durchlauf zur Bestimmung der Flatten-Größe
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.shared_conv(dummy)
            flatten_size = out.view(1, -1).shape[1]
            print(f"🔢 Flatten size: {flatten_size}")

        self.shared_fc = nn.Linear(flatten_size, 128)
        self.classifier = nn.Linear(128, num_classes)
        self.pose_regressor = nn.Linear(128, 7)

    def forward(self, x):
        x = self.shared_conv(x)
        x = torch.flatten(x, 1)
        x = self.shared_fc(x)
        class_out = self.classifier(x)
        pose_out = self.pose_regressor(x)
        return class_out, pose_out
