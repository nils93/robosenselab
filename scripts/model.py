import torch.nn as nn

class MultiTaskCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.shared_fc = nn.Linear(32 * 32 * 32, 128)

        self.classifier = nn.Linear(128, num_classes)
        self.pose_regressor = nn.Linear(128, 7)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.shared_fc(x)
        return self.classifier(x), self.pose_regressor(x)
