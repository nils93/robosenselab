import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from scripts.pose_dataset import PoseDataset
from scripts.model import MultiTaskCNN

# am Anfang von cnn_train.py:
def train_cnn():

    # ===== 1. Geräteeinstellung =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")

    # ===== 2. Dataset laden =====
    data_dir = "data"
    csv_path = os.path.join(data_dir, "pose_labels.csv")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    df = PoseDataset.load_dataframe(csv_path)
    train_df = df[df['filepath'].str.startswith('train/')].reset_index(drop=True)
    val_df = df[df['filepath'].str.startswith('val/')].reset_index(drop=True)

    train_set = PoseDataset(df=train_df, root_dir=data_dir, transform=transform)
    val_set = PoseDataset(df=val_df, root_dir=data_dir, transform=transform)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    # ===== 3. Modell initialisieren oder laden =====
    num_classes = len(train_set.label2idx)
    model = MultiTaskCNN(num_classes=num_classes).to(device)
    model_path = "cnn_pose_model.pt"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("📥 Vortrainiertes Modell geladen – Training wird fortgesetzt.")
    else:
        print("🆕 Neues Modell wird initialisiert.")

    # ===== 4. Losses & Optimizer =====
    criterion_class = nn.CrossEntropyLoss()
    criterion_pose = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ===== 5. Training =====
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_train, correct_train = 0, 0
        train_loss = 0

        for images, labels, translations, quaternions in train_loader:
            images, labels = images.to(device), labels.to(device)
            translations, quaternions = translations.to(device), quaternions.to(device)

            optimizer.zero_grad()
            class_out, pose_out = model(images)

            loss_class = criterion_class(class_out, labels)
            loss_pose = criterion_pose(pose_out[:, :3], translations) + criterion_pose(pose_out[:, 3:], quaternions)
            loss = loss_class + 0.5 * loss_pose

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = class_out.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_acc = 100 * correct_train / total_train

        # ===== Validierung =====
        model.eval()
        correct_val, total_val = 0, 0
        val_loss = 0

        with torch.no_grad():
            for images, labels, translations, quaternions in val_loader:
                images, labels = images.to(device), labels.to(device)
                translations, quaternions = translations.to(device), quaternions.to(device)

                class_out, pose_out = model(images)

                loss_class = criterion_class(class_out, labels)
                loss_pose = criterion_pose(pose_out[:, :3], translations) + criterion_pose(pose_out[:, 3:], quaternions)
                loss = loss_class + 0.5 * loss_pose
                val_loss += loss.item()

                preds = class_out.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100 * correct_val / total_val

        print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
        print(f"   🔹 Train Acc: {train_acc:.2f}%   Loss: {train_loss:.2f}")
        print(f"   🔸 Val   Acc: {val_acc:.2f}%   Loss: {val_loss:.2f}")

    # ===== 6. Speichern =====
    torch.save(model.state_dict(), model_path)
    print("\n💾 Modell gespeichert als cnn_pose_model.pt")

if __name__ == "__main__":
    train_cnn()
