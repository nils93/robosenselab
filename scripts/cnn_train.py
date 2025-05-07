import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
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

    image_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    print(f"\n🖼️  Bildinput: {image_size[0]}×{image_size[1]}")

    df = PoseDataset.load_dataframe(csv_path)
    train_df = df[df['filepath'].str.startswith('train/')].reset_index(drop=True)
    val_df = df[df['filepath'].str.startswith('val/')].reset_index(drop=True)

    train_set = PoseDataset(df=train_df, root_dir=data_dir, transform=transform)
    val_set = PoseDataset(df=val_df, root_dir=data_dir, transform=transform)

    cpu_count = os.cpu_count() or 2  # fallback
    train_loader = DataLoader(
        train_set,
        batch_size=32,  # falls dein GPU-RAM es erlaubt
        shuffle=True,
        num_workers=cpu_count // 2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=32,
        num_workers=cpu_count // 2,
        pin_memory=True
    )


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
    class_weights = torch.ones(num_classes)
    background_idx = train_set.label2idx["background"]
    class_weights[background_idx] = 0.02
    wgt = class_weights[background_idx] * 100
    print(f"🔧 {wgt}% Gewicht für Backgrounds")
    
    criterion_class = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion_pose = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler("cuda")  # für mixed precision

    # ===== 5. Training =====
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train, correct_train = 0, 0
        train_loss = 0

        for images, labels, translations, quaternions in train_loader:
            images, labels = images.to(device), labels.to(device)
            translations, quaternions = translations.to(device), quaternions.to(device)

            is_fg = (labels != train_set.label2idx["background"]).float().unsqueeze(1)  # [B,1]
            optimizer.zero_grad()
            
            with autocast("cuda"):
                class_out, pose_out = model(images)
                loss_class = criterion_class(class_out, labels)
                loss_pose_xyz = criterion_pose(pose_out[:, :3], translations) * is_fg
                loss_pose_quat = criterion_pose(pose_out[:, 3:], quaternions) * is_fg
                loss_pose = (loss_pose_xyz.mean() + loss_pose_quat.mean())
                loss = loss_class + 0.5 * loss_pose

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

                is_fg = (labels != val_set.label2idx["background"]).float().unsqueeze(1)

                with autocast("cuda"):
                    class_out, pose_out = model(images)
                    loss_class = criterion_class(class_out, labels)
                    loss_pose_xyz = criterion_pose(pose_out[:, :3], translations) * is_fg
                    loss_pose_quat = criterion_pose(pose_out[:, 3:], quaternions) * is_fg
                    loss_pose = (loss_pose_xyz.mean() + loss_pose_quat.mean())
                    loss = loss_class + 0.5 * loss_pose

                val_loss += loss.item()
                preds = class_out.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100 * correct_val / total_val

        print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
        print(f"   🔹 Train Acc: {train_acc:.2f}%   Loss: {train_loss:.2f}")
        print(f"   🔸 Val   Acc: {val_acc:.2f}%   Loss: {val_loss:.2f}")

        # ===== Early Stopping =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), model_path)
            print("💾 Bestes Modell gespeichert.")
        else:
            counter += 1
            if counter >= patience:
                print(f"⏳ Early stopping nach {patience} Epochen ohne Verbesserung.")
                break

    # ===== 6. Speichern =====
    torch.save(model.state_dict(), model_path)
    print("\n💾 Modell gespeichert als cnn_pose_model.pt")

if __name__ == "__main__":
    train_cnn()
