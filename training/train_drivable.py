from pathlib import Path
import sys
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.bdd_dataset import BDDMultiTaskDataset
from models.drivable_model import SimpleDrivableModel

import torch


def pixel_accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def mean_iou(outputs, targets, num_classes=3):
    preds = torch.argmax(outputs, dim=1)

    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union > 0:
            ious.append(intersection / union)

    return sum(ious) / len(ious) if len(ious) > 0 else 0.0

def evaluate_drivable(model, val_loader, device, num_classes=3):
    model.eval()
    total_val_loss = 0
    total_acc = 0
    total_miou = 0

    with torch.no_grad():
        for images, _, drive_masks in val_loader:
            images = images.to(device)
            drive_masks = drive_masks.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, drive_masks)

            total_val_loss += loss.item()
            total_acc += pixel_accuracy(outputs, drive_masks)
            total_miou += mean_iou(outputs, drive_masks, num_classes=num_classes)

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc = total_acc / len(val_loader)
    avg_val_miou = total_miou / len(val_loader)

    return avg_val_loss, avg_val_acc, avg_val_miou


def main():
    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    image_dir = r"D:\ADL PROJECT\bdd100k_images_10k\10k\train"
    seg_dir = r"D:\ADL PROJECT\bdd100k_seg_maps\labels\train"
    drive_dir = r"D:\ADL PROJECT\bdd100k_drivable_maps\labels\train"

    with open(data_dir / "overlap_ids.txt", "r") as f:
        overlap_ids = [line.strip() for line in f]

    train_idx = np.load(data_dir / "train_idx.npy")
    val_idx = np.load(data_dir / "val_idx.npy")

    print(f"Loaded overlap IDs: {len(overlap_ids)}")
    print(f"Train idx count: {len(train_idx)}")
    print(f"Val idx count: {len(val_idx)}")

    dataset = BDDMultiTaskDataset(
        image_dir=image_dir,
        seg_dir=seg_dir,
        drive_dir=drive_dir,
        overlap_ids=overlap_ids,
        image_size=(512, 256),
    )

    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SimpleDrivableModel(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    best_val_loss=float("inf")
    patience=3
    epochs_without_improvement=0
    history=[]

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch_idx, (images, _, drive_masks) in enumerate(train_loader):
            images = images.to(device)
            drive_masks = drive_masks.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, drive_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss, avg_val_acc, avg_val_miou = evaluate_drivable(model, val_loader, device, num_classes=3)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            checkpoint_path = results_dir / "stl_drivable_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved to: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {avg_val_acc:.4f} | "
            f"Val mIoU: {avg_val_miou:.4f}"
        )

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

        history.append({
            "epoch":epoch+1,
            "train_loss":avg_train_loss,
            "val_loss":avg_val_loss,
            "val_acc":avg_val_acc,
            "val_miou":avg_val_miou
        })

    checkpoint_path = results_dir / "stl_drivable_local.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")

    history_df = pd.DataFrame(history)
    history_path = results_dir / "stl_drivable_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Metrics saved to: {history_path}")


if __name__ == "__main__":
    main()

