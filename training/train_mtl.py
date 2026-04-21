from pathlib import Path
import sys
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.bdd_dataset import BDDMultiTaskDataset
from models.mtl_model import MultiTaskModel


def pixel_accuracy(outputs, targets, ignore_index=None):
    preds = torch.argmax(outputs, dim=1)

    if ignore_index is not None:
        valid = targets != ignore_index
        correct = (preds[valid] == targets[valid]).sum().item()
        total = valid.sum().item()
    else:
        correct = (preds == targets).sum().item()
        total = targets.numel()

    return correct / total if total > 0 else 0.0


def mean_iou(outputs, targets, num_classes, ignore_index=None):
    preds = torch.argmax(outputs, dim=1)

    if ignore_index is not None:
        valid = targets != ignore_index
        preds = preds[valid]
        targets = targets[valid]

    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union > 0:
            ious.append(intersection / union)

    return sum(ious) / len(ious) if len(ious) > 0 else 0.0


def evaluate_mtl(model, val_loader, device, seg_classes=19, drive_classes=3):
    model.eval()

    total_val_loss = 0
    total_seg_loss = 0
    total_drive_loss = 0

    total_seg_acc = 0
    total_drive_acc = 0

    total_seg_miou = 0
    total_drive_miou = 0

    with torch.no_grad():
        for images, seg_masks, drive_masks in val_loader:
            images = images.to(device)
            seg_masks = seg_masks.to(device)
            drive_masks = drive_masks.to(device)

            seg_out, drive_out = model(images)

            seg_loss = F.cross_entropy(seg_out, seg_masks, ignore_index=255)
            drive_loss = F.cross_entropy(drive_out, drive_masks)

            loss = seg_loss + drive_loss

            total_val_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_drive_loss += drive_loss.item()

            total_seg_acc += pixel_accuracy(seg_out, seg_masks, ignore_index=255)
            total_drive_acc += pixel_accuracy(drive_out, drive_masks)

            total_seg_miou += mean_iou(seg_out, seg_masks, num_classes=seg_classes, ignore_index=255)
            total_drive_miou += mean_iou(drive_out, drive_masks, num_classes=drive_classes)

    n = len(val_loader)

    return {
        "val_total_loss": total_val_loss / n,
        "val_seg_loss": total_seg_loss / n,
        "val_drive_loss": total_drive_loss / n,
        "val_seg_acc": total_seg_acc / n,
        "val_drive_acc": total_drive_acc / n,
        "val_seg_miou": total_seg_miou / n,
        "val_drive_miou": total_drive_miou / n,
    }


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

    model = MultiTaskModel(seg_classes=19, drive_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    best_val_total_loss = float("inf")
    patience = 3
    epochs_without_improvement = 0
    history=[]

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0
        total_train_seg_loss = 0
        total_train_drive_loss = 0

        for batch_idx, (images, seg_masks, drive_masks) in enumerate(train_loader):
            images = images.to(device)
            seg_masks = seg_masks.to(device)
            drive_masks = drive_masks.to(device)

            seg_out, drive_out = model(images)

            seg_loss = F.cross_entropy(seg_out, seg_masks, ignore_index=255)
            drive_loss = F.cross_entropy(drive_out, drive_masks)

            loss = seg_loss + drive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_seg_loss += seg_loss.item()
            total_train_drive_loss += drive_loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Total: {loss.item():.4f} | Seg: {seg_loss.item():.4f} | Drive: {drive_loss.item():.4f}"
                )

        avg_train_total = total_train_loss / len(train_loader)
        avg_train_seg = total_train_seg_loss / len(train_loader)
        avg_train_drive = total_train_drive_loss / len(train_loader)

        val_metrics = evaluate_mtl(model, val_loader, device, seg_classes=19, drive_classes=3)

        history.append({
            "epoch": epoch + 1,
            "train_total_loss": avg_train_total,
            "train_seg_loss": avg_train_seg,
            "train_drive_loss": avg_train_drive,
            "val_total_loss": val_metrics["val_total_loss"],
            "val_seg_loss": val_metrics["val_seg_loss"],
            "val_drive_loss": val_metrics["val_drive_loss"],
            "val_seg_acc": val_metrics["val_seg_acc"],
            "val_drive_acc": val_metrics["val_drive_acc"],
            "val_seg_miou": val_metrics["val_seg_miou"],
            "val_drive_miou": val_metrics["val_drive_miou"],
            })

        if val_metrics["val_total_loss"] < best_val_total_loss:
            best_val_total_loss = val_metrics["val_total_loss"]
            epochs_without_improvement = 0

            checkpoint_path = results_dir / "mtl_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved to: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Total: {avg_train_total:.4f} | "
            f"Train Seg: {avg_train_seg:.4f} | "
            f"Train Drive: {avg_train_drive:.4f} | "
            f"Val Total: {val_metrics['val_total_loss']:.4f} | "
            f"Val Seg: {val_metrics['val_seg_loss']:.4f} | "
            f"Val Drive: {val_metrics['val_drive_loss']:.4f} | "
            f"Val Seg Acc: {val_metrics['val_seg_acc']:.4f} | "
            f"Val Drive Acc: {val_metrics['val_drive_acc']:.4f} | "
            f"Val Seg mIoU: {val_metrics['val_seg_miou']:.4f} | "
            f"Val Drive mIoU: {val_metrics['val_drive_miou']:.4f}"
        )

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    final_checkpoint_path = results_dir / "mtl_final.pth"
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model saved to: {final_checkpoint_path}")

    history_df = pd.DataFrame(history)
    history_path = results_dir / "mtl_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Metrics saved to: {history_path}")


if __name__ == "__main__":
    main()