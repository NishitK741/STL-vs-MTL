from pathlib import Path
import sys
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.metrics_segmentation import (
    update_segmentation_confusion_matrix,
    compute_iou_from_confusion_matrix,
)
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


from dataset.bdd_dataset import BDDMultiTaskDataset
from models.seg_model import SimpleSegModel


def evaluate_segmentation(model, val_loader, device, num_classes=19, ignore_index=255):
    model.eval()
    total_val_loss = 0

    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    with torch.no_grad():
        for images, seg_masks, _ in val_loader:
            images = images.to(device)
            seg_masks = seg_masks.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, seg_masks, ignore_index=ignore_index)
            total_val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            update_segmentation_confusion_matrix(
                conf_matrix,
                preds,
                seg_masks,
                num_classes,
                ignore_index,
            )

    avg_val_loss = total_val_loss / len(val_loader)
    iou_per_class, miou = compute_iou_from_confusion_matrix(conf_matrix)

    return avg_val_loss, iou_per_class, miou


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

    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SimpleSegModel(num_classes=19).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss=float("inf")
    patience=3
    epochs_without_improvement=0
    num_epochs = 10
    history=[]

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch_idx, (images, seg_masks, _) in enumerate(train_loader):
            images = images.to(device)
            seg_masks = seg_masks.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, seg_masks, ignore_index=255)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss, iou_per_class, miou = evaluate_segmentation(
            model,
            val_loader,
            device,
            num_classes=19,
            ignore_index=255,
        )
        history.append({
            "epoch":epoch+1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "miou": miou.item() if hasattr(miou, "item") else float(miou),
        })
    
        if avg_val_loss< best_val_loss:
            best_val_loss=avg_val_loss
            epochs_without_improvement=0

            best_checkpoint_path =results_dir/"stl_segmentation_best.pth"
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"New best model saved to {best_checkpoint_path}")
        else:
            epochs_without_improvement+=1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"mIoU: {miou:.4f}"
        )
    
        if epochs_without_improvement>= patience:
            print("Early stopping triggered.")
            break


    print("Per-class IoU:")
    for class_idx, iou in enumerate(iou_per_class):
        print(f"Class {class_idx}: {iou.item():.4f}")

    checkpoint_path = results_dir / "stl_segmentation_local.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")

    history_df = pd.DataFrame(history)
    history_path = results_dir / "stl_segmentation_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Metrics saved to: {history_path}")

if __name__ == "__main__":
    main()