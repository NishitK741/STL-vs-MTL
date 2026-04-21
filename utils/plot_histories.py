from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
results_dir = PROJECT_ROOT / "results"

# ---------------- STL Segmentation ----------------
seg_history = pd.read_csv(results_dir / "stl_segmentation_history.csv")

plt.figure(figsize=(8, 5))
plt.plot(seg_history["epoch"], seg_history["train_loss"], marker="o", label="Train Loss")
plt.plot(seg_history["epoch"], seg_history["val_loss"], marker="o", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("STL Segmentation Loss")
plt.legend()
plt.grid(True)
plt.savefig(results_dir / "stl_segmentation_loss.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(seg_history["epoch"], seg_history["miou"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("mIoU")
plt.title("STL Segmentation mIoU")
plt.grid(True)
plt.savefig(results_dir / "stl_segmentation_miou.png")
plt.show()

# ---------------- STL Drivable ----------------
drive_history = pd.read_csv(results_dir / "stl_drivable_history.csv")

plt.figure(figsize=(8, 5))
plt.plot(drive_history["epoch"], drive_history["train_loss"], marker="o", label="Train Loss")
plt.plot(drive_history["epoch"], drive_history["val_loss"], marker="o", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("STL Drivable Loss")
plt.legend()
plt.grid(True)
plt.savefig(results_dir / "stl_drivable_loss.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(drive_history["epoch"], drive_history["val_acc"], marker="o", label="Val Accuracy")
plt.plot(drive_history["epoch"], drive_history["val_miou"], marker="o", label="Val mIoU")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("STL Drivable Metrics")
plt.legend()
plt.grid(True)
plt.savefig(results_dir / "stl_drivable_metrics.png")
plt.show()

# ---------------- MTL ----------------
mtl_history = pd.read_csv(results_dir / "mtl_history.csv")

plt.figure(figsize=(8, 5))
plt.plot(mtl_history["epoch"], mtl_history["train_total_loss"], marker="o", label="Train Total Loss")
plt.plot(mtl_history["epoch"], mtl_history["val_total_loss"], marker="o", label="Val Total Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MTL Total Loss")
plt.legend()
plt.grid(True)
plt.savefig(results_dir / "mtl_total_loss.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(mtl_history["epoch"], mtl_history["val_seg_miou"], marker="o", label="Seg mIoU")
plt.plot(mtl_history["epoch"], mtl_history["val_drive_miou"], marker="o", label="Drive mIoU")
plt.xlabel("Epoch")
plt.ylabel("mIoU")
plt.title("MTL Task mIoU")
plt.legend()
plt.grid(True)
plt.savefig(results_dir / "mtl_task_miou.png")
plt.show()

print("All plots saved in results folder.")
