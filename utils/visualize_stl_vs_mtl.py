from pathlib import Path
import sys
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.bdd_dataset import BDDMultiTaskDataset
from models.seg_model import SimpleSegModel
from models.drivable_model import SimpleDrivableModel
from models.mtl_model import MultiTaskModel


def denormalize_image(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    return img


def main():
    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "results"

    image_dir = r"D:\ADL PROJECT\bdd100k_images_10k\10k\train"
    seg_dir = r"D:\ADL PROJECT\bdd100k_seg_maps\labels\train"
    drive_dir = r"D:\ADL PROJECT\bdd100k_drivable_maps\labels\train"

    with open(data_dir / "overlap_ids.txt", "r") as f:
        overlap_ids = [line.strip() for line in f]

    val_idx = np.load(data_dir / "val_idx.npy")

    dataset = BDDMultiTaskDataset(
        image_dir=image_dir,
        seg_dir=seg_dir,
        drive_dir=drive_dir,
        overlap_ids=overlap_ids,
        image_size=(512, 256),
    )

    val_data = Subset(dataset, val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # STL models
    stl_seg_model = SimpleSegModel(num_classes=19).to(device)
    stl_seg_model.load_state_dict(
        torch.load(results_dir / "stl_segmentation_best.pth", map_location=device)
    )
    stl_seg_model.eval()

    stl_drive_model = SimpleDrivableModel(num_classes=3).to(device)
    stl_drive_model.load_state_dict(
        torch.load(results_dir / "stl_drivable_best.pth", map_location=device)
    )
    stl_drive_model.eval()

    # MTL model
    mtl_model = MultiTaskModel(seg_classes=19, drive_classes=3).to(device)
    mtl_model.load_state_dict(
        torch.load(results_dir / "mtl_best.pth", map_location=device)
    )
    mtl_model.eval()

    sample_indices = random.sample(range(len(val_data)), 3)

    for idx in sample_indices:
        image, seg_mask, drive_mask = val_data[idx]

        input_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            stl_seg_out = stl_seg_model(input_tensor)
            stl_seg_pred = torch.argmax(stl_seg_out, dim=1).squeeze().cpu().numpy()

            stl_drive_out = stl_drive_model(input_tensor)
            stl_drive_pred = torch.argmax(stl_drive_out, dim=1).squeeze().cpu().numpy()

            mtl_seg_out, mtl_drive_out = mtl_model(input_tensor)
            mtl_seg_pred = torch.argmax(mtl_seg_out, dim=1).squeeze().cpu().numpy()
            mtl_drive_pred = torch.argmax(mtl_drive_out, dim=1).squeeze().cpu().numpy()

        image_np = denormalize_image(image)
        seg_gt = seg_mask.cpu().numpy()
        drive_gt = drive_mask.cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(18, 8))

        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title("RGB Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(seg_gt, cmap="viridis")
        axes[0, 1].set_title("GT Segmentation")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(stl_seg_pred, cmap="viridis")
        axes[0, 2].set_title("STL Seg Prediction")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(mtl_seg_pred, cmap="viridis")
        axes[0, 3].set_title("MTL Seg Prediction")
        axes[0, 3].axis("off")

        axes[1, 0].imshow(image_np)
        axes[1, 0].set_title("RGB Image")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(drive_gt, cmap="viridis")
        axes[1, 1].set_title("GT Drivable")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(stl_drive_pred, cmap="viridis")
        axes[1, 2].set_title("STL Drive Prediction")
        axes[1, 2].axis("off")

        axes[1, 3].imshow(mtl_drive_pred, cmap="viridis")
        axes[1, 3].set_title("MTL Drive Prediction")
        axes[1, 3].axis("off")

        plt.tight_layout()

        save_path = results_dir / f"stl_vs_mtl_comparison_{idx}.png"
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved comparison to: {save_path}")

        plt.show()
        plt.close()


if __name__ == "__main__":
    main()