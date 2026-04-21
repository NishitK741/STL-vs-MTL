from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import torch
from torch.utils.data import Dataset


class BDDMultiTaskDataset(Dataset):
    """
    BDD100K multi-task dataset for:
    - semantic segmentation
    - drivable area detection

    Returns:
        image: torch.FloatTensor of shape [3, H, W]
        seg_mask: torch.LongTensor of shape [H, W]
        drive_mask: torch.LongTensor of shape [H, W]
    """

    def __init__(
        self,
        image_dir: str,
        seg_dir: str,
        drive_dir: str,
        overlap_ids: Optional[Sequence[str]] = None,
        image_size: Tuple[int, int] = (512, 256),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.seg_dir = Path(seg_dir)
        self.drive_dir = Path(drive_dir)
        self.image_size = image_size  # (width, height)

        if overlap_ids is not None:
            self.image_files = [self.image_dir / f"{sample_id}.jpg" for sample_id in overlap_ids]
            self.image_files = [p for p in self.image_files if p.exists()]
        else:
            self.image_files = sorted(list(self.image_dir.glob("*.jpg")))

        if not self.image_files:
            raise ValueError(f"No .jpg files found in {self.image_dir}")

        self.samples = []
        missing = []

        for img_path in self.image_files:
            stem = img_path.stem

            seg_matches = sorted(self.seg_dir.glob(f"{stem}_train_id*.png"))
            drive_matches = sorted(self.drive_dir.glob(f"{stem}_drivable_id*.png"))

            if seg_matches and drive_matches:
                self.samples.append((img_path, seg_matches[0], drive_matches[0]))
            else:
                missing.append(stem)

        if not self.samples:
            raise ValueError("No matched image/segmentation/drivable samples found.")

        print(f"Matched samples: {len(self.samples)}")
        print(f"Missing samples: {len(missing)}")
        if missing:
            print("First 10 missing IDs:", missing[:10])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, seg_path, drive_path = self.samples[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

        seg_mask = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
        drive_mask = cv2.imread(str(drive_path), cv2.IMREAD_UNCHANGED)

        if seg_mask is None:
            raise ValueError(f"Failed to read segmentation mask: {seg_path}")
        if drive_mask is None:
            raise ValueError(f"Failed to read drivable mask: {drive_path}")

        seg_mask = cv2.resize(seg_mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        drive_mask = cv2.resize(drive_mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        if len(seg_mask.shape) == 3:
            seg_mask = seg_mask[:, :, 0]
        if len(drive_mask.shape) == 3:
            drive_mask = drive_mask[:, :, 0]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        seg_mask = torch.tensor(seg_mask, dtype=torch.long)
        drive_mask = torch.tensor(drive_mask, dtype=torch.long)

        return image, seg_mask, drive_mask