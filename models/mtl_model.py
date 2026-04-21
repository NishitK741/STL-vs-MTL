import torch.nn as nn
import torchvision.models as models


class MultiTaskModel(nn.Module):
    def __init__(self, seg_classes=19, drive_classes=3):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        self.seg_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, seg_classes, kernel_size=1),
        )

        self.drive_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, drive_classes, kernel_size=1),
        )

        self.upsample = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)

    def forward(self, x):
        feat = self.encoder(x)

        seg_out = self.seg_head(feat)
        drive_out = self.drive_head(feat)

        seg_out = self.upsample(seg_out)
        drive_out = self.upsample(drive_out)

        return seg_out, drive_out