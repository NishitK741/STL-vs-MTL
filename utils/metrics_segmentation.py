import torch


def update_segmentation_confusion_matrix(conf_matrix, preds, targets, num_classes, ignore_index=255):
    """
    Update confusion matrix for semantic segmentation.

    preds:   [B, H, W]
    targets: [B, H, W]
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    valid_mask = targets != ignore_index
    preds = preds[valid_mask]
    targets = targets[valid_mask]

    indices = targets * num_classes + preds
    bincount = torch.bincount(indices, minlength=num_classes * num_classes)

    conf_matrix += bincount.reshape(num_classes, num_classes)


def compute_iou_from_confusion_matrix(conf_matrix):
    """
    Returns:
        iou_per_class: tensor [num_classes]
        miou: float
    """
    conf_matrix = conf_matrix.float()

    true_positives = torch.diag(conf_matrix)
    false_positives = conf_matrix.sum(dim=0) - true_positives
    false_negatives = conf_matrix.sum(dim=1) - true_positives

    denominator = true_positives + false_positives + false_negatives
    iou_per_class = true_positives / torch.clamp(denominator, min=1.0)

    valid_classes = denominator > 0
    if valid_classes.sum() > 0:
        miou = iou_per_class[valid_classes].mean().item()
    else:
        miou = 0.0

    return iou_per_class, miou