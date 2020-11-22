import torch
import gin
import typing
from collections import Counter


@gin.configurable
def calculate_weight(
    data_loaders: typing.Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ],
    num_classes: int,
) -> torch.Tensor:
    """

    Args:
        data_loaders: typing.Tuple[torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader
        ]
            Train dataloader, validation dataloader, test dataloader

        num_classes: int
            Number of classes in the dataset

    Returns:
        segmentation_class_weights: torch.Tensor
            Weights for each class for CE loss weighting as per the chargrid paper and
            https://arxiv.org/pdf/1606.02147.pdf (Section: 5.2) for balancing imbalanced training in segmentation

    """
    train_loader = data_loaders[0]
    c_segmentation_weights = 1.04
    c_segmentation_weights = torch.scalar_tensor(c_segmentation_weights)
    segmentation_class_counts = Counter()

    # To definitely include all classes atleast once
    labels = range(0, num_classes)
    segmentation_class_counts.update(labels)

    for sample in train_loader:
        labels = sample["label"]
        labels = torch.argmax(
            labels, dim=1
        )  # dim = [bs, channels, spatial_x, spatial_y]
        segmentation_class_counts.update(labels.flatten().tolist())
    total_count_segmentation_labels = max(sum(segmentation_class_counts), 1)
    segmentation_class_weights = torch.tensor(
        [
            1
            / torch.log(
                c_segmentation_weights
                + segmentation_class_counts[label] / total_count_segmentation_labels
            )
            for label in sorted(segmentation_class_counts.keys())
        ]
    )
    sum_seg_class_weights = torch.sum(segmentation_class_weights)
    segmentation_class_weights = torch.div(
        segmentation_class_weights, sum_seg_class_weights
    )

    return segmentation_class_weights
