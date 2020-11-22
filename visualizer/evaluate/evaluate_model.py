import os
import logging
import gin
import typing

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset

from visualizer.data_loader.data_loader import get_train_valid_test_loader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

LOGGER = logging.getLogger(__name__)


@gin.configurable
def evaluate_model(
    model_path: str,
    report_output_path: str,
    criterion: torch.nn,
    net: torch.nn.Module,
    data_loaders: typing.Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ],
    num_classes: int = 4,
    visualize: bool = False,
) -> None:
    """
    This function evaluates the model trained on a set of test image and provides a report with evaluation metrics.
    The evaluation metrics used are: Precision, Recall and F-score.
    The module also aids in visualizing the predictions and groundtruth labels.

    Args:
        model_path: string
        Path of the model to be used for inference
        report_output_path: string
        Path for writing the inference output report with evaluation metrics and visualization images
        criterion: torch.nn
        Loss type for evaluation
        net: torch.nn.Module
        Network architecture of the model trained
        visualize: bool
        To visualize the model predictions alongside groundtruth prediction
    """
    if not os.path.exists(os.path.dirname(report_output_path)):
        LOGGER.info(
            "Output directory does not exist. Creating directory %s",
            os.path.dirname(report_output_path),
        )
        os.makedirs(os.path.dirname(report_output_path))
    if visualize and (
        not os.path.exists(os.path.join(report_output_path, "output_images"))
    ):
        os.makedirs(os.path.join(report_output_path, "output_images"))
        LOGGER.info(
            "Saving images in the directory: %s",
            os.path.join(report_output_path, "output_images"),
        )

    device = torch.device("cpu")
    state_test = torch.load(model_path, map_location=device)
    net.load_state_dict(state_test)
    net.eval()

    # instantiate dataset
    train_loader, valid_loader, test_loader = data_loaders

    LOGGER.info(
        "Evaluating Visualizer on BraTS 2019 images for brain tumor segmentation using the model, %s", model_path
    )
    LOGGER.info("Results will be written to the path, %s", report_output_path)

    LOGGER.info("Ready to start evaluating!")

    df_columns = ["loss", "precision", "recall", "f1-score"]
    df_detailed = pd.DataFrame(columns=df_columns)
    confusion_matrix_array = np.zeros((num_classes, num_classes))
    precision_per_class = np.zeros((num_classes))
    recall_per_class = np.zeros((num_classes))
    f1score_per_class = np.zeros((num_classes))

    for data in test_loader:
        LOGGER.info("Predicting on image: %s", str(len(df_detailed) + 1))

        images = data["image"]
        labels_segmentation = data["label"]

        outputs_segmentation = net(images)

        labels_segmentation = torch.argmax(labels_segmentation, dim=1)
        loss = criterion(outputs_segmentation, labels_segmentation)

        outputs_segmentation = outputs_segmentation.argmax(axis=1)

        if visualize and len(df_detailed) % 20 == 0:
            plt.subplot(121)
            plt.imshow(torch.squeeze(outputs_segmentation).detach().numpy())
            plt.title("Prediction")
            plt.subplot(122)
            plt.imshow(torch.squeeze(labels_segmentation).detach().numpy())
            plt.title("Target label")
            plt.suptitle("Visualizer result")
            LOGGER.info("Saving image number: %s", str(len(df_detailed) + 1))
            plt.savefig(
                report_output_path
                + "/output_images/"
                + str(len(df_detailed) + 1)
                + ".jpg"
            )
            # plt.show()

        outputs_seg_flatten = torch.flatten(outputs_segmentation, start_dim=1)
        labels_seg_flatten = torch.flatten(labels_segmentation, start_dim=1)

        precision, recall, f1score = calculate_metrics(
            target=labels_seg_flatten.detach().numpy(),
            output=outputs_seg_flatten.detach().numpy(),
            include_bg_class=False,
            average="macro",
            num_classes=num_classes,
        )
        df_detailed.loc[len(df_detailed)] = [
            loss.detach().numpy(),
            precision,
            recall,
            f1score,
        ]

        image_precision, image_recall, image_f1score = calculate_metrics(
            target=labels_seg_flatten.detach().numpy(),
            output=outputs_seg_flatten.detach().numpy(),
            include_bg_class=True,
            num_classes=num_classes,
        )
        precision_per_class = precision_per_class + image_precision
        recall_per_class = recall_per_class + image_recall
        f1score_per_class = f1score_per_class + image_f1score

        confusion_matrix_array = confusion_matrix_array + get_confusion_matrix(
            labels_seg_flatten.detach().numpy(), outputs_seg_flatten.detach().numpy(), num_classes,
        )

    df_detailed.loc["mean"] = df_detailed.mean()
    df_normalized_confusion_matrix = pd.DataFrame(
        confusion_matrix_array / len(df_detailed)
    )
    df_precision_per_class = pd.DataFrame(precision_per_class / len(df_detailed))
    df_recall_per_class = pd.DataFrame(recall_per_class / len(df_detailed))
    df_f1score_per_class = pd.DataFrame(f1score_per_class / len(df_detailed))

    excel_writer = pd.ExcelWriter(
        os.path.join(report_output_path, "report.xlsx"), engine="xlsxwriter"
    )
    df_detailed.to_excel(excel_writer, sheet_name="detailed")
    df_normalized_confusion_matrix.to_excel(
        excel_writer, sheet_name="normalized_confusion_matrix"
    )
    df_precision_per_class.to_excel(excel_writer, sheet_name="precision_per_class")
    df_recall_per_class.to_excel(excel_writer, sheet_name="recall_per_class")
    df_f1score_per_class.to_excel(excel_writer, sheet_name="f1score_per_class")
    excel_writer.save()
    LOGGER.info("Results were written to %s", report_output_path)


def calculate_metrics(
    target: np.ndarray,
    output: np.ndarray,
    include_bg_class: bool = False,
    average: str = None,
    num_classes: int = 4,
) -> typing.Tuple:
    """
    Calculates the evaluation metrics precision, recall and f-score for the average
    method passed using sklearn.metrics.precision_recall_fscore_support.

    Args:
        target: np.ndarray
        Flattened label prediction
        output: np.ndarray
        Flattened model prediction
        include_bg_class: bool
        Flag to whether include background class in metric calculation
        average: string
        Average argument for sklearn.metrics.precision_recall_fscore_support

    Returns:
        precision, recall, fscore: tuple
        Evaluation metrics: precision, recall and fscore respectively

    """
    if include_bg_class:
        start_label = 0
    else:
        start_label = 1
    metrics = precision_recall_fscore_support(
        target[0], output[0], labels=list(range(start_label, num_classes)), average=average
    )

    return metrics[0], metrics[1], metrics[2]


def get_confusion_matrix(target: np.ndarray,
                         output: np.ndarray,
                         num_classes: int = 4,
                         ) -> np.ndarray:
    """
    Calculates the confusion matrix normalized for each image with regard to all the classes
    method passed using sklearn.metrics.confusion_matrix.

    Args:
        target: np.ndarray
        Flattened label prediction
        output: np.ndarray
        Flattened model prediction

    Returns:
        confusion_matrix_: np.ndarray
        Confusion matrix for all the classes

    """

    confusion_matrix_ = confusion_matrix(
        target[0], output[0], labels=list(range(0, num_classes)), normalize="all"
    )
    return confusion_matrix_