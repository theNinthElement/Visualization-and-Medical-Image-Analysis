import logging
import typing
import gin
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset

from visualizer.data_loader.data_loader import get_train_valid_test_loader
from visualizer.trainer.trainer import Trainer

LOGGER = logging.getLogger(__name__)


@gin.configurable
def train_model(
    tensorboard_dir: str,
    model_output_dir: str,
    input_height: int,
    input_width: int,
    net: torch.nn.Module,
    data_loaders: typing.Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ],
):
    # load cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device used for training: %s", device.type)

    # tensorboard writer
    tensorboard_writer = (
        SummaryWriter(tensorboard_dir) if tensorboard_dir else SummaryWriter()
    )

    train_loader, valid_loader, test_loader = data_loaders

    net.apply(net.init_weights)
    net.to(device)
    summary(
        net,
        input_size=(4, input_height, input_width),
        batch_size=train_loader.batch_size,
        device=device.type,
    )

    trainer = Trainer(
        net=net,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        summary_writer=tensorboard_writer,
        model_output_path=model_output_dir,
    )

    trainer.train()
    # Task completion information
    LOGGER.info("Train script completed")
