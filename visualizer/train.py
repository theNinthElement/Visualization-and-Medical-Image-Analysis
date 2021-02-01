import logging
import click
import gin
import torch

from visualizer.utils import setup_logging
from visualizer.train_manager.train_model import train_model

LOGGER = logging.getLogger(__name__)


def _register_configurables():
    # Optimizer
    gin.external_configurable(torch.optim.Adam, module="torch.optim")
    gin.external_configurable(torch.optim.SGD, module="torch.optim")
    # Loss functions
    gin.external_configurable(torch.nn.CrossEntropyLoss, module="torch.nn")


@click.command()
@click.option("--config", "-c", help="config file for experimentation parameters")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]),
)
@click.option("--log-dir", default="")
@click.option("--tensorboard-dir", default="")
@click.option("--model-output-dir", default="model/")
def train(config, log_level, log_dir, tensorboard_dir, model_output_dir):
    setup_logging(log_level=log_level, log_dir=log_dir)
    _register_configurables()
    gin.parse_config_file(config)
    LOGGER.info(">training model")
    train_model(tensorboard_dir, model_output_dir)


if __name__ == "__main__":
    train()
