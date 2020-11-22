import os
import timeit
import logging
import gin
from sys import maxsize

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR


LOGGER = logging.getLogger(__name__)


@gin.configurable
class Trainer:
    """
    Trainer class. This class is used to define all the training parameters and processes for training the network.
    """

    def __init__(
        self,
        net,
        segment_criterion,
        optimizer_class,
        train_loader,
        valid_loader,
        model_output_path,
        input_height,
        input_width,
        lr_step_size=5,
        lr=1e-04,
        patience=5,
        weight_decay=0,
        resume=False,
        num_epochs=50,
        scheduler=None,
        summary_writer=SummaryWriter(),
        device="cpu",
    ):
        self.net = net
        self.criterion_segmentation = segment_criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model_output_path = model_output_path
        self.lr = lr
        self.patience = patience
        self.weight_decay = weight_decay
        self.resume = resume
        self.num_epochs = num_epochs
        self.optimizer = optimizer_class(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=lr_step_size, gamma=0.1)
        self.tensorboard_writer = summary_writer
        self.device = device
        self.loss_scale = 1.0
        self.input_height = input_height
        self.input_width = input_width

    def _sample_to_device(self, sample):
        device_sample = {}
        for key, value in sample.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    device_sample[key] = value.to(self.device)
                elif isinstance(value, list):
                    device_sample[key] = [
                        t.to(self.device) if isinstance(t, torch.Tensor) else t
                        for t in value
                    ]
                else:
                    device_sample[key] = value

        return device_sample

    def train(self):

        LOGGER.info("Train Module")
        if not os.path.exists(os.path.dirname(self.model_output_path)):
            LOGGER.info(
                "Output directory does not exist. Creating directory %s",
                os.path.dirname(self.model_output_path),
            )
            os.makedirs(os.path.dirname(self.model_output_path))
        model_path = os.path.join(self.model_output_path, "model.pth")
        best_model_path = model_path
        patience_count = self.patience
        train_len = len(self.train_loader.batch_sampler)

        LOGGER.info("Ready to start training")
        tic = timeit.default_timer()
        best_validation_loss = maxsize

        for epoch in range(self.num_epochs):
            sample_size = 10
            self.net.train(True)

            running_loss = 0.0
            av_loss = 0.0

            for batch, data in enumerate(self.train_loader):
                LOGGER.info("Training: batch %d of epoch %d", batch + 1, epoch + 1)
                data = self._sample_to_device(data)

                input_image = data["image"]
                target = data["label"]
                target = torch.argmax(target, dim=1)

                self.optimizer.zero_grad()

                outputs_segmentation = self.net(input_image)

                if self.device.type == "cuda":
                    self.criterion_segmentation.cuda()

                loss = self.criterion_segmentation(outputs_segmentation, target)

                loss.backward()

                self.optimizer.step()

                av_loss += loss.item() / self.loss_scale
                running_loss += loss.item() / self.loss_scale

                if batch % sample_size == (sample_size - 1):
                    LOGGER.info(
                        "epoch: %d, step: %d, loss: %f",
                        epoch + 1,
                        batch + 1,
                        running_loss / sample_size,
                    )
                    running_loss = 0.0

            if self.scheduler:
                self.scheduler.step()

            # output training loss
            av_train_loss = av_loss / train_len

            LOGGER.info(
                "epoch: %d, average loss: %f",
                epoch + 1,
                av_train_loss,
            )

            av_valid_loss = self.validation()

            LOGGER.info(
                "epoch: %d, average validation loss: %f",
                epoch + 1,
                av_valid_loss,
            )
            self.tensorboard_writer.add_scalar(
                "training/average_train_loss", av_train_loss, epoch + 1
            )
            self.tensorboard_writer.add_scalar(
                "validation/average_valid_loss", av_valid_loss, epoch + 1
            )

            if av_valid_loss < best_validation_loss and model_path and best_model_path:
                best_validation_loss = av_valid_loss
                best_model_path = model_path
                LOGGER.info(
                    "Better model found. Saving. Epoch %d, Path %s",
                    epoch + 1,
                    best_model_path,
                )
                torch.save(self.net.state_dict(), best_model_path)
                patience_count = self.patience
            else:
                patience_count -= 1
                LOGGER.info(
                    "No better model found. Epoch %d, Patience left %d",
                    epoch + 1,
                    patience_count,
                )

            if patience_count == 0:
                LOGGER.info(
                    "Epoch %d Patience is 0. Early stopping triggered", epoch + 1
                )
                break

        toc = timeit.default_timer()
        LOGGER.info("Finished training in %f s", toc - tic)

        self.tensorboard_writer.close()

    def validation(self):
        LOGGER.info("Validation Module")
        valid_len = len(self.valid_loader.batch_sampler)

        self.net.train(False)

        av_loss = 0.0

        for batch, data in enumerate(self.valid_loader):
            data = self._sample_to_device(data)

            input_image = data["image"]
            target = data["label"]
            target = torch.argmax(target, dim=1)

            self.optimizer.zero_grad()

            outputs_segmentation = self.net(input_image)
            loss = self.criterion_segmentation(outputs_segmentation, target)

            av_loss += loss.item() / self.loss_scale

        av_valid_loss = av_loss / valid_len
        return av_valid_loss
