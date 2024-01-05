import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics.functional import confusion_matrix


class LightningWrapper(L.LightningModule):
    def __init__(self, model, optimizer, criterion):
        super(LightningWrapper, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def forward(self, inputs, target):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch, collect_metrics=False)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch, collect_metrics=True)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self._report_metrics()

    def _calc_batch_loss(self, batch, collect_metrics: bool=False):
        inputs, labels = batch
        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        if collect_metrics:
            self._update_epoch_metrics(labels, logits)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer

    def on_validation_start(self) -> None:
        self._reset_epoch_metrics()

    def on_train_start(self) -> None:
        self._reset_epoch_metrics()

    def _reset_epoch_metrics(self):
        self.epoch_metrics_data = {
            'confusion_matrix': 0,
        }

    def _update_epoch_metrics(self, labels, logits):
        preds = torch.nn.functional.sigmoid(logits).round()
        self.epoch_metrics_data['confusion_matrix'] += confusion_matrix(preds, labels, task='binary')

    def _report_metrics(self):
        if 'confusion_matrix' in self.epoch_metrics_data:
            conf_matrix = self.epoch_metrics_data['confusion_matrix']
            tn = conf_matrix[0][0]
            tp = conf_matrix[1][1]
            fp = conf_matrix[0][1]
            fn = conf_matrix[1][0]
            accuracy = (tn + tp) / torch.sum(conf_matrix)
            self.log('accuracy', accuracy, logger=True)


