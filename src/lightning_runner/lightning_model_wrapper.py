import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler


class LightningWrapper(L.LightningModule):
    def __init__(self, model, optimizer, criterion):
        super(LightningWrapper, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.current_step_outputs = {}

    def forward(self, inputs):
        logits = self.model(inputs)
        return torch.nn.functional.sigmoid(logits)

    def training_step(self, batch, batch_idx):
        self._calc_step_outputs(batch)

        loss = self.current_step_outputs['loss']
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._calc_step_outputs(batch)

        loss = self.current_step_outputs['loss']
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _calc_step_outputs(self, batch):
        inputs, metadatas = batch
        labels = metadatas['label']

        probabilities = self.forward(inputs)
        loss = self.criterion(probabilities, labels)

        # --- store temp batch data for callbacks to use
        self.current_step_outputs['probabilities'] = probabilities
        self.current_step_outputs['loss'] = loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer


