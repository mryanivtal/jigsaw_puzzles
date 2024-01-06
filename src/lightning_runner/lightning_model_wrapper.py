import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRScheduler


class LightningWrapper(L.LightningModule):
    def __init__(self, model, optimizer, criterion):
        super(LightningWrapper, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.current_step_data = {}

    def forward(self, inputs, target):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _calc_batch_loss(self, batch):
        inputs, labels = batch
        logits = self.model(inputs)
        loss = self.criterion(logits, labels)

        # --- store temp batch data for callbacks to use
        self.current_step_data['inputs'] = inputs
        self.current_step_data['labels'] = labels
        self.current_step_data['logits'] = logits
        self.current_step_data['loss'] = loss

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer


