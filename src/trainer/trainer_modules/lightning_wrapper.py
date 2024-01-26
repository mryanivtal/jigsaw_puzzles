from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT

from src.util_functions.printc import printc
from src.util_functions.util_functions import is_debug_mode


class LightningWrapper(L.LightningModule):
    def __init__(self, model, optimizer, criterion, inference_normalizer):
        super(LightningWrapper, self).__init__()

        if not is_debug_mode():
            self.save_hyperparameters(ignore=['model', 'criterion'])
        else:
            printc.yellow('===== Running in debug mode! model saving may be partial! =======')

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.current_step_outputs = {}
        self.output_normalizer = inference_normalizer

    def forward(self, inputs):
        output = self.model(inputs)

        if self.output_normalizer:
            output = self.output_normalizer(output)

        return output

    def training_step(self, batch, batch_idx):
        self._calc_step_outputs(batch)

        loss = self.current_step_outputs['loss']
        self.log("train_loss", loss, batch_size=len(batch[0]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._calc_step_outputs(batch)

        loss = self.current_step_outputs['loss']
        self.log("validation_loss", loss, batch_size = len(batch[0]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        self._calc_step_outputs(batch)

        loss = self.current_step_outputs['loss']
        self.log("test_loss", loss, batch_size = len(batch[0]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _calc_step_outputs(self, batch):
        inputs, metadatas = batch
        target = metadatas['target']

        probabilities = self(inputs)
        loss = self.criterion(probabilities, target)

        # --- store temp batch data for callbacks to use
        self.current_step_outputs['probabilities'] = probabilities
        self.current_step_outputs['loss'] = loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        inputs, _ = batch
        return self(inputs)



