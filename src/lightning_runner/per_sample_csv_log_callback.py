from pathlib import Path
from typing import Any

import pandas as pd

import lightning as L
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.functional import confusion_matrix


class PerSampleCsvLogCallback(L.Callback):
    def __init__(self, save_file_path: str, train: bool, validation: bool, test: bool):
        super(PerSampleCsvLogCallback, self).__init__()

        self.log_train = train
        self.log_test = test
        self.log_validation = validation

        self.save_file_path = save_file_path
        Path(save_file_path).parent.mkdir(exist_ok=True, parents=True)

        self.epoch_cumulative_metrics: dict = None
        self.log_dataframe: pd.DataFrame = None

        print(f'PerSampleCsvLogCallback:')
        print(f'Running on: Train: {train}, Validation: {validation}, Test: {test}')
        print(f'Saving logs to: {self.save_file_path}\n')

    def on_fit_start(self, trainer: "Trainer", pl_module: "pl.LightningModule") -> None:
        self._reset_train_data()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_train:
            self._reset_epoch_metrics('train')

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.log_train:
            self._update_epoch_metrics(pl_module, batch, 'train')

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_train:
            self._report_metrics('train', trainer.current_epoch)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_validation:
            self._reset_epoch_metrics('validation')

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.log_validation:
            self._update_epoch_metrics(pl_module, batch, 'validation')

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_validation:
            self._report_metrics('validation', trainer.current_epoch)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_test:
            self._reset_epoch_metrics('test')

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.log_test:
            self._update_epoch_metrics(pl_module, batch, 'test')

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_test:
            self._report_metrics('test', trainer.current_epoch)

    def _reset_epoch_metrics(self, trainer_stage: str):
        self.epoch_cumulative_metrics[trainer_stage] = {'id': [], 'file_path': [], 'original_size': [],
                                                        'label': [], 'logit': torch.Tensor()}

    def _update_epoch_metrics(self, pl_module: LightningModule, batch, trainer_stage: str):
        self.epoch_cumulative_metrics[trainer_stage]['id'] += batch[1]['id']
        self.epoch_cumulative_metrics[trainer_stage]['file_path'] += batch[1]['file_path']
        self.epoch_cumulative_metrics[trainer_stage]['original_size'] += batch[1]['original_size']
        self.epoch_cumulative_metrics[trainer_stage]['label'] += pl_module.current_step_data['labels'].squeeze().int().tolist()
        self.epoch_cumulative_metrics[trainer_stage]['logit'] = torch.concat([self.epoch_cumulative_metrics[trainer_stage]['logit'], pl_module.current_step_data['logits'].squeeze()])

    def _report_metrics(self, trainer_stage: str, epoch: int):
        epoch_log = pd.DataFrame()

        epoch_log['epoch'] = [epoch] * len(self.epoch_cumulative_metrics[trainer_stage]['id'])
        epoch_log['id'] = self.epoch_cumulative_metrics[trainer_stage]['id']
        epoch_log['file_path'] = self.epoch_cumulative_metrics[trainer_stage]['file_path']
        epoch_log['original_size'] = self.epoch_cumulative_metrics[trainer_stage]['original_size']
        epoch_log['label'] = self.epoch_cumulative_metrics[trainer_stage]['label']

        logit = self.epoch_cumulative_metrics[trainer_stage]['logit']
        sigmoid = logit.sigmoid()
        pred = sigmoid.round().int()

        epoch_log['logit'] = logit
        epoch_log['sigmoid'] = sigmoid
        epoch_log['pred'] = pred

        self.log_dataframe = pd.concat([self.log_dataframe, epoch_log], axis=0)
        self._save_log_to_csv()

    def _save_log_to_csv(self):
        self.log_dataframe.to_csv(self.save_file_path, index=False, sep='\t')

    def _reset_train_data(self):
        self.log_dataframe = pd.DataFrame()
        self.epoch_cumulative_metrics = {}

