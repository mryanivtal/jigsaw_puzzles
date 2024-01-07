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
        raise NotImplementedError('Not implemented yet!')
        # TODO:Yaniv: implement

        super(PerSampleCsvLogCallback, self).__init__()

        self.log_train = train
        self.log_test = test
        self.log_validation = validation

        self.save_file_path = save_file_path
        Path(save_file_path).parent.mkdir(exist_ok=True, parents=True)

        self.log_data: pd.DataFrame = None

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
            self._update_epoch_metrics(pl_module, 'train')

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_train:
            current_epoch = trainer.current_epoch
            self._report_metrics('train', current_epoch)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_validation:
            self._reset_epoch_metrics('validation')

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.log_validation:
            self._update_epoch_metrics(pl_module, 'validation')

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_validation:
            current_epoch = trainer.current_epoch
            self._report_metrics('validation', current_epoch)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_test:
            self._reset_epoch_metrics('test')

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.log_test:
            self._update_epoch_metrics(pl_module, 'test')

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.log_test:
            current_epoch = trainer.current_epoch
            self._report_metrics('test', current_epoch)

    def _reset_epoch_metrics(self, trainer_stage: str):
        self.epoch_cumulative_metrics[trainer_stage] = {'confusion_matrix': 0, 'loss': 0}

    def _update_epoch_metrics(self, pl_module: LightningModule, trainer_stage: str):
        labels = pl_module.current_step_data['labels']
        logits = pl_module.current_step_data['logits']
        loss = pl_module.current_step_data['loss']

        preds = torch.nn.functional.sigmoid(logits).round()

        self.epoch_cumulative_metrics[trainer_stage]['confusion_matrix'] += confusion_matrix(preds, labels, task='binary')
        self.epoch_cumulative_metrics[trainer_stage]['loss'] += loss

    def _report_metrics(self, trainer_stage: str, epoch: int):
        conf_matrix = self.epoch_cumulative_metrics[trainer_stage]['confusion_matrix']
        loss = self.epoch_cumulative_metrics[trainer_stage]['loss'].item()

        tn = conf_matrix[0][0].item()
        tp = conf_matrix[1][1].item()
        fp = conf_matrix[0][1].item()
        fn = conf_matrix[1][0].item()

        accuracy = (tn + tp) / torch.sum(conf_matrix).item()

        conf_matrix_dict = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

        self.log_data.at[epoch, f'{trainer_stage}_accuracy'] = accuracy
        self.log_data.at[epoch, f'{trainer_stage}_loss'] = loss
        self.log_data.at[epoch, f'{trainer_stage}_conf_matrix'] = str(conf_matrix_dict)

        self._save_log_to_csv()

    def _save_log_to_csv(self):
        self.log_data.to_csv(self.save_file_path, index_label='epoch', sep='\t')

    def _reset_train_data(self):
        self.log_data = pd.DataFrame()
