from pathlib import Path
from typing import Any

import pandas as pd

import lightning as L
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.functional import confusion_matrix


class BinclassLossAccuracyCsvCallback(L.Callback):
    def __init__(self, save_file_path: str, train: bool, validation: bool, test: bool):
        super(BinclassLossAccuracyCsvCallback, self).__init__()

        self.log_train = train
        self.log_test = test
        self.log_validation = validation

        self.save_file_path = save_file_path
        Path(save_file_path).parent.mkdir(exist_ok=True, parents=True)

        self.train_data: pd.DataFrame = None
        self.epoch_cumulative_metrics: dict = {}

        print(f'TrainLogsCallback:')
        print(f'Running on: Train: {train}, Validation: {validation}, Test: {test}')
        print(f'Saving logs to: {self.save_file_path}\n')

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._reset_train_data()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_train:
            self._reset_epoch_metrics('train')

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.log_train:
            self._update_epoch_metrics(pl_module, batch, 'train')

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_train:
            current_epoch = trainer.current_epoch
            self._report_metrics('train', current_epoch, pl_module)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_validation:
            self._reset_epoch_metrics('validation')

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.log_validation:
            self._update_epoch_metrics(pl_module, batch,'validation')

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_validation:
            current_epoch = trainer.current_epoch
            self._report_metrics('validation', current_epoch, pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._reset_train_data()

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_test:
            self._reset_epoch_metrics('test')

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.log_test:
            self._update_epoch_metrics(pl_module,  batch, 'test')

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_test:
            current_epoch = trainer.current_epoch
            self._report_metrics('test', current_epoch, pl_module)

    def _reset_epoch_metrics(self, trainer_stage: str):
        self.epoch_cumulative_metrics[trainer_stage] = {'confusion_matrix': 0, 'loss': 0}

    def _update_epoch_metrics(self, pl_module: LightningModule,  batch, trainer_stage: str):
        labels = batch[1]['label']
        probabilities = pl_module.current_step_outputs['probabilities']
        loss = pl_module.current_step_outputs['loss']
        predictions = probabilities.round()

        self.epoch_cumulative_metrics[trainer_stage]['confusion_matrix'] += confusion_matrix(predictions, labels, task='binary')
        self.epoch_cumulative_metrics[trainer_stage]['loss'] += loss

    def _report_metrics(self, trainer_stage: str, epoch: int, pl_module: LightningModule):
        conf_matrix = self.epoch_cumulative_metrics[trainer_stage]['confusion_matrix']
        loss = self.epoch_cumulative_metrics[trainer_stage]['loss'].item()

        tn = conf_matrix[0][0].item()
        tp = conf_matrix[1][1].item()
        fp = conf_matrix[0][1].item()
        fn = conf_matrix[1][0].item()

        accuracy = (tn + tp) / torch.sum(conf_matrix).item()
        pl_module.log(f'{trainer_stage}_accuracy', accuracy, logger=True)

        conf_matrix_dict = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

        self.train_data.at[epoch, f'{trainer_stage}_accuracy'] = accuracy
        self.train_data.at[epoch, f'{trainer_stage}_loss'] = loss
        self.train_data.at[epoch, f'{trainer_stage}_conf_matrix'] = str(conf_matrix_dict)

        self._save_log_to_csv()

    def _save_log_to_csv(self):
        file_path = str(self.save_file_path) + '.csv'
        self.train_data.to_csv(file_path, index_label='epoch', sep='\t')

    def _reset_train_data(self):
        self.train_data = pd.DataFrame()
