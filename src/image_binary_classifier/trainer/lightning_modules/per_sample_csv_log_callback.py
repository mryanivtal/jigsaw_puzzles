from pathlib import Path
from typing import Any

import pandas as pd

import lightning as L
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT


class PerSampleCsvLogCallback(L.Callback):
    def __init__(self, save_file_path: str, train: bool=False, validation: bool=False, test: bool=True, predict: bool=True, file_per_epoch: bool=True):
        super(PerSampleCsvLogCallback, self).__init__()

        self.log_train = train
        self.log_test = test
        self.log_validation = validation
        self.file_per_epoch = file_per_epoch

        self.save_file_path = save_file_path
        Path(save_file_path).parent.mkdir(exist_ok=True, parents=True)

        self.epoch_cumulative_metrics: dict = None
        self.log_dataframes: dict = None

        self._reset_train_data()

        print(f'PerSampleCsvLogCallback:')
        print(f'Running on: Train: {train}, Validation: {validation}, Test: {test}')
        print(f'Saving logs to: {self.save_file_path}\n')


    # ------- Train
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
            self._report_metrics('train', trainer.current_epoch)

    # ------- Validation

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_validation:
            self._reset_epoch_metrics('validation')

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.log_validation:
            self._update_epoch_metrics(pl_module, batch, 'validation')

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_validation:
            self._report_metrics('validation', trainer.current_epoch)

    # ------- Test
    def on_tes_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._reset_train_data()

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_test:
            self._reset_epoch_metrics('test')

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.log_test:
            self._update_epoch_metrics(pl_module, batch, 'test')

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_test:
            self._report_metrics('test', trainer.current_epoch)

    # ------- Predict
    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._reset_train_data()

    def on_predict_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_predict:
            self._reset_epoch_metrics('predict')

    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.log_predict:
            self._update_epoch_metrics(pl_module, batch, 'predict')

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_predict:
            self._report_metrics('predict', trainer.current_epoch)

    def _reset_epoch_metrics(self, trainer_stage: str):
        self.epoch_cumulative_metrics[trainer_stage] = {'image_metadata': [],
                                                        'label': [], 'probability': torch.Tensor()}
        if self.file_per_epoch:
            self.log_dataframes[trainer_stage] = pd.DataFrame()

    def _update_epoch_metrics(self, pl_module: LightningModule, batch, trainer_stage: str):
        label = batch[1]['label'].squeeze(axis=1).int().tolist() if 'label' in batch[1] else None

        self.epoch_cumulative_metrics[trainer_stage]['image_metadata'] += batch[1]['image_metadata']
        self.epoch_cumulative_metrics[trainer_stage]['probability'] = torch.concat([self.epoch_cumulative_metrics[trainer_stage]['probability'], pl_module.current_step_outputs['probabilities'].squeeze(axis=1).cpu()])
        self.epoch_cumulative_metrics[trainer_stage]['label'] += label

    def _report_metrics(self, trainer_stage: str, epoch: int):
        epoch_log = pd.DataFrame()

        epoch_log['epoch'] = [epoch] * len(self.epoch_cumulative_metrics[trainer_stage]['label'])
        epoch_log['metadata'] = self.epoch_cumulative_metrics[trainer_stage]['image_metadata']
        epoch_log['label'] = self.epoch_cumulative_metrics[trainer_stage]['label']

        probabilities = self.epoch_cumulative_metrics[trainer_stage]['probability']
        predictions = probabilities.round().int()

        epoch_log['probability'] = probabilities.tolist()
        epoch_log['prediction'] = predictions.tolist()

        self.log_dataframes[trainer_stage] = pd.concat([self.log_dataframes[trainer_stage], epoch_log], axis=0)
        self._save_log_to_csv(trainer_stage, epoch)

    def _save_log_to_csv(self, trainer_stage: str, epoch: int):
        if self.file_per_epoch:
            file_path = str(self.save_file_path) + f'_{trainer_stage}_{epoch}.csv'
            self.log_dataframes[trainer_stage].to_csv(file_path, index=False, sep='\t')

        else:
            file_path = str(self.save_file_path) + f'_{trainer_stage}.csv'
            self.log_dataframes[trainer_stage].to_csv(file_path, index=False, sep='\t')

    def _reset_train_data(self):
        self.log_dataframes = {
            'train': pd.DataFrame(),
            'validation': pd.DataFrame(),
            'test': pd.DataFrame()
        }
        self.epoch_cumulative_metrics = {}
