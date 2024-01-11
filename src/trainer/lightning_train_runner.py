import argparse
from pathlib import Path
from typing import Union

import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset
from src.datasets.transform_factory import get_train_transform, get_predict_transform
from src.trainer.loss_accuracy_csv_log_callback import LossAccuracyCsvLogCallback
from src.trainer.lightning_model_wrapper import LightningWrapper
from src.trainer.per_sample_csv_log_callback import PerSampleCsvLogCallback
from src.trainer.printc import printc
from src.trainer.util_functions import save_dict_to_json, load_dict_from_json
from src.models.model_getter import get_resnet18
from src.trainer.util_functions import create_output_dir


def run_train_test(run_params: dict, project_path: Union[str, Path], train_data_path: Union[str, Path], test_data_path: Union[str, Path]) -> None:
    # --- output dir creation
    outputs_path = create_output_dir(project_path, run_params['run_name'], run_params['add_timestamp_to_out_dir'])
    save_dict_to_json(run_params, Path(outputs_path) / Path('run_params.json'))

    # --- Datasets
    print('Creating Datasets')
    train_transform = get_train_transform({'resize_0': 224, 'resize_1': 224})
    predict_transform = get_predict_transform({'resize_0': 224, 'resize_1': 224})
    train_val_dataset = DogsVsCatsDataset(train_data_path, transform=train_transform, cache_data=False, shuffle=True)
    test_dataset = DogsVsCatsDataset(test_data_path, transform=predict_transform, cache_data=False, shuffle=False)

    # --- Split train and validation
    train_val_split_ratio = 0.8
    train_split_th = int(np.floor(train_val_split_ratio * len(train_val_dataset)))
    train_dataset, valid_dataset = random_split(train_val_dataset, [train_split_th, len(train_val_dataset) - train_split_th])

    # --- Debug option for very short run (5 batches in each dataloader)
    if run_params['short_debug_run']:
        printc.yellow('------------------------------------')
        printc.yellow('Warning - This is a short debug run!')
        printc.yellow('------------------------------------')
        train_dataset, valid_dataset, _ = random_split(train_val_dataset, [5, 5, len(train_val_dataset) - 10])
        test_dataset, _ = random_split(test_dataset, [10, len(test_dataset) - 10])

    # --- Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=run_params['batch_size'], num_workers=run_params['num_workers'])
    valid_dataloader = DataLoader(valid_dataset, batch_size=run_params['batch_size'], num_workers=run_params['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=run_params['batch_size'], num_workers=run_params['num_workers'])

    # --- model, optimizer, loss
    model = get_resnet18(pretrained=True, out_features=1)
    optimizer = Adam(model.parameters())
    criterion = BCELoss()

    # --- Lightning wrapper module and callbacks
    l_module = LightningWrapper(model, optimizer, criterion)

    # --- Trainer inputs
    trainer_args = dict()

    # --- Logger
    trainer_args['logger'] = TensorBoardLogger(outputs_path, name='tb_logs')

    # --- Callbacks
    train_csv_log_path = str(outputs_path / Path('train_log'))
    test_csv_log_path = str(outputs_path / Path('test_log'))
    per_sample_test_csv_log_path = str(outputs_path / Path('test_predictions_log'))

    checkpoint_path = str(outputs_path / Path('checkpoints'))

    trainer_args['callbacks'] = [
        LossAccuracyCsvLogCallback(train_csv_log_path, train=True, validation=True, test=False),
        LossAccuracyCsvLogCallback(test_csv_log_path, train=False, validation=False, test=True),
        PerSampleCsvLogCallback(per_sample_test_csv_log_path, train=False, validation=False, test=True),
        ModelCheckpoint(save_top_k=5, dirpath=checkpoint_path, monitor='validation_loss', filename='checkpoint_{epoch:02d}_{step:04d}{validation_loss:.5f}_{validation_accuracy:.5f}'),
        EarlyStopping(monitor="validation_loss", mode="min", patience=5)
    ]

    # --- Others
    trainer_args['max_epochs'] = run_params['max_epochs']
    trainer_args['check_val_every_n_epoch'] = run_params['check_val_every_n_epoch']
    trainer_args['num_sanity_val_steps'] = 1

    # --- Vamos
    trainer = L.Trainer(**trainer_args)
    trainer.fit(model=l_module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    test_results = trainer.test(model=l_module, dataloaders=test_dataloader)
    save_dict_to_json(test_results, Path(outputs_path) / Path('test_results.json'))


