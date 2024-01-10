import json

import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset
from src.datasets.transform_factory import get_train_transform, get_predict_transform
from src.env_constants import *
from src.lightning_runner.loss_accuracy_csv_log_callback import LossAccuracyCsvLogCallback
from src.lightning_runner.lightning_model_wrapper import LightningWrapper
from src.lightning_runner.per_sample_csv_log_callback import PerSampleCsvLogCallback
from src.lightning_runner.printc import printc
from src.lightning_runner.util_functions import save_dict_to_json
from src.models.model_getter import get_resnet18
from src.util_functions import create_output_dir


if __name__ == '__main__':

    # --- Run parameters
    run_params = {
        'run_name': 'yaniv_train',
        'add_timestamp_to_out_dir': False,

        'train_val_split_ratio': 0.8,
        'num_epochs': 10,
        'batch_size': 3,

        'num_workers': 0,
        'short_debug_run': True
    }

    # --- output dir creation
    outputs_path = create_output_dir(PROJECT_PATH, run_params['run_name'], run_params['add_timestamp_to_out_dir'])
    save_dict_to_json(run_params, Path(outputs_path) / Path('run_params.json'))

    # --- Datasets
    print('Creating Datasets')
    train_transform = get_train_transform()
    predict_transform = get_predict_transform()
    train_val_dataset = DogsVsCatsDataset(TRAIN_DATA_PATH, transform=train_transform, cache_data=False, shuffle=True)
    test_dataset = DogsVsCatsDataset(TEST_DATA_PATH, transform=predict_transform, cache_data=False, shuffle=False)

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

    logger = TensorBoardLogger(outputs_path, name='tb_logs')

    train_csv_log_path = str(outputs_path / Path('train_log'))
    test_csv_log_path = str(outputs_path / Path('test_log'))
    per_sample_test_csv_log_path = str(outputs_path / Path('test_predictions_log'))

    checkpoint_path = str(outputs_path / Path('checkpoints'))

    callbacks = [
        LossAccuracyCsvLogCallback(train_csv_log_path, train=True, validation=True, test=False),
        LossAccuracyCsvLogCallback(test_csv_log_path, train=False, validation=False, test=True),
        PerSampleCsvLogCallback(per_sample_test_csv_log_path, train=False, validation=False, test=True),
        ModelCheckpoint(save_top_k=5, dirpath=checkpoint_path, monitor='validation_loss', filename='checkpoint_{epoch:02d}_{validation_loss:.3f}_{validation_accuracy:.4f}'),
        EarlyStopping(monitor="validation_loss", mode="min", patience=5)]

    trainer = L.Trainer(max_epochs=run_params['num_epochs'], logger=logger, callbacks=callbacks, check_val_every_n_epoch=1, num_sanity_val_steps=0)

    trainer.fit(model=l_module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    test_results = trainer.test(model=l_module, dataloaders=test_dataloader)
    save_dict_to_json(test_results, Path(outputs_path) / Path('test_results.json'))


