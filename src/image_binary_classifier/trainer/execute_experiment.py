from pathlib import Path
from typing import Union

import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.image_binary_classifier.trainer.factories.criterion_factory import get_criterion
from src.image_binary_classifier.trainer.factories.dataset_factory import get_datasets
from src.image_binary_classifier.trainer.lightning_modules.loss_accuracy_csv_log_callback import LossAccuracyCsvLogCallback
from src.image_binary_classifier.trainer.lightning_modules.binary_classifier_lt_wrapper import BinaryClassifierLtWrapper
from src.image_binary_classifier.trainer.lightning_modules.per_sample_csv_log_callback import PerSampleCsvLogCallback
from src.image_binary_classifier.trainer.factories.model_factory import get_model
from src.util_functions.printc import printc
from src.util_functions.sample_saver import save_samples_to_output_dir
from src.util_functions.util_functions import create_output_dir, save_dict_to_json


def execute_experiment(run_params: dict, project_path: Union[str, Path], train_data_path: Union[str, Path], test_data_path: Union[str, Path], stop_before_fit = False) -> None:

    # --- Handle run param dicts
    trainer_params = run_params['trainer']
    dataset_params = run_params['dataset']
    model_params = run_params['model']
    loss_params = run_params['loss']

    # --- output dir creation
    outputs_path = create_output_dir(project_path, trainer_params['run_name'], trainer_params['add_timestamp_to_out_dir'])
    save_dict_to_json(run_params, Path(outputs_path) / Path('experiment_params.json'))

    # --- Datasets
    print('Creating Datasets')
    train_dataset, valid_dataset, test_dataset = get_datasets(dataset_params, train_data_path, test_data_path)

    # --- Save samples to folder
    num_samples_to_save = trainer_params['num_samples_to_save']
    if num_samples_to_save > 0:
        save_samples_to_output_dir(outputs_path, num_samples_to_save, train_dataset, valid_dataset, test_dataset)

    # --- Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=trainer_params['batch_size'], num_workers=trainer_params['num_workers'])
    valid_dataloader = DataLoader(valid_dataset, batch_size=trainer_params['batch_size'], num_workers=trainer_params['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=trainer_params['batch_size'], num_workers=trainer_params['num_workers'])

    # --- model, optimizer, loss
    model = get_model(model_params)
    criterion = get_criterion(loss_params)
    optimizer = Adam(model.parameters())

    # --- Lightning wrapper module and callbacks
    l_module = BinaryClassifierLtWrapper(model, optimizer, criterion)

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
        ModelCheckpoint(save_top_k=8, dirpath=checkpoint_path, monitor='validation_loss', filename='checkpoint_{epoch:02d}_{step:04d}{validation_loss:.5f}_{validation_accuracy:.5f}'),
        EarlyStopping(monitor="validation_loss", mode="min", patience=trainer_params['early_stop_patience'])
    ]

    # --- Others
    trainer_args['max_epochs'] = trainer_params['max_epochs']
    trainer_args['check_val_every_n_epoch'] = trainer_params['check_val_every_n_epoch']
    trainer_args['num_sanity_val_steps'] = 1

    # --- Vamos
    if stop_before_fit:
        printc.cyan('**** Stopped before actual run ****')
        return

    trainer = L.Trainer(**trainer_args)
    trainer.fit(model=l_module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    test_results = trainer.test(model=l_module, dataloaders=test_dataloader)
    save_dict_to_json(test_results, Path(outputs_path) / Path('test_results.json'))









