from pathlib import Path
from typing import Union

import lightning as L

from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.trainer.factories.criterion_factory import get_criterion
from src.trainer.factories.dataset_factory import get_datasets
from src.trainer.factories.lt_callbacks_factory import get_callbacks
from src.trainer.factories.optimizer_factory import get_optimizer
from src.trainer.factories.sample_saver_factory import get_sample_saver
from src.trainer.trainer_modules.lightning_wrapper import LightningWrapper
from src.trainer.factories.model_factory import get_model, get_inference_normalizer
from src.util_functions.printc import printc
from src.util_functions.util_functions import create_output_dir, save_dict_to_json


def execute_train_flow(run_params: dict, project_path: Union[str, Path], train_data_path: Union[str, Path], test_data_path: Union[str, Path], stop_before_fit = False) -> None:
    # --- Handle run param dicts
    trainer_params = run_params['trainer']
    dataset_params = run_params['dataset']
    model_params = run_params['model']
    loss_params = run_params['loss']
    sample_saver_params = run_params['sample_saver']
    flow_params = run_params['flow']
    optimizer_params = run_params['optimizer']

    # --- Flow params
    run_train = flow_params['run_train']
    run_test = flow_params['run_test']
    start_from_lt_checkpoint = flow_params.get('start_from_lt_checkpoint', None)

    # --- output dir creation
    outputs_path = create_output_dir(project_path, trainer_params['run_name'], trainer_params['add_timestamp_to_out_dir'])
    save_dict_to_json(run_params, Path(outputs_path) / Path('experiment_params.json'))

    # --- Datasets
    print('Creating Datasets')
    train_dataset, valid_dataset, test_dataset = get_datasets(dataset_params, train_data_path, test_data_path)

    # --- Save samples to folder
    sample_saver = get_sample_saver(sample_saver_params)
    sample_saver.save_samples(outputs_path, train_dataset, valid_dataset, test_dataset)

    # --- Dataloaders
    persistent_workers = True if trainer_params['num_workers'] > 0 else False
    train_dataloader = DataLoader(train_dataset, batch_size=trainer_params['batch_size'], num_workers=trainer_params['num_workers'], persistent_workers=persistent_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=trainer_params['batch_size'], num_workers=trainer_params['num_workers'], persistent_workers=persistent_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=trainer_params['batch_size'], num_workers=trainer_params['num_workers'], persistent_workers=persistent_workers)

    # --- model, optimizer, loss
    model = get_model(model_params)
    criterion = get_criterion(loss_params)
    optimizer = get_optimizer(optimizer_params, model)
    inference_normalizer = get_inference_normalizer(model_params)

    # --- Lightning wrapper module and callbacks
    if not start_from_lt_checkpoint:
        l_module = LightningWrapper(model, optimizer, criterion, inference_normalizer)
    else:
        # TODO:Yaniv: may need to load model weights separately because not in hyperparameters. need to understand
        l_module = LightningWrapper.load_from_checkpoint(start_from_lt_checkpoint, model=model, optimizer=optimizer, criterion=criterion, inference_normalizer=inference_normalizer)

    # --- Trainer inputs
    trainer_args = dict()

    # --- Logger
    trainer_args['logger'] = TensorBoardLogger(outputs_path, name='tb_logs')

    # --- Callbacks
    trainer_args['callbacks'] = get_callbacks(trainer_params, outputs_path)

    # --- Others
    trainer_args['max_epochs'] = trainer_params['max_epochs']
    trainer_args['check_val_every_n_epoch'] = trainer_params['check_val_every_n_epoch']
    trainer_args['num_sanity_val_steps'] = 1

    trainer = L.Trainer(**trainer_args)

    # --- Vamos
    if stop_before_fit:
        printc.cyan('**** Stopped before actual run ****')
        return

    if run_train:
        printc.blue('Starting train ')
        trainer.fit(model=l_module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    if run_test:
        printc.blue('Starting test ')
        test_results = trainer.test(model=l_module, dataloaders=test_dataloader)
        save_dict_to_json(test_results, Path(outputs_path) / Path('test_results.json'))









