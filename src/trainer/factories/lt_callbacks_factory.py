from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.trainer.trainer_modules.binclass_task.binclass_loss_accuracy_csv_log_callback import BinclassLossAccuracyCsvCallback
from src.trainer.trainer_modules.binclass_task.binclass_per_sample_csv_log_callback import BinclassPerSampleCsvCallback
from src.trainer.trainer_modules.patch_adg_task.patch_loss_csv_log_callback import PatchLossAccuracyCsvCallback
from src.trainer.trainer_modules.patch_adg_task.patch_per_sample_csv_log_callback import PatchPerSampleCsvCallback


def get_callbacks(params: dict, outputs_path: str):

    train_csv_log_path = str(outputs_path / Path('train_log'))
    test_csv_log_path = str(outputs_path / Path('test_log'))
    per_sample_test_csv_log_path = str(outputs_path / Path('test_predictions_log'))
    checkpoint_path = str(outputs_path / Path('checkpoints'))

    if params['task'] == 'bin_class':
        callbacks = [
            BinclassLossAccuracyCsvCallback(train_csv_log_path, train=True, validation=True, test=False),
            BinclassLossAccuracyCsvCallback(test_csv_log_path, train=False, validation=False, test=True),
            BinclassPerSampleCsvCallback(per_sample_test_csv_log_path, train=False, validation=False, test=True),
            ModelCheckpoint(save_top_k=8, dirpath=checkpoint_path, monitor='validation_loss',
                            filename='checkpoint_{epoch:02d}_{step:04d}{validation_loss:.5f}_{validation_accuracy:.5f}'),
            EarlyStopping(monitor="validation_loss", mode="min", patience=params['early_stop_patience'])
        ]

    elif params['task'] == 'patch_adjacence':
        callbacks = [
            PatchLossAccuracyCsvCallback(train_csv_log_path, train=True, validation=True, test=False),
            PatchLossAccuracyCsvCallback(test_csv_log_path, train=False, validation=False, test=True),
            PatchPerSampleCsvCallback(per_sample_test_csv_log_path, train=False, validation=False, test=True),
            ModelCheckpoint(save_top_k=8, dirpath=checkpoint_path, monitor='validation_loss',
                            filename='checkpoint_{epoch:02d}_{step:04d}{validation_loss:.5f}_{validation_accuracy:.5f}'),
            EarlyStopping(monitor="validation_loss", mode="min", patience=params['early_stop_patience'])
        ]

    else:
        raise NotImplementedError(f'Callback factory doesnt have an implementation for task {params["task"]}')

    return callbacks