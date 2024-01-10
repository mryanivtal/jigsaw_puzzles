import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset
from src.datasets.transform_factory import get_train_transform
from src.env_constants import *
from src.lightning_runner.loss_accuracy_csv_log_callback import LossAccuracyCsvLogCallback
from src.lightning_runner.lightning_model_wrapper import LightningWrapper
from src.lightning_runner.per_sample_csv_log_callback import PerSampleCsvLogCallback
from src.models.model_getter import get_resnet18
from src.util_functions import create_output_dir


if __name__ == '__main__':

    # --- Run parameters
    RUN_NAME = 'yaniv_train'
    ADD_TIMESTAMP_TO_OUT_DIR = False

    NUM_EPOCHS = 50
    BATCH_SIZE = 3
    NUM_WORKERS = 0

    VERY_SHORT_DEBUG_RUN = True     #TODO: change in production!

    # --- output dir creation
    outputs_path = create_output_dir(PROJECT_PATH, RUN_NAME, ADD_TIMESTAMP_TO_OUT_DIR)

    # --- Datasets
    train_transform = get_train_transform()
    train_val_dataset = DogsVsCatsDataset(TRAIN_DATA_PATH, transform=train_transform, cache_data=False, shuffle=True)

    # --- Split train and validation
    train_split_ratio = 0.75
    train_split_th = int(np.floor(train_split_ratio * len(train_val_dataset)))
    train_dataset, valid_dataset = random_split(train_val_dataset, [train_split_th, len(train_val_dataset) - train_split_th])

    # --- Debug option for very short run (5 batches in each dataloader)
    if VERY_SHORT_DEBUG_RUN:
        train_dataset, valid_dataset, _ = random_split(train_val_dataset,
                                                    [5, 5, len(train_val_dataset) - 10])

    # --- Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # --- model, optimizer, loss
    model = get_resnet18(pretrained=False, out_features=1)
    optimizer = Adam(model.parameters())
    criterion = BCELoss()

    # --- Lightning wrapper module and callbacks
    l_module = LightningWrapper(model, optimizer, criterion)

    logger = TensorBoardLogger(outputs_path, name='tb_logs')

    train_csv_log_path = str(outputs_path / Path('train_log'))
    test_csv_log_path = str(outputs_path / Path('test_log'))
    per_sample_validation_csv_log_path = str(outputs_path / Path('sample_predictions_log'))

    checkpoint_path = str(outputs_path / Path('checkpoints'))

    callbacks = [
        LossAccuracyCsvLogCallback(train_csv_log_path, train=True, validation=True, test=False),   # train and validation logs
        LossAccuracyCsvLogCallback(test_csv_log_path, train=False, validation=False, test=True),  # test logs
        PerSampleCsvLogCallback(per_sample_validation_csv_log_path, train=False, validation=True, test=False),
        ModelCheckpoint(save_top_k=5, dirpath=checkpoint_path, monitor='validation_loss', filename='checkpoint_{epoch:02d}_{validation_loss:.3f}_{validation_accuracy:.4f}')]   # Model checkpoints

    trainer = L.Trainer(max_epochs=NUM_EPOCHS, logger=logger, callbacks=callbacks, check_val_every_n_epoch=1, num_sanity_val_steps=0)
    trainer.fit(model=l_module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


