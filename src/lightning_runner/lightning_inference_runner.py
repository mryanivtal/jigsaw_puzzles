import numpy as np
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset
from src.datasets.transform_factory import get_train_transform, get_infer_transform
from src.env_constants import *
from src.lightning_runner.loss_accuracy_csv_log_callback import LossAccuracyCsvLogCallback
from src.lightning_runner.lightning_model_wrapper import LightningWrapper
from src.lightning_runner.per_sample_csv_log_callback import PerSampleCsvLogCallback
from src.models.model_getter import get_resnet18
from src.util_functions import create_output_dir


if __name__ == '__main__':

    # --- Run parameters
    RUN_NAME = 'yaniv_infer'
    ADD_TIMESTAMP_TO_OUT_DIR = False
    CHECKPOINT_PATH = Path('D:\docs\Study\DSML_IDC\Final project\project\outputs\checkpoint_epoch=22_validation_loss=0.277_validation_accuracy=0.9202.ckpt')
    BATCH_SIZE = 2
    NUM_WORKERS = 0

    VERY_SHORT_DEBUG_RUN = True     #TODO: change in production!

    # --- output dir creation
    outputs_path = create_output_dir(PROJECT_PATH, RUN_NAME, ADD_TIMESTAMP_TO_OUT_DIR)

    # --- Datasets
    test_transform = get_infer_transform()
    test_dataset = DogsVsCatsDataset(TEST_DATA_PATH, transform=test_transform, cache_data=False, shuffle=False)

    # --- Debug option for very short run (5 batches in each dataloader)
    if VERY_SHORT_DEBUG_RUN:
        test_dataset, _ = random_split(test_dataset, [5, len(test_dataset) - 5])

    # --- Dataloaders
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # --- model, optimizer, loss
    model = get_resnet18(pretrained=False, out_features=1)
    optimizer = Adam(model.parameters())
    criterion = BCELoss()

    # --- Lightning wrapper module and callbacks
    l_module = LightningWrapper.load_from_checkpoint(CHECKPOINT_PATH, model=model, optimizer=optimizer, criterion=criterion)

    pred_output_file = str(outputs_path / Path('sample_predictions_log.csv'))

    trainer = L.Trainer()
    preds = trainer.predict(model=l_module, dataloaders=test_dataloader)

    #--- prepare and save output csv
    preds_formatted = [p.numpy() for p in preds]
    preds_formatted = [p[i][0] for p in preds_formatted for i in range(len(p))]

    dataset_idx = test_dataset.index
    dataset_idx['prediction'] = preds_formatted

    dataset_idx.to_csv(pred_output_file)