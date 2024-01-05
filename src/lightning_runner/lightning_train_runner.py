import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset
from src.datasets.transform_factory import get_train_transform
from src.env_constants import *
from src.lightning_runner.lightning_model_wrapper import LightningWrapper
from src.models.model_getter import get_resnet18
from src.util_functions import create_output_dir
import lightning as L


if __name__ == '__main__':
    run_name = 'yaniv_test'

    outputs_path = create_output_dir(PROJECT_PATH, run_name, True)

    train_transform = get_train_transform()
    train_val_dataset = DogsVsCatsDataset(TRAIN_DATA_PATH, transform=train_transform, cache_data=True, shuffle=True)

    # --- Split train and validation
    train_split_ratio = 0.75

    train_split_th = int(np.floor(train_split_ratio * len(train_val_dataset)))
    train_val_indices = list(range(len(train_val_dataset)))

    train_dataset, valid_dataset = random_split(train_val_dataset, [train_split_th, len(train_val_dataset) - train_split_th])

    # --- Dataloaders
    BATCH_SIZE = 2
    NUM_WORKERS = 0

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # --- model, optimizer, loss
    model = get_resnet18(pretrained=False, out_features=1)
    optimizer = Adam(model.parameters())
    criterion = BCEWithLogitsLoss()

    # --- Lightning wrapper module
    l_module = LightningWrapper(model, Adam, criterion)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    NUM_EPOCHS = 10

    trainer = L.Trainer(max_epochs=NUM_EPOCHS, check_val_every_n_epoch=1)
    trainer.fit(model=l_module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)






















