import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.datasets.dogs_vs_cats_dataset import DogsVsCatsDataset
from src.datasets.transform_factory import get_train_transform
from src.env_constants import *
from src.models.model_getter import get_resnet18
from src.util_functions import create_output_dir


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

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    # --- model, optimizer, loss
    model = get_resnet18(pretrained=False, out_features=1)
    optimizer = Adam(model.parameters())
    criterion = BCEWithLogitsLoss()

    # --- train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = model.to(device)
    model.train()

    NUM_EPOCHS = 10
    num_batches = len(train_dataloader)

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            preds = model(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'epoch {epoch}, batch {batch_idx}/{num_batches}, loss: {loss}')





















