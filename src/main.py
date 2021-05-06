import logging
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from mlflow import log_artifacts, log_metric, log_param, start_run, set_tracking_uri
from mlflow.pytorch import log_model
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from constants import IMG_SIZE, OUTPUT_FOLDER_PATH, ARTIFACTS_PATH
from model import LinearVAE
from utils import load_data

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.DEBUG)
set_tracking_uri("postgresql://username:password@postgres:5432/mlflow_db")


def global_loss(bce_loss, mu, log_var):
    """
    This function adds the reconstruction loss (BCELoss) and the
    KL-Divergence.
    The KL-Divergence estimates the distance between to distributions
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    kl_divergence = 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce_loss - kl_divergence


def fit(
    model, dataloader, criterion, learning_rate, device,
):
    model.train()
    running_loss = 0.0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for _, (data, _) in enumerate(dataloader):
        data = data.to(device).view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, log_var = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = global_loss(bce_loss, mu, log_var)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss / len(dataloader.dataset)
    return train_loss


def validate(model, dataloader, epoch, criterion, device, num_rows=8):
    model.eval()
    running_loss = 0

    dataset_iter = iter(dataloader)
    save_image(
        next(dataset_iter)[0].view(dataloader.batch_size, 1, IMG_SIZE, IMG_SIZE).cpu(),
        os.path.join(OUTPUT_FOLDER_PATH, f"output_epoch_{0}.png"),
        nrow=num_rows,
    )

    with torch.no_grad():
        for batch_index, (data, _) in enumerate(dataloader):
            data = data.to(device).view(data.size(0), -1)
            reconstruction, mu, log_var = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = global_loss(bce_loss, mu, log_var)
            running_loss += loss.item()

            # save the first batch input and output of specific epoch
            if batch_index == 0:
                if epoch % 10 == 0:
                    save_image(
                        reconstruction.view(
                            dataloader.batch_size, 1, IMG_SIZE, IMG_SIZE
                        ).cpu(),
                        os.path.join(OUTPUT_FOLDER_PATH, f"output_epoch_{epoch}.png"),
                        nrow=num_rows,
                    )

    val_loss = running_loss / len(dataloader.dataset)
    return val_loss


def train_model(num_epochs=20, batch_size=64, num_features=20, learning_rate=0.0001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = load_data()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = LinearVAE(num_features).to(device)
    criterion = nn.BCELoss(reduction="sum")

    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.mkdir("outputs")

    with start_run():

        validate(model, val_loader, 0, criterion, device)
        val_epoch_loss = np.inf

        for epoch in range(1, num_epochs + 1):
            logging.info("Epoch %s/%s", epoch, num_epochs)
            train_epoch_loss = fit(
                model, train_loader, criterion, learning_rate, device,
            )
            val_epoch_loss = validate(model, val_loader, epoch, criterion, device)
            logging.info(
                "Train Loss: %s", train_epoch_loss,
            )
            logging.info(f"Val Loss: %s \n", val_epoch_loss)

        log_param("num_features", num_features)
        log_param("learning_rate", learning_rate)
        log_metric("validation_loss", val_epoch_loss)
        log_metric("num_epochs", num_epochs)
        log_artifacts(OUTPUT_FOLDER_PATH, ARTIFACTS_PATH)
        log_model(model, "LinearVAE", registered_model_name="LinearVAE")


if __name__ == "__main__":
    train_model()
