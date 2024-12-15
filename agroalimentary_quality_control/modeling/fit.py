import torch
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import dagshub
import mlflow
import numpy as np

import os
import argparse
from tqdm import tqdm
import random

from agroalimentary_quality_control.modeling.dataset import RocketDataset
from agroalimentary_quality_control.modeling.regressor import RocketRegressor


def training_step(model, loader, device, optimizer, criterion):
    model.train()

    train_loss = 0

    for images, targets in tqdm(
        loader,
        desc="Training",
        leave=False
    ):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()

        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(loader)


def validation_step(model, loader, device, criterion):
    model.eval()

    val_loss = 0

    with torch.no_grad():
        for images, targets in tqdm(
            loader,
            desc="Validation",
            leave=False
        ):
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            loss = criterion(predictions, targets)

            val_loss += loss.item()

    return val_loss / len(loader)


def _fit(
    fold,
    split_path,
    model_path,
    train_set_file_name,
    val_set_file_name,
    pics_path,
    target_cols,
    resize_ratio,
    batch_size,
    learning_rate,
    weight_decay,
    epochs,
    patience,
    parent_run
):
    train_set = RocketDataset(
        f'{split_path}/{train_set_file_name}',
        pics_path,
        target_cols,
        resize=resize_ratio
    )

    val_set = RocketDataset(
        f'{split_path}/{val_set_file_name}',
        pics_path,
        target_cols,
        resize=resize_ratio
    )

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RocketRegressor(target_cols)
    model = model.to(device)

    criterion = MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    epochs_no_improve = 0

    child_run = mlflow.start_run(run_name=f"Fold #{fold}", nested=True)

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", leave=False):
        avg_train_loss = training_step(
            model,
            train_loader,
            device,
            optimizer,
            criterion
        )

        mlflow.log_metric(f"Training loss", avg_train_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Train Loss: {avg_train_loss}")

        avg_val_loss = validation_step(
            model,
            val_loader,
            device,
            criterion
        )

        mlflow.log_metric(f"Validation loss", avg_val_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_train_loss = avg_train_loss
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            torch.save({
                'weights': model.state_dict(),
                'child_run_id': child_run.info.run_id,
                'parent_run_id': parent_run.info.run_id
            }, model_path)
            
            mlflow.log_metric(f"Output validation loss", avg_val_loss, step=epoch)
            mlflow.log_metric(f"Output training loss", avg_train_loss, step=epoch)
            
            tqdm.write("Best model weights have been saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            tqdm.write(f"Early stopping triggered after {epoch} epochs.")
            break

    mlflow.end_run()

    return best_train_loss, best_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--models_path', type=str)
    parser.add_argument('--splits_path', type=str)
    parser.add_argument('--train_set_file_name', type=str)
    parser.add_argument('--val_set_file_name', type=str)
    parser.add_argument('--pics_path', type=str)
    parser.add_argument('--resize_ratio', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--target_cols', nargs='+', type=str)
    parser.add_argument('--repo_owner', type=str)
    parser.add_argument('--repo_name', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dagshub.init(repo_owner=args.repo_owner, repo_name=args.repo_name, mlflow=True)
    mlflow.set_experiment(args.experiment_name)

    os.makedirs(args.models_path)

    with mlflow.start_run() as parent_run:
        mlflow.log_param("Epochs", args.epochs)
        mlflow.log_param("Initial learning rate", args.learning_rate)
        mlflow.log_param("Batch size", args.batch_size)
        mlflow.log_param("Patience", args.patience)
        mlflow.log_param("Weight decay", args.weight_decay)
        mlflow.log_param("Resize ratio", args.resize_ratio)
        mlflow.log_param("Random seed", args.seed)

        splits = os.listdir(args.splits_path)

        losses = []

        for split in splits:
            split_path = f'{args.splits_path}/{split}'
            model_path = f'{args.models_path}/{split}.pth'

            loss = _fit(
                split,
                split_path,
                model_path,
                args.train_set_file_name,
                args.val_set_file_name,
                args.pics_path,
                args.target_cols,
                args.resize_ratio,
                args.batch_size,
                args.learning_rate,
                args.weight_decay,
                args.epochs,
                args.patience,
                parent_run
            )

            losses.append(loss)

        avg_train_loss, avg_val_loss = np.array(losses).mean(axis=0)

        mlflow.log_metric("Avg. Training Loss", avg_train_loss)
        mlflow.log_metric("Avg. Validation Loss", avg_val_loss)
