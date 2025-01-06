import torch
from torch import optim
from torch.nn import Module, CosineEmbeddingLoss, MSELoss
from torch.utils.data import DataLoader
import dagshub
import mlflow
import numpy as np
import pandas as pd

import os
import argparse
from tqdm import tqdm
import random

from agroalimentary_quality_control.modeling.dataset import RocketDataset, ContrastiveRocketDataset
from agroalimentary_quality_control.modeling.regressor import RocketRegressor


class RSELoss(Module):
    def __init__(self):
        super(RSELoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        numerator = torch.sum((y_true - y_pred) ** 2)
        denominator = torch.sum((y_true - torch.mean(y_true)) ** 2) + 1e-8
        rse = numerator / denominator
        return rse


def dynamic_binning(images, targets, n_bins):
    quantiles = torch.quantile(targets, torch.linspace(0, 1, steps=n_bins))
    bin_edges = quantiles.tolist()
    bin_indices = torch.bucketize(targets, torch.tensor(bin_edges))
    negative_bin_indices = (bin_indices + (n_bins / 2)) % n_bins

    images_contrast = []
    labels = []

    for i in range(len(images)):
        anchor_bin = bin_indices[i]
        negative_bin = negative_bin_indices[i]
        
        anchor_indices = torch.where(bin_indices == anchor_bin)[0]
        negative_indices = torch.where(bin_indices == negative_bin)[0]

        if len(negative_indices) == 0 or torch.rand(1).item() < 0.5:
            anchor_idx = anchor_indices[torch.randint(0, len(anchor_indices), (1,))]
            anchor_image = images[anchor_idx].squeeze(0)
            images_contrast.append(anchor_image)
            labels.append(1)
        else:
            negative_idx = negative_indices[torch.randint(0, len(negative_indices), (1,))]
            negative_image = images[negative_idx].squeeze(0)
            images_contrast.append(negative_image)
            labels.append(-1)

    images_contrast = torch.stack(images_contrast)

    labels = torch.tensor(labels)

    return images_contrast, labels


def contrastive_training_step(model, loader, device, optimizer, contrastive_loss_fn, n_bins):
    model.train()

    tot_train_loss = 0

    for images, targets in tqdm(
        loader,
        desc="Contrastive training",
        leave=False
    ):
        images_contrast, labels = dynamic_binning(images, targets, n_bins)

        images = images.to(device)
        images_contrast = images_contrast.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        _, img_emb = model(images)
        _, con_emb = model(images_contrast)

        train_loss = contrastive_loss_fn(img_emb, con_emb, labels)
        train_loss.backward()

        optimizer.step()

        tot_train_loss += train_loss.item()

    return tot_train_loss / len(loader)


def contrastive_validation_step(model, loader, device, contrastive_loss_fn, n_bins):
    model.eval()

    val_loss_tot = 0

    with torch.no_grad():
        for images, targets in tqdm(
            loader,
            desc="Validation contrastive",
            leave=False
        ):
            images_contrast, labels = dynamic_binning(images, targets, n_bins)

            images = images.to(device)
            images_contrast = images_contrast.to(device)
            labels = labels.to(device)

            _, img_emb = model(images)
            _, con_emb = model(images_contrast)

            val_loss = contrastive_loss_fn(img_emb, con_emb, labels)

            val_loss_tot += val_loss.item()

    return val_loss_tot / len(loader)


def regression_training_step(model, loader, device, optimizer, regression_loss_fn):
    model.train()

    tot_train_loss = 0

    for images, targets in tqdm(
        loader,
        desc="Training regression",
        leave=False
    ):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        
        predictions, _ = model(images)

        train_loss = regression_loss_fn(predictions, targets)

        train_loss.backward()

        optimizer.step()

        tot_train_loss += train_loss.item()

    return tot_train_loss / len(loader)


def regression_validation_step(model, loader, device, criterion):
    model.eval()

    val_loss = 0

    with torch.no_grad():
        for images, targets in tqdm(
            loader,
            desc="Validation regression",
            leave=False
        ):
            images, targets = images.to(device), targets.to(device)
            predictions, _ = model(images)
            loss = criterion(predictions, targets)

            val_loss += loss.item()

    return val_loss / len(loader)


def _fit(
    pretrained_model_output_size,
    pretrained_model_path,
    fold,
    split_path,
    model_path,
    train_set_file_name,
    val_set_file_name,
    target_col,
    resize_ratio,
    batch_size,
    learning_rate,
    weight_decay,
    epochs,
    patience,
    filename_col,
    n_bins,
    parent_run
):
    train_df = pd.read_csv(f'{split_path}/{train_set_file_name}')
    val_df = pd.read_csv(f'{split_path}/{val_set_file_name}')

    train_set = RocketDataset(
        train_df,
        target_col,
        filename_col,
        resize=resize_ratio
    )

    val_set = RocketDataset(
        val_df,
        target_col,
        filename_col,
        resize=resize_ratio
    )

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RocketRegressor(
        pretrained_model_output_size,
        pretrained_model_path,
        device
    )
    model = model.to(device)

    child_run = mlflow.start_run(run_name=f"Fold #{fold}", nested=True)

    tqdm.write('CONTRASTIVE TRAINING')

    contrastive_loss_fn = CosineEmbeddingLoss()

    best_train_loss_con = float('inf')
    best_val_loss_con = float('inf')
    epochs_no_improve = 0

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    for epoch in range(1, epochs + 1):
        tqdm.write(f'Epoch {epoch} out of {epochs}')
        
        avg_train_loss = contrastive_training_step(
            model,
            train_loader,
            device,
            optimizer,
            contrastive_loss_fn,
            n_bins
        )

        mlflow.log_metric(f"Train CosEmb Loss", avg_train_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Train Cosine Embedding Loss: {avg_train_loss}")

        avg_val_loss = contrastive_validation_step(
            model,
            val_loader,
            device,
            contrastive_loss_fn,
            n_bins
        )

        mlflow.log_metric(f"Val CosEmb Loss", avg_val_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Validation Cosine Embedding Loss: {avg_val_loss}")

        # Early stopping
        if avg_val_loss < best_val_loss_con:
            best_train_loss_con = avg_train_loss
            best_val_loss_con = avg_val_loss
            epochs_no_improve = 0
            
            mlflow.log_metric(f"Out Val CosEmb Loss", avg_val_loss, step=epoch)
            mlflow.log_metric(f"Out Val CosEmb Loss", avg_train_loss, step=epoch)
            
            tqdm.write("New best model found.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            tqdm.write(f"Early stopping triggered after {epoch} epochs.")
            break

    tqdm.write('REGRESSION TRAINING')

    for param in model.model.parameters():
        param.requires_grad = False

    for param in model.embedder.parameters():
        param.requires_grad = False

    regression_loss_fn = MSELoss()

    best_train_loss_reg = float('inf')
    best_val_loss_reg = float('inf')
    epochs_no_improve = 0

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    for epoch in range(1, epochs + 1):
        tqdm.write(f'Epoch {epoch} out of {epochs}')
        
        avg_train_loss = regression_training_step(
            model,
            train_loader,
            device,
            optimizer,
            regression_loss_fn,
        )

        mlflow.log_metric(f"Train MSE Loss", avg_train_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Train MSE Loss: {avg_train_loss}")

        avg_val_loss = regression_validation_step(
            model,
            val_loader,
            device,
            regression_loss_fn
        )

        mlflow.log_metric(f"Val MSE Loss", avg_val_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Validation MSE Loss: {avg_val_loss}")

        # Early stopping
        if avg_val_loss < best_val_loss_reg:
            best_val_loss_reg = avg_train_loss
            best_val_loss_reg = avg_val_loss
            epochs_no_improve = 0
            
            torch.save({
                'weights': model.state_dict(),
                'child_run_id': child_run.info.run_id,
                'parent_run_id': parent_run.info.run_id
            }, model_path)
            
            mlflow.log_metric(f"Out Val MSE Loss", avg_val_loss, step=epoch)
            mlflow.log_metric(f"Out Val MSE Loss", avg_train_loss, step=epoch)
            
            tqdm.write("Best model weights have been saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            tqdm.write(f"Early stopping triggered after {epoch} epochs.")
            break

    mlflow.end_run()

    return best_train_loss_con, best_val_loss_con, best_train_loss_reg, best_val_loss_reg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_output_size', type=int)
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--models_path', type=str)
    parser.add_argument('--aug_splits_path', type=str)
    parser.add_argument('--train_set_file_name', type=str)
    parser.add_argument('--val_set_file_name', type=str)
    parser.add_argument('--resize_ratio', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--target_col', type=str)
    parser.add_argument('--repo_owner', type=str)
    parser.add_argument('--repo_name', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--filename_col', type=str)
    parser.add_argument('--n_bins', type=int)

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
        mlflow.log_param("Target", args.target_col)
        mlflow.log_param("Random seed", args.seed)

        splits = os.listdir(args.aug_splits_path)

        losses = []

        for split in splits:
            split_path = f'{args.aug_splits_path}/{split}'
            model_path = f'{args.models_path}/{split}.pth'

            loss = _fit(
                args.pretrained_model_output_size,
                args.pretrained_model_path,
                split,
                split_path,
                model_path,
                args.train_set_file_name,
                args.val_set_file_name,
                args.target_col,
                args.resize_ratio,
                args.batch_size,
                args.learning_rate,
                args.weight_decay,
                args.epochs,
                args.patience,
                args.filename_col,
                args.n_bins,
                parent_run
            )

            losses.append(loss)

        avg_train_loss_con, avg_val_loss_con, avg_train_loss_reg, avg_val_loss_reg = np.array(losses).mean(axis=0)

        mlflow.log_metric("Avg. Train CosEmb Loss", avg_train_loss_con)
        mlflow.log_metric("Avg. Val. CosEmb Loss", avg_val_loss_con)
        mlflow.log_metric("Avg. Train MSE Loss", avg_train_loss_reg)
        mlflow.log_metric("Avg. Val. MSE Loss", avg_val_loss_reg)

        tqdm.write(f"Avg. Train CosEmb Loss {avg_train_loss_con}")
        tqdm.write(f"Avg. Val. CosEmb Loss {avg_val_loss_con}")
        tqdm.write(f"Avg. Train MSE Loss {avg_train_loss_reg}")
        tqdm.write(f"Avg. Val. MSE Loss {avg_val_loss_reg}")
