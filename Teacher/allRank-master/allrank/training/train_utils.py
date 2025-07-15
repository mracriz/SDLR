import os
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import mlflow  # <-- Import MLflow

import allrank.models.metrics as metrics_module
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_num_params, log_num_params
from allrank.training.early_stop import EarlyStop
from allrank.utils.ltr_logging import get_logger
from allrank.utils.tensorboard_utils import TensorboardSummaryWriter
from allrank import config as conf # Use `conf` for global parameters

logger = get_logger()

# =================================================================================
# Helper functions (Your original code - no changes needed here)
# =================================================================================

def estimate_propensities_pairwise(model, dataloader, device, num_positions, t_plus, t_minus):
    logger.info("Estimating propensities...")
    model.eval()
    t_plus_num = torch.zeros(num_positions, device=device)
    t_minus_num = torch.zeros(num_positions, device=device)
    epsilon = 1e-9
    with torch.no_grad():
        for (features, labels, indices) in tqdm(dataloader, desc="Estimating Propensities"):
            features, labels = features.to(device), labels.to(device)
            scores = model.score(features, (labels == PADDED_Y_VALUE), indices)
            for i in range(labels.shape[0]):
                clicked_indices = (labels[i] > 0).nonzero(as_tuple=True)[0]
                unclicked_indices = (labels[i] == 0).nonzero(as_tuple=True)[0]
                for clicked_idx in clicked_indices:
                    for unclicked_idx in unclicked_indices:
                        loss_pair = torch.sigmoid(-(scores[i, clicked_idx] - scores[i, unclicked_idx]))
                        if clicked_idx < num_positions and unclicked_idx < num_positions:
                            t_plus_num[clicked_idx] += loss_pair / (t_minus[unclicked_idx] + epsilon)
                            t_minus_num[unclicked_idx] += loss_pair / (t_plus[clicked_idx] + epsilon)
    new_t_plus = t_plus_num / (t_plus_num[0] + epsilon)
    new_t_minus = t_minus_num / (t_minus_num[0] + epsilon)
    new_t_plus = torch.clamp(new_t_plus, min=0.1)
    new_t_minus = torch.clamp(new_t_minus, min=0.1)
    new_t_plus[0] = 1.0
    new_t_minus[0] = 1.0
    model.train()
    logger.info("Propensity estimation complete.")
    return new_t_plus, new_t_minus


def loss_batch(model, loss_func, xb, yb, indices, gradient_clipping_norm, opt=None, inverse_propensities_list=None):
    mask = (yb == PADDED_Y_VALUE)
    loss_kwargs = {"xb": xb}
    if inverse_propensities_list is not None:
        loss_kwargs["inverse_propensities_list"] = inverse_propensities_list
    loss = loss_func(model(xb, mask, indices), yb, **loss_kwargs)
    if opt is not None:
        loss.backward()
        if gradient_clipping_norm:
            clip_grad_norm_(model.parameters(), gradient_clipping_norm)
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def metric_on_batch(metric, model, xb, yb, indices):
    mask = (yb == PADDED_Y_VALUE)
    y_pred_scores = model.score(xb, mask, indices)
    return metric(y_pred_scores, yb)


def metric_on_epoch(metric, model, dl, dev):
    metric_values = torch.mean(
        torch.cat(
            [metric_on_batch(metric, model, xb.to(device=dev), yb.to(device=dev), indices.to(device=dev))
             for xb, yb, indices in dl]
        ), dim=0
    ).cpu().numpy()
    return metric_values


def compute_metrics(metrics, model, dl, dev):
    metric_values_dict = {}
    for metric_name, ats in metrics.items():
        metric_func = getattr(metrics_module, metric_name)
        metric_func_with_ats = partial(metric_func, ats=ats)
        metrics_values = metric_on_epoch(metric_func_with_ats, model, dl, dev)
        metrics_names = [f"{metric_name}_{at}" for at in ats]
        metric_values_dict.update(dict(zip(metrics_names, metrics_values)))
    return metric_values_dict


def epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics):
    summary = f"Epoch : {epoch} Train loss: {train_loss:.4f} Val loss: {val_loss:.4f}"
    for name, value in train_metrics.items():
        summary += f" Train {name}: {np.mean(value):.4f}"
    for name, value in val_metrics.items():
        summary += f" Val {name}: {np.mean(value):.4f}"
    return summary


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

# =================================================================================
# Main `fit` function with MLflow Integration
# =================================================================================

def fit(epochs, model, loss_func, optimizer, scheduler, train_dl, valid_dl, config,
        gradient_clipping_norm, early_stopping_patience, device, output_dir, tensorboard_output_path):
    
    tensorboard_summary_writer = TensorboardSummaryWriter(tensorboard_output_path)
    num_params = get_num_params(model)
    log_num_params(num_params)
    early_stop = EarlyStop(early_stopping_patience)
    
    is_ips_loss = "IPS" in config.loss.name
    if is_ips_loss:
        logger.info(f"IPS loss '{config.loss.name}' detected. Initializing propensity estimation.")
        num_positions = config.data.slate_length
        t_plus = torch.ones(num_positions, device=device)
        t_minus = torch.ones(num_positions, device=device)

    # Your custom BandWidth initialization logic using the global `conf`
    conf.BandWidth_LR = torch.tensor(conf.BandWidth_LR, dtype=torch.float32).to(device)
    conf.BandWidth_Changes = []
    Temp = []
    for xb, yb, indices in train_dl:
        Temp += torch.unique(yb).tolist()
    Unique_Labels = torch.sort(torch.unique(torch.tensor(Temp))).values
    Unique_Labels = Unique_Labels[1:]
    conf.BandWidth = torch.ones(size=(Unique_Labels.shape[0], xb.shape[-1]), dtype=torch.float64)
    Temp_Coefficient = torch.ones(Unique_Labels.shape[0], 1)
    if Unique_Labels.shape[0]>0:
        Temp_Coefficient[0] = torch.multiply(Temp_Coefficient[0], torch.tensor(0.71))
    if Unique_Labels.shape[0]>1:
        Temp_Coefficient[1:] = torch.multiply(Temp_Coefficient[1:], torch.tensor(1.0))

    conf.BandWidth = torch.multiply(conf.BandWidth, Temp_Coefficient)
    conf.BandWidth = conf.BandWidth.to(device)
    conf.Best_BandWidth = torch.clone(conf.BandWidth)

    # --- Main Training Loop ---
    for epoch in range(epochs):
        logger.info(f"Current learning rate: {get_current_lr(optimizer)}")
        
        if is_ips_loss and epoch > 0:
            t_plus, t_minus = estimate_propensities_pairwise(model, train_dl, device, num_positions, t_plus, t_minus)

        loss_func.keywords["epoch"] = epoch
        conf.BandWidth_Loss_Derivation = torch.zeros(size=(conf.BandWidth.shape[:2]), dtype=torch.float32).to(device)

        model.train()
        train_losses, train_nums = [], []
        for xb, yb, indices in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            xb, yb, indices = xb.to(device), yb.to(device), indices.to(device)
            inverse_propensities_list = None
            if is_ips_loss:
                inverse_propensities_itemwise = torch.ones_like(yb, dtype=torch.float32)
                for i in range(yb.shape[0]):
                    for j in range(yb.shape[1]):
                        if j < num_positions:
                            if yb[i, j] > 0:
                                inverse_propensities_itemwise[i, j] = 1.0 / t_plus[j]
                            else:
                                inverse_propensities_itemwise[i, j] = 1.0 / t_minus[j]
                inverse_propensities_list = torch.mean(inverse_propensities_itemwise, dim=1)
            loss, num = loss_batch(model, loss_func, xb, yb, indices, gradient_clipping_norm, optimizer, inverse_propensities_list)
            train_losses.append(loss)
            train_nums.append(num)
        
        train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        train_metrics = compute_metrics(config.metrics, model, train_dl, device)

        model.eval()
        with torch.no_grad():
            val_losses, val_nums = zip(*[loss_batch(model, loss_func, xb.to(d), yb.to(d), ind.to(d), gradient_clipping_norm)
                                         for xb, yb, ind in tqdm(valid_dl, desc=f"Epoch {epoch+1}/{epochs} [Validation]")
                                         for d in [device]])
            val_metrics = compute_metrics(config.metrics, model, valid_dl, device)
        val_loss = np.sum(np.multiply(val_losses, val_nums)) / np.sum(val_nums)
        
        conf.BandWidth_Changes.append(torch.mean(torch.multiply(conf.BandWidth_LR, conf.BandWidth_Loss_Derivation)).item())
        Temp_Update = torch.abs(torch.subtract(conf.BandWidth, torch.multiply(conf.BandWidth_LR, conf.BandWidth_Loss_Derivation)))
        Temp_Update_Indices = ((Temp_Update < conf.BandWidth).to(torch.int8) + (Temp_Update < 0.3).to(torch.int8) > 0.9).to(torch.int8)
        conf.BandWidth = torch.add(torch.multiply(1.0 - Temp_Update_Indices, conf.BandWidth), torch.multiply(Temp_Update_Indices, Temp_Update))
        
        logger.info(epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics))

        # --- MLflow Logging Integration ---
        if mlflow.active_run():
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)
            mlflow.log_metric("bandwidth_change", conf.BandWidth_Changes[-1], step=epoch)
        # --- End of MLflow Integration ---
        
        current_val_metric_value = val_metrics.get(config.val_metric)
        if scheduler:
            scheduler.step(current_val_metric_value if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)
        
        if early_stop.step(current_val_metric_value, epoch):
            conf.Best_BandWidth = torch.clone(conf.BandWidth)

        if early_stop.stop_training(epoch):
            logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # --- Final Artifact Saving ---
    # Save files locally first, then log them to MLflow
    params_dir = os.path.join(output_dir, "parameters")
    os.makedirs(params_dir, exist_ok=True)
    logger.info(f"Training finished. Saving learned parameters to {params_dir}")

    bw_path = os.path.join(params_dir, "Sigma_All_Score.csv")
    best_bw_path = os.path.join(params_dir, "Sigma_All_Score_Best.csv")
    changes_path = os.path.join(params_dir, "Sigma_Changes.csv")
    
    pd.DataFrame(conf.BandWidth.cpu().numpy()).to_csv(bw_path, index=False)
    pd.DataFrame(conf.Best_BandWidth.cpu().numpy()).to_csv(best_bw_path, index=False)
    pd.DataFrame(conf.BandWidth_Changes, columns=["Changes"]).to_csv(changes_path, index=False)

    if mlflow.active_run():
        logger.info("Logging final parameters as MLflow artifacts.")
        mlflow.log_artifact(bw_path, "parameters")
        mlflow.log_artifact(best_bw_path, "parameters")
        mlflow.log_artifact(changes_path, "parameters")
        if is_ips_loss:
            propensity_path = os.path.join(params_dir, "propensities.pt")
            torch.save({'t_plus': t_plus, 't_minus': t_minus}, propensity_path)
            mlflow.log_artifact(propensity_path, "parameters")
    
    # NOTE: The model itself is saved in main.py using mlflow.pytorch.log_model
    # We do not need to save it again here.
   
    tensorboard_summary_writer.close_all_writers()

    return {
        "epochs": epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "num_params": num_params
    }