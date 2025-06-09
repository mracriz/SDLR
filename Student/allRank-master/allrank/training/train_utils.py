import os
from functools import partial
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from csv import DictWriter
from pathlib import Path  # <-- Added pathlib
import allrank.models.metrics as metrics_module
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_num_params, log_num_params
from allrank.training.early_stop import EarlyStop
from allrank.utils.ltr_logging import get_logger
from allrank.utils.tensorboard_utils import TensorboardSummaryWriter
from allrank import config

logger = get_logger()


# NOTE: estimate_propensities_pairwise function is removed as Student does not estimate parameters.

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


def metric_on_batch(path,name,epoch,epochs,flag,metric, model, xb, yb, indices):
    d=0
    mask = (yb == PADDED_Y_VALUE)
    if flag=="test":
        if epoch == epochs-1:
            d = 1
    y_pred_scores = model.score(xb, mask, indices)
    true_labels = yb
    return metric(y_pred_scores, true_labels)


def metric_on_epoch(path,name,epoch,epochs,flag,metric, model, dl, dev):
    metric_values = torch.mean(
        torch.cat(
            [metric_on_batch(path,name,epoch,epochs,flag,metric, model, xb.to(device=dev), yb.to(device=dev), indices.to(device=dev))
             for xb, yb, indices in dl]
        ), dim=0
    ).cpu().numpy()
    return metric_values


def compute_metrics(path, name, epoch,epochs,flag,metrics, model, dl, dev):
    metric_values_dict = {}
    for metric_name, ats in metrics.items():
        metric_func = getattr(metrics_module, metric_name)
        metric_func_with_ats = partial(metric_func, ats=ats)
        metrics_values = metric_on_epoch(path,name,epoch,epochs,flag,metric_func_with_ats, model, dl, dev)
        metrics_names = ["{metric_name}_{at}".format(metric_name=metric_name, at=at) for at in ats]
        metric_values_dict.update(dict(zip(metrics_names, metrics_values)))
    return metric_values_dict


def epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics):
    summary = "Epoch : {epoch} Train loss: {train_loss} Val loss: {val_loss}".format(
        epoch=epoch, train_loss=train_loss, val_loss=val_loss)
    for metric_name, metric_value in train_metrics.items():
        summary += " Train {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)
    for metric_name, metric_value in val_metrics.items():
        summary += " Val {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)
    return summary


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit(epochs, model, loss_func, optimizer, scheduler, train_dl, valid_dl, config,
        gradient_clipping_norm, early_stopping_patience, device, output_dir, tensorboard_output_path):
    tensorboard_summary_writer = TensorboardSummaryWriter(tensorboard_output_path)
    num_params = get_num_params(model)
    log_num_params(num_params)
    early_stop = EarlyStop(early_stopping_patience)
    config.data.slate_length = int(np.ceil(max([train_dl.dataset.longest_query_length, valid_dl.dataset.longest_query_length]) / 10) * 10)
    
    params_path_name = config.data.path[config.data.path.rfind("Dataset/") + 8:].replace("/", "_")
    loss_func.keywords["Parameters_Path"] = params_path_name
    
    is_ips_loss = "IPS" in config.loss.name
    
    if is_ips_loss:
        num_positions = config.data.slate_length
        try:
            # --- PATH MODIFIED AS PER YOUR REQUEST ---
            base_path = Path(os.getcwd()).parent.parent.parent / "Teacher" / "allRank-master" / "allrank" / "Parameters" / "One"
            propensity_path = base_path / f"propensities_{params_path_name}.pt"
            # -----------------------------------------

            logger.info(f"Student (IPS): loading propensities from {propensity_path}")
            propensities = torch.load(propensity_path)
            t_plus = propensities['t_plus'].to(device)
            t_minus = propensities['t_minus'].to(device)
        except FileNotFoundError:
            logger.error(f"Propensity file not found for student model at {propensity_path}. Please run the teacher first.")
            return

    for epoch in range(epochs):
        logger.info("Current learning rate: {}".format(get_current_lr(optimizer)))
        
        loss_func.keywords["epoch"] = epoch
        
        model.train()

        train_losses = []
        train_nums = []
        for xb, yb, indices in train_dl:
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

            loss, num = loss_batch(
                model, loss_func, xb, yb, indices, gradient_clipping_norm, optimizer, inverse_propensities_list=inverse_propensities_list)
            train_losses.append(loss)
            train_nums.append(num)
        
        train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        train_metrics = compute_metrics(params_path_name, config.loss.name, epoch, epochs, "train", config.metrics, model, train_dl, device)

        model.eval()
        with torch.no_grad():
            val_losses, val_nums = zip(
                *[loss_batch(model, loss_func, xb.to(device=device), yb.to(device=device), indices.to(device=device),
                             gradient_clipping_norm) for
                  xb, yb, indices in valid_dl])
            val_metrics = compute_metrics(params_path_name, config.loss.name, epoch, epochs, "test", config.metrics, model, valid_dl, device)

        val_loss = np.sum(np.multiply(val_losses, val_nums)) / np.sum(val_nums)
        
        logger.info(epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics))
        
        current_val_metric_value = val_metrics.get(config.val_metric)
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_val_metric_value)
            else:
                scheduler.step()
        
        early_stop.step(current_val_metric_value, epoch)

        if early_stop.stop_training(epoch):
            logger.info("Early stopping triggered.")
            break

    logger.info("Student training finished.")
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pkl"))
    tensorboard_summary_writer.close_all_writers()

    return {
        "epochs": epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "num_params": num_params
    }