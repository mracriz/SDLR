import numpy as np
import torch
import os
from pathlib import Path
import pandas as pd
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS

def listSDStu_IPS(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, xb=None, epoch=0, Parameters_Path=None, inverse_propensities_list=None):

    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
    mask_padded = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask_padded] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss_per_doc = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    observation_loss_per_doc[mask_padded] = 0.0

    observation_loss = torch.sum(observation_loss_per_doc, dim=1)

    target_file_in_teacher = Path(os.getcwd()).parent.parent.parent / "Teacher" / "allRank-master" / "allrank" / "Parameters" / "One" / f"Sigma_All_Score_{str(Parameters_Path)}.csv"
    
    df_bandwidth = pd.read_csv(target_file_in_teacher, index_col=0)
    initial_bandwidth_np = df_bandwidth.to_numpy()
    initial_bandwidth_np[np.where(initial_bandwidth_np < 1e-11)] = 1
    Initial_BandWidth = torch.tensor(initial_bandwidth_np, dtype=torch.float32, device=xb.device)

    doc_mask = (y_true != padded_value_indicator).float()
    
    y_true_int = y_true.long().clamp(0, Initial_BandWidth.shape[0] - 1)
    batch_bandwidths = Initial_BandWidth[y_true_int]
    
    pairwise_sq_diffs = torch.pow(xb.unsqueeze(2) - xb.unsqueeze(1), 2)
    
    kernel_divisor = 2 * torch.pow(batch_bandwidths.unsqueeze(2), 2) + eps
    exponent = torch.sum(pairwise_sq_diffs / kernel_divisor, dim=-1)
    kernels = torch.exp(-exponent)
    
    doc_mask_2d = doc_mask.unsqueeze(-1) * doc_mask.unsqueeze(1)
    identity_matrix = torch.eye(kernels.shape[1], device=xb.device).unsqueeze(0)
    valid_kernels = kernels * doc_mask_2d * (1 - identity_matrix)

    num_docs_in_list = doc_mask.sum(dim=1, keepdim=True)
    doc_density = torch.sum(valid_kernels, dim=2) / (num_docs_in_list - 1).clamp(min=1)
    log_prior_x = torch.sum(torch.log(doc_density + eps) * doc_mask, dim=1)
    Prior_X = torch.exp(log_prior_x)

    weighted_loss = torch.multiply(observation_loss, Prior_X)
    
    if inverse_propensities_list is not None:
        weighted_loss = torch.multiply(weighted_loss, inverse_propensities_list.to(weighted_loss.device))

    return torch.mean(weighted_loss)