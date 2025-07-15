import numpy as np
import torch

from allrank import config as conf
import pandas as pd
import os

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS

def listSD(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, xb=None, epoch=0, Parameters_Path=None):
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    Initial_BandWidth = torch.clone(conf.Best_BandWidth).to(xb.device)
    Prior_X = torch.ones(size=(xb.shape[0], 1), dtype=torch.float32, device=xb.device)
    Temp_BandWidth_Loss = torch.zeros(size=([xb.shape[0]] + list(conf.BandWidth.shape)), dtype=torch.float32, device=xb.device)

    for i in range(xb.shape[0]):
        X_Temp = xb[i]

        Padded_Doc = torch.where(torch.sum(X_Temp, dim=1) == 0)[0].tolist()
        Temp_docs = list(range(X_Temp.shape[0]))
        [Temp_docs.remove(j) for j in Padded_Doc]
        X_Temp = X_Temp[Temp_docs]

        for j in range(X_Temp.shape[0]):
            if j > 5: break
            
            # --- THIS IS THE FIX ---
            # 1. Skip documents with relevance label 0, as they have no bandwidth calculated.
            current_label = y_true[i, j]
            if current_label == 0:
                continue

            # 2. Use the correct index for the bandwidth tensor (label 1 -> index 0, etc.)
            bandwidth_index = int(current_label) - 1
            # --- END OF FIX ---

            Temp = list(range(X_Temp.shape[0]))
            Temp.remove(j)
            Temp = X_Temp[Temp]
            if Temp.shape[0] == 0:
                break
            
            Temp = torch.subtract(Temp, X_Temp[j])
            Temp = torch.pow(Temp, 2)
            
            Temp_2 = torch.divide(Temp, torch.add(torch.multiply(torch.tensor(2), torch.pow(Initial_BandWidth[bandwidth_index], 2)), torch.tensor(1e-23)))
            Temp_2 = torch.sum(Temp_2, dim=1)
            Temp_2 = Temp_2.reshape((Temp_2.shape[0], 1))
            Temp_2 = torch.exp(torch.multiply(torch.tensor(-1), Temp_2))

            ### BandWidth Loss Derivation ###
            Temp_3 = torch.add(torch.pow(Initial_BandWidth[bandwidth_index], 2), Temp)
            Temp_3 = torch.divide(Temp_3, torch.add(torch.pow(Initial_BandWidth[bandwidth_index], 3), torch.tensor(1e-23)))
            Coefficient = torch.tensor(1.0)
            Temp_3 = torch.multiply(Coefficient, Temp_3)
            Temp_3 = torch.multiply(Temp_3, Temp_2)
            Temp_3 = torch.sum(Temp_3, dim=0) / X_Temp.shape[0]
            
            Temp_BandWidth_Loss[i, bandwidth_index] = torch.multiply(Temp_BandWidth_Loss[i, bandwidth_index], Temp_3)
            if torch.sum(Temp_BandWidth_Loss[i, bandwidth_index]) == 0:
                Temp_BandWidth_Loss[i, bandwidth_index] = Temp_3
            ###

            Coefficient = torch.tensor(1.0)
            Temp_2 = torch.multiply(Coefficient, Temp_2)
            Temp_2 = torch.sum(Temp_2)
            
            if epoch > 31 and (epoch % 2 == 1):
                Prior_X[i] *= (Temp_2 / X_Temp.shape[0])
    
    observation_loss[mask] = 0.0
    observation_loss = torch.sum(observation_loss, dim=1)

    Temp = torch.reshape(observation_loss, shape=(observation_loss.shape[0], 1, 1))
    Temp_BandWidth_Loss = torch.multiply(Temp, Temp_BandWidth_Loss[:, :, :])
    Temp_BandWidth_Loss = torch.sum(Temp_BandWidth_Loss, dim=0)
    conf.BandWidth_Loss_Derivation = torch.add(conf.BandWidth_Loss_Derivation, Temp_BandWidth_Loss)
    
    observation_loss = torch.multiply(Prior_X[:, 0], observation_loss)
    observation_loss = torch.multiply(torch.tensor(10.0), observation_loss)

    return torch.mean(observation_loss)