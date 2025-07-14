import numpy as np
import torch
import os
from pathlib import Path

from allrank import config as conf
import pandas as pd

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS


def listSDStu_IPS(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, xb=None, epoch=0, Parameters_Path=None, inverse_propensities_list=None):
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

    # --- THIS LINE IS CORRECTED ---
    target_file_in_teacher = Path(os.getcwd()).parent.parent.parent / "Teacher" / "allRank-master" / "allrank" / "Parameters" / "One" / "Sigma_All_Score_Best.csv"
    
    # Logic to read the .csv file
    df_bandwidth = pd.read_csv(target_file_in_teacher, index_col=0)
    initial_bandwidth_np = df_bandwidth.to_numpy()
    initial_bandwidth_np[np.where(initial_bandwidth_np < 1e-11)] = 1
    Initial_BandWidth = torch.tensor(initial_bandwidth_np, dtype=torch.float32, device=xb.device)
    # ---------------------------------------------------

    Prior_X = torch.ones(size=(xb.shape[0], 1), dtype=torch.float32, device=xb.device)

    for i in range(xb.shape[0]):
        X_Temp = xb[i]

        Padded_Doc = torch.where(torch.sum(X_Temp, dim=1) == 0)[0].tolist()
        Temp_indices_list = list(range(X_Temp.shape[0]))
        [Temp_indices_list.remove(j) for j in Padded_Doc]
        X_Temp = X_Temp[Temp_indices_list]

        for j in range(X_Temp.shape[0]):
            if j > 10:
                break
            
            Current_X_Temp_indices = list(range(X_Temp.shape[0]))
            if not Current_X_Temp_indices: continue
            if j < len(Current_X_Temp_indices):
                Current_X_Temp_indices.remove(j)
                Temp_loop_var = X_Temp[Current_X_Temp_indices]
            else:
                continue

            if Temp_loop_var.shape[0] == 0:
                break
            
            Temp_loop_var = torch.subtract(Temp_loop_var, X_Temp[j])
            Temp_loop_var = torch.pow(Temp_loop_var, 2)
            
            original_doc_index_in_list_i = Temp_indices_list[j]
            current_y_true_label = y_true[i, original_doc_index_in_list_i]
            
            Temp_2 = torch.divide(
                                Temp_loop_var,
                                torch.add(
                                    torch.multiply(
                                        torch.tensor(2.0, dtype=torch.float32, device=xb.device),
                                        torch.pow(Initial_BandWidth[int(current_y_true_label)], 2)
                                    ),
                                    torch.tensor(1e-23, dtype=torch.float32, device=xb.device)
                                )
                            )
            
            Temp_2 = torch.sum(Temp_2, dim=1)
            Temp_2 = Temp_2.reshape((Temp_2.shape[0], 1))
            Temp_2 = torch.exp(torch.multiply(torch.tensor(-1.0, device=xb.device), Temp_2))

            Coefficient = torch.tensor(1.0, device=xb.device)
            Temp_2 = torch.multiply(Coefficient, Temp_2)
            Temp_2 = torch.sum(Temp_2)

            if X_Temp.shape[0] > 0:
                Prior_X[i] *= (Temp_2 / X_Temp.shape[0])

    observation_loss[mask] = 0.0
    observation_loss = torch.sum(observation_loss, dim=1)
    observation_loss = torch.multiply(Prior_X[:, 0], observation_loss)
    
    if inverse_propensities_list is not None:
        observation_loss = torch.multiply(observation_loss, inverse_propensities_list.to(observation_loss.device))

    return torch.mean(observation_loss)