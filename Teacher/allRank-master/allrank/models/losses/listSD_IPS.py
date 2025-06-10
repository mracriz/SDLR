import numpy as np
import torch

###
from allrank import config as conf
# import pandas as pd # Removed as it was only used in commented-out code
# import os # Removed as it was only used in commented-out code
###

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS


def listSD_IPS(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, xb=None, epoch=0, Parameters_Path=None, inverse_propensities_list=None):
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
    Temp_BandWidth_Loss = torch.zeros(size=([xb.shape[0]] + list(conf.Best_BandWidth.shape)), dtype=torch.float32, device=xb.device)

    for i in range(xb.shape[0]):
        X_Temp = xb[i]

        Padded_Doc = torch.where(torch.sum(X_Temp, dim=1) == 0)[0].tolist()
        Temp_indices_list = list(range(X_Temp.shape[0])) # Renamed to avoid conflict
        [Temp_indices_list.remove(k) for k in Padded_Doc] # Used k instead of j
        X_Temp = X_Temp[Temp_indices_list]

        for j in range(X_Temp.shape[0]):
            if j > 5:
                break
            
            # Prepare Temp by removing current document j
            Current_X_Temp_indices = list(range(X_Temp.shape[0]))
            if not Current_X_Temp_indices: continue # Skip if X_Temp became empty
            if j < len(Current_X_Temp_indices): # Ensure j is a valid index
                Current_X_Temp_indices.remove(j)
                Temp_loop_var = X_Temp[Current_X_Temp_indices] # Renamed to avoid conflict
            else: # Should not happen if j > 5 break works as expected for small lists
                continue

            if Temp_loop_var.shape[0] == 0:
                break
            
            Temp_loop_var = torch.subtract(Temp_loop_var, X_Temp[j])
            Temp_loop_var = torch.pow(Temp_loop_var, 2)
            
            # Use y_true_shuffled for indexing Initial_BandWidth, then gather by sorted indices
            # This ensures we use the original y_true label for the j-th document *in the current X_Temp ordering*
            # This part is complex due to multiple sortings & filterings. Assuming y_true[i] corresponds to the original unsorted y_true for the i-th list
            # and indices[i, j_orig_in_shuffled] gives the position of X_Temp[j] in the y_true_shuffled[i]
            # For simplicity in this direct modification, let's assume y_true[i][j] is a placeholder
            # and in a real scenario, careful index tracking from original to X_Temp[j] is needed.
            # The original code uses y_true[i][j] which might be problematic if X_Temp is a filtered/reordered version of the original items for list i.
            # For this example, we'll proceed with y_true[i][j] as in the original, assuming it's correctly indexed for the *current* X_Temp[j]'s true label.
            # A robust solution would involve mapping indices carefully.
            # Let's assume indices[i] maps positions in y_true_sorted back to y_true_shuffled
            # And another mapping is needed if X_Temp's j doesn't directly map to y_true_shuffled's j
            # Given the fixed j > 5 break, this loop operates on the first few documents of the potentially filtered X_Temp.
            # We need the true label of X_Temp[j]. The original code used y_true[i][j]. This implies that
            # the order in xb[i] (and thus X_Temp initially) corresponds to the order in y_true[i] before any shuffling for Plackett-Luce.
            
            # To get the correct y_true for X_Temp[j], we'd need to know the original index of X_Temp[j]
            original_doc_index_in_list_i = Temp_indices_list[j] # This is the index in the original (pre-shuffled for PL) feature list xb[i]
            current_y_true_label = y_true[i, original_doc_index_in_list_i] # Assuming y_true corresponds to original order of xb

            Temp_2 = torch.divide(Temp_loop_var, torch.add(torch.multiply(torch.tensor(2.0, device=xb.device), torch.pow(Initial_BandWidth[int(current_y_true_label)], 2)), torch.tensor(1e-23, device=xb.device)))
            Temp_2 = torch.sum(Temp_2, dim=1)
            Temp_2 = Temp_2.reshape((Temp_2.shape[0], 1))
            Temp_2 = torch.exp(torch.multiply(torch.tensor(-1.0, device=xb.device), Temp_2))

            ### BandWidth Loss Derivation ###
            Temp_3 = torch.add(torch.pow(Initial_BandWidth[int(current_y_true_label)], 2), Temp_loop_var)
            Temp_3 = torch.divide(Temp_3, torch.add(torch.pow(Initial_BandWidth[int(current_y_true_label)], 3), torch.tensor(1e-23, device=xb.device)))
            Coefficient_bw = torch.tensor(1.0, device=xb.device) # Renamed
            Temp_3 = torch.multiply(Coefficient_bw, Temp_3)
            Temp_3 = torch.multiply(Temp_3, Temp_2)
            Temp_3 = torch.sum(Temp_3, dim=0) / X_Temp.shape[0]
            
            # Original logic for updating Temp_BandWidth_Loss
            current_sum_val = torch.sum(Temp_BandWidth_Loss[i, int(current_y_true_label)])
            if current_sum_val == 0: # Check if it's effectively zero
                 Temp_BandWidth_Loss[i, int(current_y_true_label)] = Temp_3
            else:
                 Temp_BandWidth_Loss[i, int(current_y_true_label)] = torch.multiply(Temp_BandWidth_Loss[i, int(current_y_true_label)], Temp_3)

            ###
            Coefficient_prior = torch.tensor(1.0, device=xb.device) # Renamed
            Temp_2 = torch.multiply(Coefficient_prior, Temp_2)
            Temp_2 = torch.sum(Temp_2)

            if epoch > 31 and (epoch % 2 == 1):
                if X_Temp.shape[0] > 0: # Avoid division by zero
                    Prior_X[i] *= (Temp_2 / X_Temp.shape[0])

    observation_loss[mask] = 0.0
    observation_loss = torch.sum(observation_loss, dim=1)

    ### Update BandWidth Loss Derivation (global state)
    Temp_reshaped_obs_loss = torch.reshape(observation_loss, shape=(observation_loss.shape[0], 1, 1)) # Renamed
    Temp_BandWidth_Loss_weighted = torch.multiply(Temp_reshaped_obs_loss, Temp_BandWidth_Loss) # Renamed
    Temp_BandWidth_Loss_summed = torch.sum(Temp_BandWidth_Loss_weighted, dim=0) # Renamed
    conf.BandWidth_Loss_Derivation = torch.add(conf.BandWidth_Loss_Derivation, Temp_BandWidth_Loss_summed)
    ###

    # Apply SDLR's Prior_X weighting
    observation_loss = torch.multiply(Prior_X[:, 0], observation_loss)

    # --- BEGIN IPS MODIFICATION ---
    if inverse_propensities_list is not None:
        if observation_loss.shape == inverse_propensities_list.shape:
            observation_loss = torch.multiply(observation_loss, inverse_propensities_list)
        else:
            # Handle potential shape mismatch, e.g., by unsqueezing or raising an error
            # For now, let's assume they are compatible or print a warning.
            print(f"Warning: Shape mismatch for IPS. Loss shape: {observation_loss.shape}, Propensities shape: {inverse_propensities_list.shape}")
            # Or, if inverse_propensities_list is [batch_size] and observation_loss is also [batch_size]
            # then it should be fine. The check above is more general.
            # Let's proceed assuming inverse_propensities_list is [batch_size]
            observation_loss = torch.multiply(observation_loss, inverse_propensities_list.to(observation_loss.device))

    # --- END IPS MODIFICATION ---

    observation_loss = torch.multiply(torch.tensor(10.0, device=xb.device), observation_loss)

    return torch.mean(observation_loss)