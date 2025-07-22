import numpy as np
import torch
from pathlib import Path
import pandas as pd
import os

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS

# Supondo que este é o seu arquivo de perda IPS do Student
def listSDStu_IPS(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, xb=None, epoch=0, Parameters_Path=None, inverse_propensities_list=None):
    """
    Função de perda listSD Student para IPS, corrigida para aceitar e usar os pesos de propensão inversa.
    """
    # shuffle para resolução de empates aleatória
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

    # --- ESTA É A CORREÇÃO LÓGICA ---
    # Se os pesos de propensão inversa (IPS) forem fornecidos, aplique-os.
    # Esta é a maneira padrão de usar pesos IPS em uma função de perda.
    if inverse_propensities_list is not None:
        observation_loss *= inverse_propensities_list.unsqueeze(1)
    # --- FIM DA CORREÇÃO ---

    # Constrói o caminho robusto para o arquivo de parâmetros do Teacher
    project_root = Path(os.getcwd()).parent.parent
    target_file_in_teacher = project_root / "Teacher" / "allRank-master" / "allrank" / "results" / "teacher_run" / "parameters" / "Sigma_All_Score_Best.csv"
    
    try:
        df_bandwidth = pd.read_csv(target_file_in_teacher, index_col=0)
    except FileNotFoundError:
        print(f"ERRO FATAL em listSDStu_IPS: Não foi possível encontrar o arquivo de bandwidth do Teacher.")
        print(f"Caminho tentado: {target_file_in_teacher}")
        raise

    # ... (o resto da sua lógica de cálculo de perda com o bandwidth continua aqui)
    initial_bandwidth_np = df_bandwidth.to_numpy()
    initial_bandwidth_np[np.where(initial_bandwidth_np < 1e-11)] = 1
    Initial_BandWidth = torch.tensor(initial_bandwidth_np, dtype=torch.float32, device=xb.device)

    Prior_X = torch.ones(size=(xb.shape[0], 1), dtype=torch.float32, device=xb.device)

    for i in range(xb.shape[0]):
        X_Temp = xb[i]

        Padded_Doc = torch.where(torch.sum(X_Temp, dim=1) == 0)[0].tolist()
        Temp_docs = list(range(X_Temp.shape[0]))
        [Temp_docs.remove(j) for j in Padded_Doc]
        X_Temp = X_Temp[Temp_docs]

        for j in range(X_Temp.shape[0]):
            if j > 10: break

            current_label = y_true[i, j]
            if current_label == 0:
                continue

            bandwidth_index = int(current_label) - 1
            current_bandwidth_row = Initial_BandWidth[bandwidth_index]

            if X_Temp.shape[1] != current_bandwidth_row.shape[0]:
                num_features_in_bandwidth = current_bandwidth_row.shape[0]
                X_Temp_sliced = X_Temp[:, :num_features_in_bandwidth]
            else:
                X_Temp_sliced = X_Temp

            Temp = list(range(X_Temp_sliced.shape[0]))
            Temp.remove(j)
            Temp = X_Temp_sliced[Temp]
            if Temp.shape[0] == 0:
                break
            
            Temp = torch.subtract(Temp, X_Temp_sliced[j])
            Temp = torch.pow(Temp, 2)
            
            Temp_2 = torch.divide(
                                Temp,
                                torch.add(
                                    torch.multiply(
                                        torch.tensor(2.0, dtype=torch.float32, device=xb.device),
                                        torch.pow(current_bandwidth_row, 2)
                                    ),
                                    torch.tensor(1e-23, dtype=torch.float32, device=xb.device)
                                )
                            )
            Temp_2 = torch.sum(Temp_2, dim=1)
            Temp_2 = Temp_2.reshape((Temp_2.shape[0], 1))
            Temp_2 = torch.exp(torch.multiply(torch.tensor(-1), Temp_2))

            Coefficient = torch.tensor(1.0)
            Temp_2 = torch.multiply(Coefficient, Temp_2)
            Temp_2 = torch.sum(Temp_2)
            Prior_X[i] *= (Temp_2 / X_Temp_sliced.shape[0])
            
    observation_loss[mask] = 0.0
    observation_loss = torch.sum(observation_loss, dim=1)
    observation_loss = torch.multiply(Prior_X[:, 0], observation_loss)

    return torch.mean(observation_loss)