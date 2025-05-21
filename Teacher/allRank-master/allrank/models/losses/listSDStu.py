import numpy as np
import torch

###
from allrank import config as conf
import pandas as pd
import os
###

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS


def listSDStu(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, xb=None, epoch = 0, Parameters_Path = None):
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

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max # log()

    # Prior Calculation
    ###AA = preds_sorted_by_true_minus_max.exp().flip(dims=[1])
    ###BB = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1)
    ###print(os.getcwd())
    Initial_BandWidth = pd.read_csv(os.getcwd() + "/Parameters/One/Sigma_All_Score_" + str(Parameters_Path) + ".csv", index_col = 0)
    Initial_BandWidth = Initial_BandWidth.to_numpy()
    Initial_BandWidth[np.where(Initial_BandWidth < 1e-11)] = 1
    #print(Sigma_Label_Based.head(5))
    #print("---\n", int(y_true[0, 0]), "\n----")
    ###
    ###
    ###Initial_BandWidth = torch.clone(conf.BandWidth).to(torch.device("cuda:0"))
    #Prior_X = torch.ones(size=(xb.shape[0], 1)).type(torch.float64).to(torch.device("cuda:0"))

    Prior_X = torch.ones(size=(xb.shape[0], 1), dtype=torch.float32, device=xb.device)

    for i in range(xb.shape[0]):
        X_Temp = xb[i]

        Padded_Doc = torch.where(torch.sum(X_Temp, dim = 1) == 0)[0].tolist()
        Temp = list(range(X_Temp.shape[0]))
        [Temp.remove(j) for j in Padded_Doc]
        X_Temp = X_Temp[Temp]

        for j in range(X_Temp.shape[0]):
            if j > 10: break
            Temp = list(range(X_Temp.shape[0]))
            Temp.remove(j)
            Temp = X_Temp[Temp]
            if Temp.shape[0] == 0:
                break
            # Temp = torch.Tensor(Temp)
            Temp = torch.subtract(Temp, X_Temp[j])
            Temp = torch.pow(Temp, 2)
            """
            print(Temp, Temp.device)
            input("\nS")
            print("======================= ======================= =======================")
            print(torch.multiply(torch.tensor(2), torch.pow(Initial_BandWidth[int(y_true[i][j])], 2)))
            print(torch.multiply(torch.tensor(2), torch.pow(Initial_BandWidth[int(y_true[i][j])], 2)).device)
            print(Initial_BandWidth, Initial_BandWidth.device)
            input("\nS")
            """
            #Temp_2 = torch.divide(Temp, torch.add(torch.multiply(torch.tensor(2.0), torch.pow(torch.tensor(Initial_BandWidth[int(y_true[i][j])]).to(torch.device("cuda:0")), 2)),torch.tensor(1e-23)))
            Temp_2 = torch.divide(
                                Temp,
                                torch.add(
                                    torch.multiply(
                                        torch.tensor(2.0, dtype=torch.float32, device=xb.device),
                                        torch.pow(Initial_BandWidth[int(y_true[i][j])], 2)  # Key change here: using the element directly
                                    ),
                                    torch.tensor(1e-23, dtype=torch.float32, device=xb.device)
                                )
                            )
            # print("\nTemp_2:", Temp_2.shape, "\n", Temp_2)
            Temp_2 = torch.sum(Temp_2, dim=1)
            Temp_2 = Temp_2.reshape((Temp_2.shape[0], 1))
            Temp_2 = torch.exp(torch.multiply(torch.tensor(-1), Temp_2))

            # Coefficient = torch.divide(1, torch.add(torch.multiply(torch.sqrt(torch.tensor(2.0) * torch.pi), torch.prod(Initial_BandWidth[int(y_true[i][j])])), 1e-23))
            Coefficient = torch.tensor(1.0)
            Temp_2 = torch.multiply(Coefficient, Temp_2)  # Temp_3
            Temp_2 = torch.sum(Temp_2)
            # print()
            # print("\nTemp_2:", Temp_2.shape, "\n", Temp_2)
            # print("\nTemp_3:", np.multiply(Temp_2, Temp))

            # print("\nDivide:", Temp_2)
            Prior_X[i] *= (Temp_2 / X_Temp.shape[0])
    """
    Mean_Label_Based = pd.read_csv(os.getcwd() + "/Parameters/One/Mean_All_Score_" + str(Parameters_Path) + ".csv", index_col = 0)
    Prior_Loss = 0
    XX = torch.clone(xb)
    PX_XX = torch.clone(xb)
    for i in range(len(Unique_Labels)):
        Selected_Indices = torch.where(y_true == Unique_Labels[i])
        Temp = xb[list(Selected_Indices[0].tolist()), list(Selected_Indices[1].tolist()), :]
        Temp = torch.abs(Temp)
        Temp_Mean = torch.tensor(Mean_Label_Based.iloc[int(Unique_Labels[i]), :].values).to(torch.device("cuda:0"))
        Temp_Sigma = torch.tensor(Sigma_Label_Based.iloc[int(Unique_Labels[i]), :].values).to(torch.device("cuda:0"))

        Coefficient = torch.tensor(1) # torch.pow((1 / ( 2 * torch.pi * Temp_Sigma)), (len(Selected_Indices) / 2) )
        Temp[torch.where(Temp == 0)] = float("-inf")
        Temp = torch.multiply(Coefficient, torch.exp(-1 * (torch.pow(torch.subtract(Temp, Temp_Mean), 2) / (2 * Temp_Sigma))))

        ###
        PX_XX[list(Selected_Indices[0].tolist()), list(Selected_Indices[1].tolist()), :] = torch.multiply(Temp, PX_XX[list(Selected_Indices[0].tolist()), list(Selected_Indices[1].tolist()), :]).type(torch.float32)
        ###

        Temp = torch.log(torch.norm(Temp, dim=1) + eps)
        Prior_Loss += torch.sum(Temp)

    Prior_Loss /= xb.shape[0] # Mean Of Priors
    """
    #print()
    """
    xb = torch.abs(xb)
    sigma = ((torch.max(xb, dim=1).values - torch.min(xb, dim=1).values) / 6.0) + 1e-10
    #prior6 = torch.multiply(1 / (sigma + eps), torch.exp(((xb - torch.min(xb)) / (torch.max(xb) - torch.min(xb))) / (2 * sigma) ))
    prior6 = torch.multiply(1 / (sigma + eps), torch.exp( -1 * (torch.pow(xb, 2) / (2 * sigma)) ) )

    temp = torch.linalg.norm(prior6, dim=-1)
    ###prior7 = torch.cumsum(temp.flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = observation_loss - torch.log(prior6)

    """
    #print()
    """
    if not os.path.isdir("./Parameters/One/"): os.makedirs("./Parameters/One/")
    torch.save(Temp_Sigma, 'Temp_Sigma.pt')
    XX = pd.DataFrame(XX[0].cpu())
    XX.to_csv("./Parameters/One/XX_" + str(epoch).zfill(4) + ".csv")
    PX_XX = pd.DataFrame(PX_XX[0].cpu())
    PX_XX.to_csv("./Parameters/One/PX_XX_" + str(epoch).zfill(4) + ".csv")

    Mean_Label_Based = pd.DataFrame(Mean_Label_Based, index = Unique_Labels.to(torch.int8).tolist())
    Mean_Label_Based.to_csv("./Parameters/One/Mean_" + str(epoch).zfill(4) + ".csv")
    Sigma_Label_Based = pd.DataFrame(Sigma_Label_Based, index = Unique_Labels.to(torch.int8).tolist())
    Sigma_Label_Based.to_csv("./Parameters/One/Sigma_" + str(epoch).zfill(4) + ".csv")
    """
    observation_loss[mask] = 0.0

    observation_loss = torch.sum(observation_loss, dim = 1)

    observation_loss = torch.multiply(Prior_X[:, 0], observation_loss)
    #observation_loss = torch.multiply(xb.shape[-1] * 100, observation_loss)  # 100

    return torch.mean(observation_loss)
