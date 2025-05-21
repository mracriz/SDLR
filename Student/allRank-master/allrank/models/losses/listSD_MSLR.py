import numpy as np
import torch

###
from allrank import config as conf
import pandas as pd
import os
###

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS


def listSD_MSLR(y_pred, y_true, eps = DEFAULT_EPS, padded_value_indicator = PADDED_Y_VALUE, xb=None, epoch = 0, Parameters_Path = None):
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

    """
    Unique_Labels = torch.sort(torch.unique(y_true)).values
    Unique_Labels = Unique_Labels[1:]
    Unique_Labels = torch.tensor([0, 1, 2, 3, 4])
    Initial_BandWidth = torch.ones(size=(Unique_Labels.shape[0], xb.shape[-1])).type(torch.float64) * 0.7071
    """
    #Initial_BandWidth = torch.clone(conf.BandWidth).to(torch.device("cuda:0"))
    Initial_BandWidth = torch.clone(conf.Best_BandWidth).to(torch.device("cuda:0"))
    Prior_X = torch.ones(size=(xb.shape[0], 1)).type(torch.float64).to(torch.device("cuda:0"))
    Temp_BandWidth_Loss = torch.zeros(size=([xb.shape[0]] + list(conf.BandWidth.shape))).type(torch.float64).to(torch.device("cuda:0"))
    for i in range(xb.shape[0]):
        X_Temp = xb[i]

        Padded_Doc = torch.where(torch.sum(X_Temp, dim = 1) == 0)[0].tolist()
        Temp = list(range(X_Temp.shape[0]))
        [Temp.remove(j) for j in Padded_Doc]
        X_Temp = X_Temp[Temp]

        for j in range(X_Temp.shape[0]):
            if j > 3: break
            Temp = list(range(X_Temp.shape[0]))
            Temp.remove(j)
            Temp = X_Temp[Temp]
            if Temp.shape[0] == 0:
                break
            #Temp = torch.Tensor(Temp)
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
            Temp_2 = torch.divide(Temp, torch.add(torch.multiply(torch.tensor(2), torch.pow(Initial_BandWidth[int(y_true[i][j])], 2)), torch.tensor(1e-23)))
            # print("\nTemp_2:", Temp_2.shape, "\n", Temp_2)
            Temp_2 = torch.sum(Temp_2, dim=1)
            Temp_2 = Temp_2.reshape((Temp_2.shape[0], 1))
            Temp_2 = torch.exp(torch.multiply(torch.tensor(-1), Temp_2))

            ### BandWidth Loss Derivation ###
            Temp_3 = torch.add(torch.pow(Initial_BandWidth[int(y_true[i][j])], 2), Temp)
            Temp_3 = torch.divide(Temp_3, torch.add(torch.pow(Initial_BandWidth[int(y_true[i][j])], 3), torch.tensor(1e-23)))
            #Coefficient = torch.divide(1, torch.add(torch.multiply(torch.sqrt(torch.tensor(2.0) * torch.pi), torch.prod(Initial_BandWidth[int(y_true[i][j])])), 1e-23))
            Coefficient = torch.tensor(1.0)
            Temp_3 = torch.multiply(Coefficient, Temp_3)
            Temp_3 = torch.multiply(Temp_3, Temp_2)
            Temp_3 = torch.sum(Temp_3, dim = 0) / X_Temp.shape[0]
            #Temp_3 = Temp_3.to(torch.device("cuda:0"))
            #Temp_3 = torch.multiply(Temp_3, observation_loss[i,torch.where(indices[i] == j)[0]])
            ##Temp_3 = torch.multiply(Temp_3, observation_loss[i, j])
            ##Temp_BandWidth_Loss[int(y_true[i][j])] = torch.multiply(Temp_BandWidth_Loss[int(y_true[i][j])], Temp_3)
            Temp_BandWidth_Loss[i, int(y_true[i][j])] = torch.multiply(Temp_BandWidth_Loss[i, int(y_true[i][j])], Temp_3)
            if torch.sum(Temp_BandWidth_Loss[i, int(y_true[i][j])]) == 0:
                Temp_BandWidth_Loss[i, int(y_true[i][j])] = Temp_3
            ###

            #Coefficient = torch.divide(torch.tensor(1.0), torch.add(torch.multiply(torch.sqrt(torch.tensor(2.0) * torch.pi), torch.prod(Initial_BandWidth[int(y_true[i][j])])), torch.tensor(1e-23)))
            Coefficient = torch.tensor(10.0)

            #Coefficient = torch.divide(torch.tensor(1.0), torch.add(torch.multiply(torch.sqrt(torch.tensor(2.0) * torch.pi), torch.prod(Initial_BandWidth[int(y_true[i][j])][:1])), torch.tensor(1e-23)))
            Temp_2 = torch.multiply(Coefficient, Temp_2)  # Temp_3
            Temp_2 = torch.sum(Temp_2)
            #print()
            # print("\nTemp_2:", Temp_2.shape, "\n", Temp_2)
            # print("\nTemp_3:", np.multiply(Temp_2, Temp))

            # print("\nDivide:", Temp_2)

            #if epoch > 31: # True: #j < 6:
            if epoch > 31 and (epoch % 2 == 1):
                Prior_X[i] *= (Temp_2 / X_Temp.shape[0])
                #Prior_X[i] *= 100
        """
        print("---\n---")
        print(i, Prior_X[i])
        print("---\n---")
        input("SS")
        """
        ##conf.BandWidth_Loss_Derivation = torch.add(conf.BandWidth_Loss_Derivation, Temp_BandWidth_Loss)
        ###TT = torch.clone(conf.BandWidth_Loss_Derivation)
    """
    # Prior Calculation
    ###AA = preds_sorted_by_true_minus_max.exp().flip(dims=[1])
    ###BB = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1)
    Unique_Labels = torch.sort(torch.unique(y_true)).values[1:]
    Sigma_Label_Based = []
    Mean_Label_Based = []
    Prior_Loss = 0
    XX = torch.clone(xb)
    PX_XX = torch.clone(xb)
    for i in range(len(Unique_Labels)):
        ###print(AA[i], ":", torch.where(y_true == AA[i])[1])
        Selected_Indices = torch.where(y_true == Unique_Labels[i])
        Temp = xb[list(Selected_Indices[0].numpy()), list(Selected_Indices[1].numpy()), :]
        Temp = torch.abs(Temp)
        Temp_Mean = torch.mean(Temp, dim = 0)
        Mean_Label_Based += [Temp_Mean.tolist()]
        Temp_Sigma = (torch.max(Temp, dim = 0).values - torch.min(Temp, dim=0).values) / 6.0
        Temp_Sigma[torch.where(Temp_Sigma == 0)[0].tolist()] = 1
        Sigma_Label_Based += [Temp_Sigma.tolist()]
        Coefficient = 1 # torch.pow((1 / ( 2 * torch.pi * Temp_Sigma)), (len(Selected_Indices) / 2) )
        Temp[torch.where(Temp == 0)] = float("-inf")
        Temp = torch.multiply(Coefficient, torch.exp(-1 * (torch.pow(torch.subtract(Temp, Temp_Mean), 2) / (2 * Temp_Sigma))))

        ###
        PX_XX[list(Selected_Indices[0].numpy()), list(Selected_Indices[1].numpy()), :] = torch.multiply(Temp, PX_XX[list(Selected_Indices[0].numpy()), list(Selected_Indices[1].numpy()), :])
        ###

        Temp = torch.log(torch.norm(Temp, dim=1) + eps)
        Prior_Loss += torch.sum(Temp)
        ##observation_loss[:, Selected_Indices] -= Prior_Loss
        #print()
    Prior_Loss /= xb.shape[0] # Mean Of Priors
    ###input("S")
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
    XX = pd.DataFrame(XX[0])
    XX.to_csv("./Parameters/One/XX_" + str(epoch).zfill(4) + ".csv")
    PX_XX = pd.DataFrame(PX_XX[0])
    PX_XX.to_csv("./Parameters/One/PX_XX_" + str(epoch).zfill(4) + ".csv")

    Mean_Label_Based = pd.DataFrame(Mean_Label_Based, index = Unique_Labels.to(torch.int8).tolist())
    Mean_Label_Based.to_csv("./Parameters/One/Mean_" + str(epoch).zfill(4) + ".csv")
    Sigma_Label_Based = pd.DataFrame(Sigma_Label_Based, index = Unique_Labels.to(torch.int8).tolist())
    Sigma_Label_Based.to_csv("./Parameters/One/Sigma_" + str(epoch).zfill(4) + ".csv")
    """

    observation_loss[mask] = 0.0

    observation_loss = torch.sum(observation_loss, dim=1)

    ###
    Temp = torch.reshape(observation_loss, shape = (observation_loss.shape[0], 1, 1))
    Temp_BandWidth_Loss = torch.multiply(Temp, Temp_BandWidth_Loss[:, :, :])
    Temp_BandWidth_Loss = torch.sum(Temp_BandWidth_Loss, dim = 0)
    conf.BandWidth_Loss_Derivation = torch.add(conf.BandWidth_Loss_Derivation, Temp_BandWidth_Loss)
    ###
    #print("///////////////////////", torch.max(Prior_X[:, 0]), "///////////////////////")
    observation_loss = torch.multiply(Prior_X[:, 0], observation_loss)
    #observation_loss = torch.multiply(xb.shape[-1] * 1, observation_loss) # 100

    return torch.mean(observation_loss)