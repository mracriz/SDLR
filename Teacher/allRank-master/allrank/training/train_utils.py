import os
from functools import partial
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from csv import DictWriter
import allrank.models.metrics as metrics_module
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_num_params, log_num_params
from allrank.training.early_stop import EarlyStop
from allrank.utils.ltr_logging import get_logger
from allrank.utils.tensorboard_utils import TensorboardSummaryWriter
from allrank.training.Compute_BandWidth_torch import Compute_BandWidth
from allrank import config

logger = get_logger()


def loss_batch(model, loss_func, xb, yb, indices, gradient_clipping_norm, opt=None):
    mask = (yb == PADDED_Y_VALUE)
    """pxb = torch.abs(xb)

    sigma = torch.max(pxb, dim=0).values - torch.min(pxb, dim=0).values
    sigma = torch.sign(sigma)

    prior6 = torch.multiply(1 / (sigma + 1e-10),
                            torch.exp(((pxb - torch.min(pxb)) / (torch.max(pxb) - torch.min(pxb))) ))

    pxb_xb = torch.multiply(prior6,xb)
    loss = loss_func(model(pxb_xb, mask, indices), yb, xb=xb)#, 10)
    """
    loss = loss_func(model(xb, mask, indices), yb, xb=xb)#, 10)

    if opt is not None:
        loss.backward()
        if gradient_clipping_norm:
            clip_grad_norm_(model.parameters(), gradient_clipping_norm)
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


# def metric_on_batch(path,name,epoch,epochs,flag,metric, model, xb, yb, indices):
#     #xb = feature, yb= lable
#     d=0
#     mask = (yb == PADDED_Y_VALUE)
#     if flag=="test":
#         if epoch == epochs-1:
#             d = 1
#     return metric(path,name,d,model.score(xb, mask, indices), yb)

def metric_on_batch(path,name,epoch,epochs,flag,metric, model, xb, yb, indices):
    d=0
    mask = (yb == PADDED_Y_VALUE) # Assuming PADDED_Y_VALUE is defined
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

    ###
    config.data.slate_length = int(np.ceil(max([train_dl.dataset.longest_query_length, valid_dl.dataset.longest_query_length]) / 10) * 10)
    ###
    loss_func.keywords["Parameters_Path"] = config.data.path[config.data.path.rfind("Dataset/") + 8:].replace("/", "_")
    #
    """
    # Calculating Sigma Besed On KDE
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    with torch.no_grad():
        X_Evaluation = []
        Y_Evaluation = []
        Indices_Evaluation = []
        i = 0
        for xb, yb, indices in train_dl:
            i += 1
            X_Evaluation = torch.concat((X_Evaluation, xb), dim=0) if len(X_Evaluation) > 0 else xb
            Y_Evaluation = torch.concat((Y_Evaluation, yb), dim=0) if len(Y_Evaluation) > 0 else yb
            Indices_Evaluation = torch.concat((Indices_Evaluation, indices), dim=0) if len(Indices_Evaluation) > 0 else indices
            if (i + 1) % 23 == 0: print(i, X_Evaluation.shape)
            torch.cuda.empty_cache()
            gc.collect()

        X_Evaluation = X_Evaluation[: int(0.5 * X_Evaluation.shape[0])]
        Y_Evaluation = Y_Evaluation[: int(0.5 * Y_Evaluation.shape[0])]
        Indices_Evaluation = Indices_Evaluation[: int(0.5 * Indices_Evaluation.shape[0])]
        Sigma_Label_Based = Compute_BandWidth(X_Evaluation, Y_Evaluation, Iteration = 23, Stop_Threshold = 71e-2)


    import pandas as pd
    #Mean_Label_Based = pd.DataFrame(Mean_Label_Based, index = [0]).transpose()
    #Mean_Label_Based.to_csv("./Parameters/One/Mean_All_Score_" + loss_func.keywords["Parameters_Path"] + ".csv")
    Sigma_Label_Based = pd.DataFrame(Sigma_Label_Based, dtype = np.float64).transpose()
    Sigma_Label_Based.to_csv("./Parameters/One/Sigma_All_Score_" + loss_func.keywords["Parameters_Path"] + ".csv")
    print()
    return None
    """
    #
    from allrank import config as conf
    #conf.BandWidth_LR = torch.tensor(conf.BandWidth_LR).to(torch.device("cuda:0"))
    conf.BandWidth_LR = torch.tensor(conf.BandWidth_LR, dtype=torch.float32).to(device)

    Temp = []
    for xb, yb, indices in train_dl:
        Temp += torch.unique(yb).tolist()
    Unique_Labels = torch.sort(torch.unique(torch.tensor(Temp))).values
    Unique_Labels = Unique_Labels[1:]
    conf.BandWidth = torch.ones(size=(Unique_Labels.shape[0], xb.shape[-1])).type(torch.float64)
    Temp_Coefficient = torch.ones(Unique_Labels.shape[0], 1)
    Temp_Coefficient[0] = torch.multiply(Temp_Coefficient[0], torch.tensor(0.71)) #2.9  20
    Temp_Coefficient[1:] = torch.multiply(Temp_Coefficient[1:], torch.tensor(1.0)) #10.0  30
    #Temp_Coefficient = torch.tensor([1, 10, 10]).reshape(-1, 1)
    conf.BandWidth = torch.multiply(conf.BandWidth, Temp_Coefficient)
    ###conf.BandWidth = torch.rand(size=(Unique_Labels.shape[0], xb.shape[-1]), dtype = torch.float64) * 20
    #conf.BandWidth = conf.BandWidth.to(torch.device("cuda:0"))
    conf.BandWidth = conf.BandWidth.to(device)

    conf.Best_BandWidth = torch.clone(conf.BandWidth)
    #
    for epoch in range(epochs):
        logger.info("Current learning rate: {}".format(get_current_lr(optimizer)))

        ###
        loss_func.keywords["epoch"] = epoch
        ###
        """
        print("----------------------- ----------------------- -----------------------")
        print(conf.BandWidth[0])
        print("----------------------- ----------------------- -----------------------")
        """
        #conf.BandWidth_Loss_Derivation = torch.zeros(size=(conf.BandWidth.shape[:2])).type(torch.float64).to(torch.device("cuda:0"))
        conf.BandWidth_Loss_Derivation = torch.zeros(size=(conf.BandWidth.shape[:2]), dtype=torch.float32).to(device)
        ###

        model.train()
        # xb dim: [batch_size, slate_length, embedding_dim]
        # yb dim: [batch_size, slate_length]

        train_losses, train_nums = zip(
            *[loss_batch(model, loss_func, xb.to(device=device), yb.to(device=device), indices.to(device=device),
                         gradient_clipping_norm, optimizer) for
              xb, yb, indices in train_dl])
        train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        a_string = config.data.path

        alphanumeric = ""
        for character in a_string:

            if character.isalnum():
                alphanumeric += character
        train_metrics = compute_metrics(alphanumeric,config.loss.name,epoch,epochs,"train",config.metrics, model, train_dl, device)


        model.eval()
        with torch.no_grad():
            val_losses, val_nums = zip(
                *[loss_batch(model, loss_func, xb.to(device=device), yb.to(device=device), indices.to(device=device),
                             gradient_clipping_norm) for
                  xb, yb, indices in valid_dl])
            val_metrics = compute_metrics(alphanumeric,config.loss.name,epoch,epochs,"test",config.metrics, model, valid_dl, device)

        val_loss = np.sum(np.multiply(val_losses, val_nums)) / np.sum(val_nums)

        ###

        conf.BandWidth_Changes += [torch.mean(torch.multiply(conf.BandWidth_LR, conf.BandWidth_Loss_Derivation))]
        #conf.BandWidth = torch.subtract(conf.BandWidth, torch.multiply(conf.BandWidth_LR, conf.BandWidth_Loss_Derivation))
        Temp_Update = torch.absolute(torch.subtract(conf.BandWidth, torch.multiply(conf.BandWidth_LR , conf.BandWidth_Loss_Derivation) ))
        ##Temp_Update_Indices = (Temp_Update < conf.BandWidth).to(torch.int8)
        Temp_Update_Indices = torch.add((Temp_Update < conf.BandWidth).to(torch.int8), (Temp_Update < 0.3).to(torch.int8))
        Temp_Update_Indices = (Temp_Update_Indices > 0.9).to(torch.int8)
        conf.BandWidth = torch.add(torch.multiply(torch.subtract(torch.tensor(1.0), Temp_Update_Indices), conf.BandWidth), torch.multiply(Temp_Update_Indices, Temp_Update))
        #conf.BandWidth = torch.absolute(conf.BandWidth)
        if not os.path.isdir("./Parameters/One/" + loss_func.keywords["Parameters_Path"] + "/") : os.makedirs("./Parameters/One/" + loss_func.keywords["Parameters_Path"] + "/")
        Temp_BandWidth = pd.DataFrame(conf.BandWidth.cpu().numpy())
        Temp_BandWidth.to_csv("./Parameters/One/" + loss_func.keywords["Parameters_Path"] + "/Sigma_All_Score_" + loss_func.keywords["Parameters_Path"] + "_" + str(epoch).zfill(3) + ".csv")

        ###


        tensorboard_metrics_dict = {("train", "loss"): train_loss, ("val", "loss"): val_loss}

        train_metrics_to_tb = {("train", name): value for name, value in train_metrics.items()}
        tensorboard_metrics_dict.update(train_metrics_to_tb)
        val_metrics_to_tb = {("val", name): value for name, value in val_metrics.items()}
        tensorboard_metrics_dict.update(val_metrics_to_tb)
        tensorboard_metrics_dict.update({("train", "lr"): get_current_lr(optimizer)})

        tensorboard_summary_writer.save_to_tensorboard(tensorboard_metrics_dict, epoch)

        logger.info(epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics))
        field_names = ['ndcg_1', 'ndcg_2', 'ndcg_3', 'ndcg_4', 'ndcg_5', 'ndcg_6', 'ndcg_7', 'ndcg_8', 'ndcg_9',
                       'ndcg_10', 'ndcg_20','mrr_1', 'mrr_2', 'mrr_3', 'mrr_4', 'mrr_5', 'mrr_6', 'mrr_7', 'mrr_8', 'mrr_9', 'mrr_10',
                       'mrr_20']

        def append_dict_as_row(file_name, dict_of_elem, field_names):
            # Open file in append mode
            with open(file_name, 'a+', newline='') as write_obj:
                # Create a writer object from csv module
                dict_writer = DictWriter(write_obj, fieldnames=field_names, )
                # Add dictionary as row in the csv
                if iter == 1:
                    dict_writer.writeheader()
                dict_writer.writerow(dict_of_elem)

        if epoch == epochs - 1:
            append_dict_as_row(config.loss.name + alphanumeric+'.csv', val_metrics, field_names)
        current_val_metric_value = val_metrics.get(config.val_metric)
        if scheduler:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                args = [val_metrics[config.val_metric]]
                scheduler.step(*args)
            else:
                scheduler.step()

        Previous_Value = early_stop.best_value

        early_stop.step(current_val_metric_value, epoch, (config.loss.name + alphanumeric + '_Best.csv', val_metrics, field_names))
        ###
        if early_stop.best_value != Previous_Value:
            if os.path.isfile(config.loss.name + alphanumeric + "_Best.csv") : os.remove(config.loss.name + alphanumeric + "_Best.csv")
            if os.path.isfile(config.loss.name + alphanumeric + "_BestMRR.csv") : os.remove(config.loss.name + alphanumeric + "_BestMRR.csv")
            if os.path.isfile(config.loss.name + alphanumeric + "_BestNDCG.csv") : os.remove(config.loss.name + alphanumeric + "_BestNDCG.csv")
            append_dict_as_row(config.loss.name + alphanumeric + '_Best.csv', val_metrics, field_names)
            compute_metrics(alphanumeric + "_Best" , config.loss.name, epochs - 1, epochs, "test", config.metrics, model, valid_dl,device)
            conf.Best_BandWidth = torch.clone(conf.BandWidth)
        ###
        if early_stop.stop_training(epoch):
            append_dict_as_row(config.loss.name + alphanumeric + '.csv', val_metrics, field_names)
            logger.info(
                "early stopping at epoch {} since {} didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, config.val_metric, early_stop.best_epoch, early_stop.best_value, current_val_metric_value
                ))
            break

    ###
    torch.cuda.empty_cache()
    ###
    """
    X_Evaluation = []
    Y_Evaluation = []
    Indices_Evaluation = []
    for xb, yb, indices in train_dl:
        X_Evaluation = torch.concat((X_Evaluation, xb), dim = 0) if len(X_Evaluation) > 0 else xb
        Y_Evaluation = torch.concat((Y_Evaluation, yb), dim = 0) if len(Y_Evaluation) > 0 else yb
        Indices_Evaluation = torch.concat((Indices_Evaluation, indices), dim = 0) if len(Indices_Evaluation) > 0 else indices
    #
    Reshape = [X_Evaluation.shape[0] * X_Evaluation.shape[1]] + list(X_Evaluation.shape[2:])
    X_Evaluation = torch.reshape(X_Evaluation, tuple(Reshape))
    Reshape = [Y_Evaluation.shape[0] * Y_Evaluation.shape[1]] + list(Y_Evaluation.shape[2:])
    Y_Evaluation = torch.reshape(Y_Evaluation, tuple(Reshape))
    Indices_Evaluation = torch.reshape(Indices_Evaluation, tuple(Reshape))
    ###
    print(X_Evaluation.shape)
    input("S.O.S.")
    ###
    Prediction_Count = 2
    Y_Prediction = []
    Counter = torch.tensor((X_Evaluation.shape[0] / Prediction_Count) + 1).type(torch.int32)
    for i in range(Counter):
        Temp_1 = X_Evaluation[i * Prediction_Count: (i + 1) * Prediction_Count]
        Temp_1 = torch.reshape(Temp_1, (1, X_Evaluation[i * Prediction_Count: (i + 1) * Prediction_Count].shape[0], X_Evaluation[i * Prediction_Count: (i + 1) * Prediction_Count].shape[-1]))

        Temp_2 = (Y_Evaluation[i * Prediction_Count: (i + 1) * Prediction_Count]== PADDED_Y_VALUE)
        Temp_2 = torch.reshape(Temp_2, (1, Temp_2.shape[0]))

        Temp_3 = Indices_Evaluation[i * Prediction_Count: (i + 1) * Prediction_Count]
        Temp_3 = torch.reshape(Temp_3, (1, Temp_3.shape[0]))

        Temp_Prediction = model(Temp_1, Temp_2, Temp_3)
        Y_Prediction = torch.concat((Y_Prediction, Temp_Prediction), dim = 1) if len(Y_Prediction) > 0 else Temp_Prediction
        #print(Y_Prediction.shape)
    ###
    Y_Prediction = []
    #Y_Evaluation = []
    Temp_Means = 
    for xb, yb, indices in train_dl:
        Unique_Labels = torch.sort(torch.unique(yb)).values[1:]
        print(Unique_Labels)
        input("S.O.S.")
        Temp_Prediction = model(xb, (yb == PADDED_Y_VALUE), indices)
        Temp_Prediction = torch.reshape(Temp_Prediction, (1, Temp_Prediction.shape[0] * Temp_Prediction.shape[1]))
        Y_Prediction = torch.concat((Y_Prediction, Temp_Prediction), dim=1) if len(Y_Prediction) > 0 else Temp_Prediction
    print(Y_Prediction.shape, "\n Eval:") # , Y_Evaluation.shape
    input("S.O.S.")
    #Y_Prediction = Y_Prediction.cpu()
    print(Y_Prediction.shape)
    Unique_Labels = torch.sort(torch.unique(Y_Evaluation)).values[1:]
    Sigma_Label_Based = []
    Mean_Label_Based = []
    Prior_Loss = 0
    ###Prediction = torch.clone(y_pred)
    for i in range(len(Unique_Labels)):
        ###print(AA[i], ":", torch.where(y_true == AA[i])[1])
        Selected_Indices = torch.where(Y_Evaluation == Unique_Labels[i])
        Temp = Y_Prediction[0, Selected_Indices[0].tolist()]
        Temp_Mean = torch.mean(Temp)  # , dim = 0
        Mean_Label_Based += [Temp_Mean.tolist()]
        # Temp_Sigma = (torch.max(Temp, dim = 0).values - torch.min(Temp, dim=0).values) / 6.0
        Temp_Sigma = (torch.max(Temp) - torch.min(Temp)) / 6.0
        Temp_Sigma = torch.add((Temp_Sigma == 0).type(torch.int8), torch.multiply(Temp_Sigma, (Temp_Sigma != 0).type(torch.int8)))
        Sigma_Label_Based += [Temp_Sigma.tolist()]
        Coefficient = 1  # torch.pow((1 / ( 2 * torch.pi * Temp_Sigma)), (len(Selected_Indices) / 2) )
        Temp[torch.where(Temp == 0)] = float("-inf")
        Temp = torch.multiply(Coefficient, torch.exp(-1 * (torch.pow(torch.subtract(Temp, Temp_Mean), 2) / (2 * Temp_Sigma))))

        ###
        ###PX_XX[list(Selected_Indices[0].numpy()), list(Selected_Indices[1].numpy()), :] = torch.multiply(Temp, PX_XX[list(Selected_Indices[0].numpy()), list(Selected_Indices[1].numpy()), :])
        ###

        Temp = torch.log(torch.add(Temp, 1e-10))
        Prior_Loss += torch.sum(Temp)
        ##observation_loss[:, Selected_Indices] -= Prior_Loss
        # print()
    Prior_Loss /= X_Evaluation.shape[0]  # Mean Of Priors
    ###input("S")
    
    import pandas as pd
    Mean_Label_Based = pd.DataFrame(Mean_Label_Based, index=Unique_Labels.to(torch.int8).tolist())
    Mean_Label_Based.to_csv("./Parameters/One/Mean_All_Score_" + loss_func.keywords["Parameters_Path"] + ".csv")
    Sigma_Label_Based = pd.DataFrame(Sigma_Label_Based, index=Unique_Labels.to(torch.int8).tolist())
    Sigma_Label_Based.to_csv("./Parameters/One/Sigma_All_Score_" + loss_func.keywords["Parameters_Path"] + ".csv")
    ###
    """
    ###
    """
    # Old X Distribution Parameter
    import gc
    Y_Prediction = []
    # Y_Evaluation = []
    Mean_Label_Based = {}
    Sigma_Label_Based = {}
    Data_Count = 0
    for xb, yb, indices in train_dl:
        with torch.no_grad():
            Unique_Labels = torch.sort(torch.unique(yb)).values[1:].tolist()
            Y_Evaluation = torch.reshape(yb, (1, yb.shape[0] * yb.shape[1]))

            Temp_Prediction = model(xb, (yb == PADDED_Y_VALUE), indices)
            Temp_Prediction = torch.reshape(Temp_Prediction, (1, Temp_Prediction.shape[0] * Temp_Prediction.shape[1]))
            #Y_Prediction = torch.concat((Y_Prediction, Temp_Prediction), dim=1) if len(Y_Prediction) > 0 else Temp_Prediction
            ##print(Data_Count)
            for i in range(len(Unique_Labels)):
                Data_Count += Y_Evaluation.shape[1]
                Selected_Indices = torch.where(Y_Evaluation == Unique_Labels[i])
                Temp = Temp_Prediction[0, Selected_Indices[1].tolist()]
                try:
                    Mean_Label_Based[str(Unique_Labels[i])] += torch.sum(Temp)
                except:
                    Mean_Label_Based[str(Unique_Labels[i])] = torch.sum(Temp)
                try:
                    if torch.max(Temp) > Sigma_Label_Based[str(Unique_Labels[i]) + "_Max"]:
                        Sigma_Label_Based[str(Unique_Labels[i]) + "_Max"] = torch.max(Temp)
                    if torch.min(Temp) < Sigma_Label_Based[str(Unique_Labels[i]) + "_Min"]:
                        Sigma_Label_Based[str(Unique_Labels[i]) + "_Min"] = torch.min(Temp)
                except:
                    Sigma_Label_Based[str(Unique_Labels[i]) + "_Max"] = torch.max(Temp)
                    Sigma_Label_Based[str(Unique_Labels[i]) + "_Min"] = torch.min(Temp)

        torch.cuda.empty_cache()
        gc.collect()

    ###print(Data_Count)
    #Unique_Labels = [float(i) for i in list(Mean_Label_Based.keys())]
    #Unique_Labels = torch.Tensor(Unique_Labels).type(torch.float16)
    ###print(Unique_Labels)
    ###print(Sigma_Label_Based)

    Temp = Sigma_Label_Based
    Sigma_Label_Based = {}

    print(Temp)

    for i in Mean_Label_Based.keys():
        Mean_Label_Based[i] = float(Mean_Label_Based[i] / Data_Count )
        Sigma_Label_Based[i] = (Temp[i + "_Max"] - Temp[i + "_Min"]) / 6.0
        Sigma_Label_Based[i] = torch.add((Sigma_Label_Based[i] == 0).type(torch.int8), torch.multiply(Sigma_Label_Based[i], (Sigma_Label_Based[i] != 0).type(torch.int8)))
        Sigma_Label_Based[i] = float(Sigma_Label_Based[i])

    ###print(Mean_Label_Based)
    ###print(Sigma_Label_Based)
    ###input("S.O.S.")
    
    import pandas as pd
    #Mean_Label_Based = pd.DataFrame(Mean_Label_Based, index = [0]).transpose()
    #Mean_Label_Based.to_csv("./Parameters/One/Mean_All_Score_" + loss_func.keywords["Parameters_Path"] + ".csv")
    Sigma_Label_Based = pd.DataFrame(Sigma_Label_Based, index = [0]).transpose()
    Sigma_Label_Based.to_csv("./Parameters/One/Sigma_All_Score_" + loss_func.keywords["Parameters_Path"] + ".csv")
    """
    ###
    Temp_BandWidth = pd.DataFrame(conf.BandWidth.cpu().numpy())
    Temp_BandWidth.to_csv("./Parameters/One/Sigma_All_Score_" + loss_func.keywords["Parameters_Path"] + ".csv")
    Temp_Best_BandWidth = pd.DataFrame(conf.Best_BandWidth.cpu().numpy())
    Temp_Best_BandWidth.to_csv("./Parameters/One/Sigma_All_Score_" + loss_func.keywords["Parameters_Path"] + "_Best.csv")
    Temp = pd.DataFrame(torch.tensor(conf.BandWidth_Changes).cpu().numpy(), columns = ["Changes"])
    Temp.to_csv("./Parameters/One/Sigma_Changes_" + loss_func.keywords["Parameters_Path"] + ".csv")
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pkl"))
    tensorboard_summary_writer.close_all_writers()

    return {
        "epochs": epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "num_params": num_params
    }
