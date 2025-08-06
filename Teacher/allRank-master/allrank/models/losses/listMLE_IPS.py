import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS

def listMLE_IPS(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, inverse_propensities_list=None):
    """
    Versão IPS da perda ListMLE.
    :param y_pred: predições do modelo, shape [batch_size, slate_length]
    :param y_true: labels verdadeiros, shape [batch_size, slate_length]
    :param eps: valor epsilon para estabilidade numérica
    :param padded_value_indicator: indicador de padding
    :param inverse_propensities_list: pesos de propensão inversa (IPS) por item no batch, shape [batch_size]
    :return: valor da perda, um torch.Tensor
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

    observation_loss[mask] = 0.0

    # Soma a perda para cada lista de documentos
    slate_loss = torch.sum(observation_loss, dim=1)

    if inverse_propensities_list is not None:
        slate_loss = slate_loss * inverse_propensities_list

    return torch.mean(slate_loss)