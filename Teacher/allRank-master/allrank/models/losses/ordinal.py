import torch
from torch.nn import BCELoss, Sigmoid

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_torch_device


def with_ordinals(y, n, padded_value_indicator=PADDED_Y_VALUE):
    """
    Função auxiliar para a perda ordinal, transformando os labels de entrada em valores ordinais.
    """
    dev = get_torch_device()
    one_to_n = torch.arange(start=1, end=n + 1, dtype=torch.float, device=dev)
    unsqueezed = y.unsqueeze(2).repeat(1, 1, n)
    mask = unsqueezed == padded_value_indicator
    ordinals = (unsqueezed >= one_to_n).type(torch.float)
    
    # --- ESTA É A CORREÇÃO DEFINITIVA ---
    # Em vez de preencher com -1, preenchemos com 0.0. A máscara garante que
    # esses valores não afetarão o cálculo da perda, e o valor 0.0 satisfaz
    # a verificação da BCELoss (target_val >= 0 && target_val <= 1).
    ordinals[mask] = 0.0
    # --- FIM DA CORREÇÃO ---
    
    return ordinals


def ordinal(y_pred, y_true, n, padded_value_indicator=PADDED_Y_VALUE):
    """
    Função de perda Ordinal, robusta e corrigida.
    """
    device = get_torch_device()

    # Aplica Sigmoid internamente para garantir que as predições sejam probabilidades
    y_pred = Sigmoid()(y_pred.clone())

    y_true = with_ordinals(y_true.clone(), n)

    # A máscara aqui é baseada no y_true original, antes de ser modificado
    mask = y_true == padded_value_indicator

    ls = BCELoss(reduction='none')(y_pred, y_true)
    
    # Zera a perda para os documentos de padding
    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=2)
    
    # Garante que estamos contando apenas as listas que têm pelo menos um documento válido
    valid_slates = torch.sum(y_true != padded_value_indicator, dim=1) > 0
    
    if valid_slates.sum() > 0:
        loss_output = torch.sum(document_loss) / valid_slates.sum()
    else:
        loss_output = torch.tensor(0.0, device=device)

    return loss_output