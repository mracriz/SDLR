import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS
from allrank.models.losses.loss_utils import deterministic_neural_sort, sinkhorn_scaling, stochastic_neural_sort
from allrank.models.metrics import dcg
from allrank.models.model_utils import get_torch_device


def neuralNDCG(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, temperature=1., 
               powered_relevancies=True, k=None, stochastic=False, n_samples=32, 
               beta=0.1, log_scores=True):
    """
    Implementação corrigida do NeuralNDCG baseada no paper original:
    "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting"
    
    Fórmula do paper: NeuralNDCG_k(τ)(s,y) = N^(-1)_k * Σ(j=1 to k) (scale(P̂)g(y))_j * d(j)
    
    Args:
        y_pred: scores preditos [batch_size, slate_length] ou [batch_size, slate_length, 1]
        y_true: labels verdadeiros [batch_size, slate_length]
        padded_value_indicator: valor indicando padding
        temperature: parâmetro τ para controlar suavidade da aproximação
        powered_relevancies: se aplica função de ganho 2^x - 1
        k: truncamento no rank k
        stochastic: se usa variante estocástica
        n_samples: número de amostras para variante estocástica
        beta: parâmetro para NeuralSort estocástico
        log_scores: se aplica log nos scores
        
    Returns:
        loss: -NeuralNDCG (negativo porque queremos maximizar)
    """
    device = get_torch_device()
    
    # Handle different input shapes
    if y_pred.dim() == 3:
        if y_pred.shape[-1] == 1:
            # Shape: [batch_size, slate_length, 1] -> [batch_size, slate_length]
            y_pred = y_pred.squeeze(-1)
        else:
            # Shape: [batch_size, slate_length, num_classes] 
            # Take the last dimension as scores (common in classification models)
            # Or sum across classes, or use max - let's try summing
            y_pred = y_pred.sum(dim=-1)
    elif y_pred.dim() != 2:
        raise ValueError(f"y_pred deve ter 2 ou 3 dimensões, recebeu shape: {y_pred.shape}")
    
    batch_size, slate_length = y_pred.shape
    
    # Default k to slate_length
    if k is None:
        k = slate_length
    k = min(k, slate_length)
    
    # Create padding mask
    mask = (y_true == padded_value_indicator)
    
    # Step 1: Compute approximate permutation matrix P̂ using NeuralSort
    if stochastic:
        P_hat = stochastic_neural_sort(
            y_pred.unsqueeze(-1), n_samples=n_samples, tau=temperature, 
            mask=mask, beta=beta, log_scores=log_scores
        )  # [n_samples, batch_size, slate_length, slate_length]
        n_samples_actual = P_hat.shape[0]
    else:
        P_hat = deterministic_neural_sort(
            y_pred.unsqueeze(-1), tau=temperature, mask=mask
        ).unsqueeze(0)  # [1, batch_size, slate_length, slate_length]
        n_samples_actual = 1
    
    # Step 2: Apply Sinkhorn scaling to make P̂ doubly stochastic
    # Reshape for sinkhorn: [n_samples * batch_size, slate_length, slate_length]
    P_hat_reshaped = P_hat.view(n_samples_actual * batch_size, slate_length, slate_length)
    mask_repeated = mask.repeat(n_samples_actual, 1)
    
    P_hat_scaled = sinkhorn_scaling(P_hat_reshaped, mask_repeated, tol=1e-6, max_iter=50)
    P_hat = P_hat_scaled.view(n_samples_actual, batch_size, slate_length, slate_length)
    
    # Step 3: Apply gain function g(y) to true labels
    y_true_clean = y_true.masked_fill(mask, 0.0)
    if powered_relevancies:
        gains = torch.pow(2.0, y_true_clean) - 1.0  # g(y) = 2^y - 1
    else:
        gains = y_true_clean  # g(y) = y
    
    # Step 4: Compute quasi-sorted gains: scale(P̂) * g(y)
    # P̂ @ g(y): [n_samples, batch_size, slate_length, 1] -> [n_samples, batch_size, slate_length]
    gains_expanded = gains.unsqueeze(0).unsqueeze(-1)  # [1, batch_size, slate_length, 1]
    quasi_sorted_gains = torch.matmul(P_hat, gains_expanded).squeeze(-1)
    
    # Step 5: Apply discount function d(j) = 1/log2(j+2)
    positions = torch.arange(1, slate_length + 1, dtype=torch.float32, device=device)
    discounts = 1.0 / torch.log2(positions + 1.0)  # d(j) = 1/log2(j+1) onde j começa em 1
    
    # Step 6: Truncate to top-k positions for NDCG@k
    quasi_sorted_gains_k = quasi_sorted_gains[:, :, :k]
    discounts_k = discounts[:k]
    
    # Step 7: Compute DCG = Σ(j=1 to k) quasi_sorted_gains_j * d(j)
    dcg_values = torch.sum(quasi_sorted_gains_k * discounts_k.unsqueeze(0).unsqueeze(0), dim=-1)
    
    # Step 8: Compute IDCG (Ideal DCG) using original labels
    # Sort true gains in descending order and apply same discounts
    if powered_relevancies:
        ideal_gains = torch.pow(2.0, y_true_clean) - 1.0
    else:
        ideal_gains = y_true_clean
    
    # Sort gains in descending order for each query
    ideal_gains_sorted, _ = torch.sort(ideal_gains, descending=True)
    ideal_gains_k = ideal_gains_sorted[:, :k]
    
    # Compute IDCG
    idcg = torch.sum(ideal_gains_k * discounts_k.unsqueeze(0), dim=-1)
    
    # Step 9: Normalize by IDCG to get NDCG
    ndcg = dcg_values / (idcg.unsqueeze(0) + DEFAULT_EPS)
    
    # Step 10: Handle queries with IDCG = 0 (all labels are 0)
    idcg_mask = (idcg == 0.0)
    ndcg = ndcg.masked_fill(idcg_mask.unsqueeze(0), 0.0)
    
    # Verify all NDCG values are valid
    assert torch.all(ndcg >= 0.0), "Todos os valores de NDCG devem ser não-negativos"
    assert torch.all(ndcg <= 1.0 + 1e-6), "Todos os valores de NDCG devem ser <= 1"
    
    # Step 11: Compute mean over samples and valid queries
    if idcg_mask.all():
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    valid_queries = ~idcg_mask
    mean_ndcg = ndcg[:, valid_queries].mean()
    
    return -mean_ndcg  # Return negative because we want to maximize NDCG

def neuralNDCG_transposed(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, 
                         temperature=1., powered_relevancies=True, k=None, 
                         stochastic=False, n_samples=32, beta=0.1, log_scores=True,
                         max_iter=50, tol=1e-6):
    """
    Implementação corrigida do NeuralNDCG Transposed baseada no paper:
    
    Fórmula do paper: NeuralNDCG^T_k(τ)(s,y) = N^(-1)_k * Σ(i=1 to n) g(y_i) * (scale(P̂^T) * d)_i
    
    Diferença: soma sobre documentos em vez de posições.
    """
    device = get_torch_device()
    
    # Handle different input shapes
    if y_pred.dim() == 3:
        if y_pred.shape[-1] == 1:
            # Shape: [batch_size, slate_length, 1] -> [batch_size, slate_length]
            y_pred = y_pred.squeeze(-1)
        else:
            # Shape: [batch_size, slate_length, num_classes] 
            # Take the last dimension as scores (common in classification models)
            # Or sum across classes, or use max - let's try summing
            y_pred = y_pred.sum(dim=-1)
    elif y_pred.dim() != 2:
        raise ValueError(f"y_pred deve ter 2 ou 3 dimensões, recebeu shape: {y_pred.shape}")
        
    batch_size, slate_length = y_pred.shape
    
    # Default k to slate_length
    if k is None:
        k = slate_length
    k = min(k, slate_length)
    
    # Create padding mask
    mask = (y_true == padded_value_indicator)
    
    # Step 1: Compute approximate permutation matrix P̂
    if stochastic:
        P_hat = stochastic_neural_sort(
            y_pred.unsqueeze(-1), n_samples=n_samples, tau=temperature,
            mask=mask, beta=beta, log_scores=log_scores
        )
        n_samples_actual = P_hat.shape[0]
    else:
        P_hat = deterministic_neural_sort(
            y_pred.unsqueeze(-1), tau=temperature, mask=mask
        ).unsqueeze(0)
        n_samples_actual = 1
    
    # Step 2: Transpose P̂ to get P̂^T (approximate unsorting matrix)
    P_hat_T = P_hat.transpose(-2, -1)  # Transpose last two dimensions
    
    # Step 3: Apply Sinkhorn scaling to P̂^T to make it doubly stochastic
    P_hat_T_reshaped = P_hat_T.view(n_samples_actual * batch_size, slate_length, slate_length)
    mask_repeated = mask.repeat(n_samples_actual, 1)
    
    P_hat_T_scaled = sinkhorn_scaling(P_hat_T_reshaped, mask_repeated, tol=tol, max_iter=max_iter)
    P_hat_T = P_hat_T_scaled.view(n_samples_actual, batch_size, slate_length, slate_length)
    
    # Step 4: Create discount vector with truncation at k
    positions = torch.arange(1, slate_length + 1, dtype=torch.float32, device=device)
    discounts = 1.0 / torch.log2(positions + 1.0)
    
    # Zero out discounts for positions > k
    discounts[k:] = 0.0
    
    # Step 5: Compute expected discounts per document: P̂^T @ d
    discounts_expanded = discounts.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, slate_length, 1]
    expected_discounts = torch.matmul(P_hat_T, discounts_expanded).squeeze(-1)  # [n_samples, batch_size, slate_length]
    
    # Step 6: Apply gain function to true labels
    y_true_clean = y_true.masked_fill(mask, 0.0)
    if powered_relevancies:
        gains = torch.pow(2.0, y_true_clean) - 1.0
    else:
        gains = y_true_clean
    
    # Step 7: Compute weighted gains: g(y) * expected_discounts
    weighted_gains = gains.unsqueeze(0) * expected_discounts  # [n_samples, batch_size, slate_length]
    
    # Step 8: Sum over documents to get DCG
    dcg_values = torch.sum(weighted_gains, dim=-1)  # [n_samples, batch_size]
    
    # Step 9: Compute IDCG using original labels
    if powered_relevancies:
        ideal_gains = torch.pow(2.0, y_true_clean) - 1.0
    else:
        ideal_gains = y_true_clean
    
    ideal_gains_sorted, _ = torch.sort(ideal_gains, descending=True)
    ideal_gains_k = ideal_gains_sorted[:, :k]
    discounts_k = discounts[:k]
    idcg = torch.sum(ideal_gains_k * discounts_k.unsqueeze(0), dim=-1)
    
    # Step 10: Normalize by IDCG
    ndcg = dcg_values / (idcg.unsqueeze(0) + DEFAULT_EPS)
    
    # Handle queries with IDCG = 0
    idcg_mask = (idcg == 0.0)
    ndcg = ndcg.masked_fill(idcg_mask.unsqueeze(0), 0.0)
    
    # Verify validity
    assert torch.all(ndcg >= 0.0), "Todos os valores de NDCG devem ser não-negativos"
    assert torch.all(ndcg <= 1.0 + 1e-6), "Todos os valores de NDCG devem ser <= 1"
    
    # Compute mean over samples and valid queries
    if idcg_mask.all():
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    valid_queries = ~idcg_mask
    mean_ndcg = ndcg[:, valid_queries].mean()
    
    return -mean_ndcg
