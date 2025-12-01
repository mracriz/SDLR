import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS
from allrank.models.losses.loss_utils import deterministic_neural_sort, sinkhorn_scaling, stochastic_neural_sort
from allrank.models.metrics import dcg
from allrank.models.model_utils import get_torch_device


def neuralNDCG_IPS(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, temperature=1., 
                   powered_relevancies=True, k=None, stochastic=False, n_samples=32, 
                   beta=0.1, log_scores=True, inverse_propensities_list=None):
    """
    Versão IPS (Inverse Propensity Scoring) do NeuralNDCG.
    
    Aplica pesos de propensão inversa para combater o viés posicional em dados de clique.
    A implementação segue a mesma lógica do neuralNDCG base, mas pondera as queries
    pelos inverse propensities para remover viés.
    
    Args:
        y_pred: scores preditos [batch_size, slate_length] ou shapes compatíveis
        y_true: labels verdadeiros [batch_size, slate_length] 
        padded_value_indicator: valor indicando padding (-1)
        temperature: parâmetro τ para controlar suavidade da aproximação
        powered_relevancies: se aplica função de ganho 2^x - 1
        k: truncamento no rank k
        stochastic: se usa variante estocástica
        n_samples: número de amostras para variante estocástica
        beta: parâmetro para NeuralSort estocástico
        log_scores: se aplica log nos scores
        inverse_propensities_list: tensor com pesos IPS [batch_size] ou compatível
        
    Returns:
        loss: -NeuralNDCG ponderado por IPS (negativo porque queremos maximizar)
    """
    device = get_torch_device()
    
    # Handle different input shapes (mesma lógica do neuralNDCG base)
    if y_pred.dim() == 3:
        if y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)
        else:
            # Para modelos de classificação, soma sobre classes
            y_pred = y_pred.sum(dim=-1)
    elif y_pred.dim() != 2:
        raise ValueError(f"y_pred deve ter 2 ou 3 dimensões, recebeu shape: {y_pred.shape}")
    
    batch_size, slate_length = y_pred.shape
    
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
        )
        n_samples_actual = P_hat.shape[0]
    else:
        P_hat = deterministic_neural_sort(
            y_pred.unsqueeze(-1), tau=temperature, mask=mask
        ).unsqueeze(0)
        n_samples_actual = 1
    
    # Step 2: Apply Sinkhorn scaling
    P_hat_reshaped = P_hat.view(n_samples_actual * batch_size, slate_length, slate_length)
    mask_repeated = mask.repeat(n_samples_actual, 1)
    
    P_hat_scaled = sinkhorn_scaling(P_hat_reshaped, mask_repeated, tol=1e-6, max_iter=50)
    P_hat = P_hat_scaled.view(n_samples_actual, batch_size, slate_length, slate_length)
    
    # Step 3: Apply gain function
    y_true_clean = y_true.masked_fill(mask, 0.0)
    if powered_relevancies:
        gains = torch.pow(2.0, y_true_clean) - 1.0
    else:
        gains = y_true_clean
    
    # Step 4: Compute quasi-sorted gains
    gains_expanded = gains.unsqueeze(0).unsqueeze(-1)
    quasi_sorted_gains = torch.matmul(P_hat, gains_expanded).squeeze(-1)
    
    # Step 5: Apply discount function
    positions = torch.arange(1, slate_length + 1, dtype=torch.float32, device=device)
    discounts = 1.0 / torch.log2(positions + 1.0)
    
    # Step 6: Truncate to top-k
    quasi_sorted_gains_k = quasi_sorted_gains[:, :, :k]
    discounts_k = discounts[:k]
    
    # Step 7: Compute DCG
    dcg_values = torch.sum(quasi_sorted_gains_k * discounts_k.unsqueeze(0).unsqueeze(0), dim=-1)
    
    # Step 8: Compute IDCG
    if powered_relevancies:
        ideal_gains = torch.pow(2.0, y_true_clean) - 1.0
    else:
        ideal_gains = y_true_clean
    
    ideal_gains_sorted, _ = torch.sort(ideal_gains, descending=True)
    ideal_gains_k = ideal_gains_sorted[:, :k]
    idcg = torch.sum(ideal_gains_k * discounts_k.unsqueeze(0), dim=-1)
    
    # Step 9: Normalize by IDCG
    ndcg = dcg_values / (idcg.unsqueeze(0) + DEFAULT_EPS)
    
    # Step 10: Handle queries with IDCG = 0
    idcg_mask = (idcg == 0.0)
    ndcg = ndcg.masked_fill(idcg_mask.unsqueeze(0), 0.0)
    
    # Step 11: Apply IPS weights (PRINCIPAL DIFERENÇA)
    if inverse_propensities_list is not None:
        # Garantir que os pesos IPS tenham a forma correta
        if inverse_propensities_list.dim() == 1:
            # Shape: [batch_size] -> [1, batch_size]
            ips_weights = inverse_propensities_list.unsqueeze(0)
        elif inverse_propensities_list.dim() == 2:
            # Shape: [n_samples, batch_size] ou [batch_size, slate_length]
            if inverse_propensities_list.shape[0] == batch_size:
                # [batch_size, slate_length] -> [1, batch_size] (média sobre slate)
                ips_weights = inverse_propensities_list.mean(dim=1).unsqueeze(0)
            else:
                # [n_samples, batch_size] -> usar diretamente
                ips_weights = inverse_propensities_list
        else:
            raise ValueError(f"inverse_propensities_list tem dimensões incompatíveis: {inverse_propensities_list.shape}")
        
        # Expandir weights para compatibilidade com ndcg: [n_samples, batch_size]
        if ips_weights.shape[0] != n_samples_actual:
            ips_weights = ips_weights.repeat(n_samples_actual, 1)
        
        # Aplicar pesos IPS: NDCG ponderado
        ndcg = ndcg * ips_weights
    
    # Clamp para estabilidade numérica
    ndcg = torch.clamp(ndcg, 0.0, 1.0)
    
    # Verify validity
    assert torch.all(ndcg >= 0.0), "Todos os valores de NDCG devem ser não-negativos"
    assert torch.all(ndcg <= 1.0 + 1e-4), f"Todos os valores de NDCG devem ser <= 1, max: {torch.max(ndcg):.6f}"
    
    # Step 12: Compute mean over samples and valid queries
    if idcg_mask.all():
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    valid_queries = ~idcg_mask
    
    if inverse_propensities_list is not None:
        # Para IPS, computar média ponderada apenas sobre queries válidas
        valid_ndcg = ndcg[:, valid_queries]
        if valid_ndcg.numel() > 0:
            mean_ndcg = valid_ndcg.sum() / valid_ndcg.numel()
        else:
            mean_ndcg = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        # Sem IPS, média normal
        mean_ndcg = ndcg[:, valid_queries].mean()
    
    return -mean_ndcg  # Negativo para maximizar NDCG