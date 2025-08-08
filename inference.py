import torch
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
import mlflow

# --- Bloco de Correção de Path ---
try:
    student_project_root = Path(__file__).resolve().parent / "Student" / "allRank-master"
    if str(student_project_root) not in sys.path:
        sys.path.insert(0, str(student_project_root))
    from allrank.models.model import LTRModel
    print("Successfully added project to path and found LTRModel.")
except ImportError:
    print("Warning: Could not add project root to sys.path. Ensure 'allrank' is installed or the path is set correctly.")
# --- Fim do Bloco de Correção ---

def get_max_features_from_svm(file_path):
    """
    Lê um arquivo SVM Rank para descobrir o maior índice de feature.
    Isso evita a necessidade de especificar --n_features manualmente.
    """
    print("Detectando o número de features no arquivo SVM...")
    max_feat_idx = 0
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            for part in parts[2:]:  # Pula o label e o qid
                if ":" in part:
                    try:
                        feat_idx = int(part.split(':')[0])
                        if feat_idx > max_feat_idx:
                            max_feat_idx = feat_idx
                    except (ValueError, IndexError):
                        # Ignora partes malformadas
                        continue
    print(f"Detecção concluída. O maior índice de feature encontrado foi: {max_feat_idx}")
    return max_feat_idx

def parse_args():
    """
    Analisa os argumentos da linha de comando para o script de inferência.
    """
    parser = argparse.ArgumentParser(description="Inferência e avaliação NDCG para modelos allRank.")
    parser.add_argument("--svm_file_path", required=True, help="Caminho para os dados de avaliação no formato SVMLight.")
    parser.add_argument("--model_file_path", required=True, help="Caminho para o arquivo do modelo treinado (model.pkl).")
    parser.add_argument("--mlflow_run_id", required=False, default=None, help="ID da execução do MLflow para registrar as métricas.")
    # --- MUDANÇA AQUI ---
    # n_features agora é opcional. Se não for fornecido, será detectado automaticamente.
    parser.add_argument("--n_features", type=int, default=None, help="(Opcional) Número de features no dataset. Se não fornecido, será detectado automaticamente.")
    # --- FIM DA MUDANÇA ---
    parser.add_argument("--slate_length", type=int, default=30, help="Tamanho da slate usado para padding durante a inferência.")
    return parser.parse_args()

def get_inference_device():
    # ... (código inalterado) ...
    if torch.cuda.is_available():
        print("Using CUDA for inference.")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS for inference.")
        return torch.device("mps")
    else:
        print("Using CPU for inference.")
        return torch.device("cpu")

def ndcg_at_k(df_group, k, score_col='predicted_score', label_col='manual_label'):
    # ... (código inalterado) ...
    top_k_items = df_group.sort_values(by=score_col, ascending=False).head(k)
    relevance = top_k_items[label_col].values
    discounts = np.log2(np.arange(len(relevance)) + 2)
    dcg = np.sum((np.power(2, relevance) - 1) / discounts)
    ideal_top_k_items = df_group.sort_values(by=label_col, ascending=False).head(k)
    ideal_relevance = ideal_top_k_items[label_col].values
    ideal_discounts = np.log2(np.arange(len(ideal_relevance)) + 2)
    idcg = np.sum((np.power(2, ideal_relevance) - 1) / ideal_discounts)
    return 0.0 if idcg == 0 else dcg / idcg

def main():
    """
    Função principal para orquestrar o carregamento do modelo, inferência e avaliação.
    """
    args = parse_args()
    device = get_inference_device()

    # --- 1. Carregar Modelo ---
    # ... (código inalterado) ...
    print(f"Loading model from: {args.model_file_path}")
    if not os.path.exists(args.model_file_path):
        print(f"FATAL: Model file not found at '{args.model_file_path}'")
        sys.exit(1)
    try:
        loaded_model = torch.load(args.model_file_path, map_location=device, weights_only=False)
        loaded_model.eval()
        print("Model loaded successfully and set to evaluation mode.")
    except Exception as e:
        print(f"FATAL: Failed to load the model. Ensure 'allrank' is installed ('pip install -e .').")
        print(f"Specific error: {e}")
        sys.exit(1)

    # --- 2. Carregar Dados ---
    print(f"Loading evaluation data from: {args.svm_file_path}")
    
    # --- MUDANÇA AQUI ---
    # Detecta n_features se não for fornecido
    n_features = args.n_features
    if n_features is None:
        n_features = get_max_features_from_svm(args.svm_file_path)
    # --- FIM DA MUDANÇA ---
    
    try:
        X_sparse, y_labels, q_ids = load_svmlight_file(
            args.svm_file_path, n_features=n_features, query_id=True, zero_based="auto"
        )
        X_dense = X_sparse.toarray()
        print(f"Data loaded: {X_dense.shape[0]} documents, {X_dense.shape[1]} features.")
    except Exception as e:
        print(f"FATAL: Failed to load SVM file. Check format. Error: {e}")
        sys.exit(1)

    # --- 3. Executar Inferência ---
    # ... (o resto do código permanece o mesmo, mas agora usa o `n_features` correto) ...
    all_predictions = []
    unique_query_ids = np.unique(q_ids)
    print(f"\nStarting inference on {len(unique_query_ids)} unique queries...")
    with torch.no_grad():
        for q_id in unique_query_ids:
            doc_indices = np.where(q_ids == q_id)[0]
            X_query = X_dense[doc_indices]
            num_docs = X_query.shape[0]

            X_padded = np.zeros((args.slate_length, n_features), dtype=np.float32)
            X_padded[:num_docs] = X_query
            
            mask = np.ones(args.slate_length, dtype=bool)
            mask[:num_docs] = False

            X_tensor = torch.tensor(X_padded, dtype=torch.float32).unsqueeze(0).to(device)
            mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
            
            try:
                scores = loaded_model.score(X_tensor, mask_tensor, indices=None)
            except AttributeError:
                scores = loaded_model(X_tensor, mask_tensor, indices=None)
            
            scores_np = scores.squeeze(0).cpu().numpy()
            all_predictions.append(scores_np[:num_docs])

    predicted_scores = np.concatenate(all_predictions).flatten()

    # --- 4. Avaliar e Registrar Resultados ---
    # ... (código inalterado) ...
    df_results = pd.DataFrame({
        'query_id': q_ids, 'manual_label': y_labels, 'predicted_score': predicted_scores
    })
    print("\n--- Calculating and Logging NDCG ---")
    if args.mlflow_run_id:
        with mlflow.start_run(run_id=args.mlflow_run_id):
            print(f"Successfully connected to MLflow Run ID: {args.mlflow_run_id}")
            ks = [1, 3, 5, 10, 20]
            grouped = df_results.groupby('query_id')
            for k in ks:
                avg_ndcg = grouped.apply(lambda g: ndcg_at_k(g, k)).mean()
                print(f"Average NDCG@{k}: {avg_ndcg:.4f}")
                mlflow.log_metric(f"ndcg_at_{k}", avg_ndcg)

            results_path = "inference_scores.csv"
            df_results.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path, "evaluation_output")
            os.remove(results_path)
            print("NDCG metrics and scores CSV logged to MLflow.")
    else:
        print("No MLflow Run ID provided. Printing metrics locally.")
        ks = [1, 3, 5, 10]
        grouped = df_results.groupby('query_id')
        for k in ks:
            avg_ndcg = grouped.apply(lambda g: ndcg_at_k(g, k)).mean()
            print(f"Average NDCG@{k}: {avg_ndcg:.4f}")

if __name__ == "__main__":
    main()