import torch
import os
import sys
import json
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
    parser.add_argument("--slate_length", type=int, default=40, help="Tamanho da slate usado para padding durante a inferência.")
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

def mrr_at_k(df_group, k, score_col='predicted_score', label_col='manual_label'):
    """
    Calcula Mean Reciprocal Rank (MRR) at k para um grupo de documentos.
    MRR considera a posição do primeiro documento relevante.
    """
    # Ordena por score predito em ordem decrescente
    sorted_items = df_group.sort_values(by=score_col, ascending=False).head(k)
    
    # Encontra a posição do primeiro documento relevante (label > 0)
    for position, (_, row) in enumerate(sorted_items.iterrows(), start=1):
        if row[label_col] > 0:  # Documento relevante
            return 1.0 / position
    
    return 0.0  # Nenhum documento relevante encontrado no top-k

def create_detailed_ranking_report(df_results):
    """
    Cria um relatório detalhado dos rankings para cada query.
    """
    ranking_reports = []
    
    for query_id, group in df_results.groupby('query_id'):
        # Ordena por score predito (ranking do modelo)
        ranked_docs = group.sort_values(by='predicted_score', ascending=False).reset_index(drop=True)
        
        # Adiciona informações de posição
        ranked_docs['predicted_rank'] = range(1, len(ranked_docs) + 1)
        
        # Ordena por relevância real para comparação
        ideal_ranking = group.sort_values(by='manual_label', ascending=False).reset_index(drop=True)
        ideal_ranking['ideal_rank'] = range(1, len(ideal_ranking) + 1)
        
        # Calcula métricas para esta query
        ndcg_scores = {}
        mrr_scores = {}
        for k in [1, 3, 5, 10]:
            if len(group) >= k:
                ndcg_scores[f'ndcg_at_{k}'] = ndcg_at_k(group, k)
                mrr_scores[f'mrr_at_{k}'] = mrr_at_k(group, k)
        
        ranking_report = {
            'query_id': query_id,
            'num_docs': len(group),
            'num_relevant_docs': sum(group['manual_label'] > 0),
            'metrics': {**ndcg_scores, **mrr_scores},
            'ranking': ranked_docs.to_dict('records'),
            'ideal_ranking': ideal_ranking[['manual_label', 'ideal_rank']].to_dict('records')
        }
        
        ranking_reports.append(ranking_report)
    
    return ranking_reports

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
    df_results = pd.DataFrame({
        'query_id': q_ids, 'manual_label': y_labels, 'predicted_score': predicted_scores
    })
    
    print("\n--- Calculating NDCG and MRR Metrics ---")
    if args.mlflow_run_id:
        with mlflow.start_run(run_id=args.mlflow_run_id):
            print(f"Successfully connected to MLflow Run ID: {args.mlflow_run_id}")
            
            # Calcula métricas agregadas
            ks = [1, 3, 5, 10, 20]
            grouped = df_results.groupby('query_id')
            
            print("\n📊 NDCG Metrics:")
            for k in ks:
                avg_ndcg = grouped.apply(lambda g: ndcg_at_k(g, k)).mean()
                print(f"Average NDCG@{k}: {avg_ndcg:.4f}")
                mlflow.log_metric(f"ndcg_at_{k}", avg_ndcg)
            
            print("\n🎯 MRR Metrics:")
            for k in ks:
                avg_mrr = grouped.apply(lambda g: mrr_at_k(g, k)).mean()
                print(f"Average MRR@{k}: {avg_mrr:.4f}")
                mlflow.log_metric(f"mrr_at_{k}", avg_mrr)

            # Salva dados básicos de scores
            results_path = "inference_scores.csv"
            df_results.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path, "evaluation_output")
            
            # Cria e salva relatório detalhado de rankings
            print("\n📝 Creating detailed ranking reports...")
            detailed_reports = create_detailed_ranking_report(df_results)
            
            # Salva relatório detalhado como JSON
            detailed_report_path = "detailed_ranking_report.json"
            with open(detailed_report_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_reports, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(detailed_report_path, "evaluation_output")
            
            # Cria relatório resumido por query
            query_summary = []
            for report in detailed_reports:
                summary = {
                    'query_id': report['query_id'],
                    'num_docs': report['num_docs'],
                    'num_relevant_docs': report['num_relevant_docs'],
                    **report['metrics']
                }
                query_summary.append(summary)
            
            query_summary_df = pd.DataFrame(query_summary)
            query_summary_path = "query_level_metrics.csv"
            query_summary_df.to_csv(query_summary_path, index=False)
            mlflow.log_artifact(query_summary_path, "evaluation_output")
            
            # Cria relatório de rankings comparativos (top-10 predito vs ideal)
            comparative_rankings = []
            for query_id, group in df_results.groupby('query_id'):
                # Top-10 predito
                predicted_top10 = group.sort_values('predicted_score', ascending=False).head(10)
                predicted_ranking = list(zip(predicted_top10['manual_label'].values, 
                                           predicted_top10['predicted_score'].values))
                
                # Top-10 ideal
                ideal_top10 = group.sort_values('manual_label', ascending=False).head(10)
                ideal_ranking = list(zip(ideal_top10['manual_label'].values,
                                       ideal_top10['predicted_score'].values))
                
                comparative_rankings.append({
                    'query_id': query_id,
                    'predicted_top10': [{'relevance': r, 'score': s} for r, s in predicted_ranking],
                    'ideal_top10': [{'relevance': r, 'score': s} for r, s in ideal_ranking]
                })
            
            comparative_path = "comparative_rankings.json"
            with open(comparative_path, 'w', encoding='utf-8') as f:
                json.dump(comparative_rankings, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(comparative_path, "evaluation_output")
            
            # Remove arquivos temporários
            os.remove(results_path)
            os.remove(detailed_report_path)
            os.remove(query_summary_path)
            os.remove(comparative_path)
            
            print("✅ All metrics, rankings, and detailed reports logged to MLflow!")
            print("📁 Artifacts saved:")
            print("   - inference_scores.csv: Basic scores per document")
            print("   - detailed_ranking_report.json: Complete ranking analysis")
            print("   - query_level_metrics.csv: Per-query NDCG and MRR metrics")
            print("   - comparative_rankings.json: Predicted vs Ideal top-10 rankings")
    else:
        print("No MLflow Run ID provided. Printing metrics locally.")
        ks = [1, 3, 5, 10]
        grouped = df_results.groupby('query_id')
        
        print("\n📊 NDCG Metrics:")
        for k in ks:
            avg_ndcg = grouped.apply(lambda g: ndcg_at_k(g, k)).mean()
            print(f"Average NDCG@{k}: {avg_ndcg:.4f}")
            
        print("\n🎯 MRR Metrics:")
        for k in ks:
            avg_mrr = grouped.apply(lambda g: mrr_at_k(g, k)).mean()
            print(f"Average MRR@{k}: {avg_mrr:.4f}")

if __name__ == "__main__":
    main()