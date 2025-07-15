import torch
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
import mlflow

# --- THIS IS THE FINAL FIX ---
# This block ensures that Python can find the 'allrank' library when
# torch.load() tries to unpickle the model file. It needs to know the
# model's class definition.
try:
    # We navigate from this script's location up to the SDLR root,
    # then down into the Student's project where the 'allrank' code lives.
    # This makes the script runnable by the orchestrator.
    student_project_root = Path(__file__).resolve().parent / "Student" / "allRank-master"
    if str(student_project_root) not in sys.path:
        sys.path.insert(0, str(student_project_root))
    
    # This import is just to confirm the path is correct.
    from allrank.models.model import LTRModel
    print("Successfully added project to path and found LTRModel.")
except ImportError:
    print("Warning: Could not add project root to sys.path. Ensure 'allrank' is installed or the path is set correctly.")
# --- END OF FIX ---


def parse_args():
    """
    Parses command-line arguments for the inference script.
    """
    parser = argparse.ArgumentParser(description="Inference and NDCG evaluation for allRank models.")
    parser.add_argument("--svm_file_path", required=True, help="Path to the evaluation data in SVMLight format.")
    parser.add_argument("--model_file_path", required=True, help="Path to the trained model.pkl file.")
    parser.add_argument("--mlflow_run_id", required=False, default=None, help="ID of the MLflow run to log metrics to.")
    parser.add_argument("--n_features", type=int, default=11, help="Number of features in the dataset.")
    parser.add_argument("--slate_length", type=int, default=30, help="Slate length used for padding during inference.")
    return parser.parse_args()

def get_inference_device():
    """
    Determines the best available device for inference.
    """
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
    """
    Calculates NDCG@k for a single query group.
    """
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
    Main function to orchestrate model loading, inference, and evaluation.
    """
    args = parse_args()
    device = get_inference_device()

    # --- 1. Load Model ---
    print(f"Loading model from: {args.model_file_path}")
    if not os.path.exists(args.model_file_path):
        print(f"FATAL: Model file not found at '{args.model_file_path}'")
        sys.exit(1)
    
    try:
        loaded_model = torch.load(args.model_file_path, map_location=device, weights_only=False)
        loaded_model.eval()
        print("Model loaded successfully and set to evaluation mode.")
    except Exception as e:
        print(f"FATAL: Failed to load the model.")
        print(f"Specific error: {e}")
        sys.exit(1)

    # --- 2. Load Data ---
    print(f"Loading evaluation data from: {args.svm_file_path}")
    try:
        X_sparse, y_labels, q_ids = load_svmlight_file(
            args.svm_file_path, n_features=args.n_features, query_id=True, zero_based="auto"
        )
        X_dense = X_sparse.toarray()
        print(f"Data loaded: {X_dense.shape[0]} documents, {X_dense.shape[1]} features.")
    except Exception as e:
        print(f"FATAL: Failed to load SVM file. Check format and n_features. Error: {e}")
        sys.exit(1)

    # --- 3. Run Inference ---
    all_predictions = []
    unique_query_ids = np.unique(q_ids)
    print(f"\nStarting inference on {len(unique_query_ids)} unique queries...")

    with torch.no_grad():
        for q_id in unique_query_ids:
            doc_indices = np.where(q_ids == q_id)[0]
            X_query = X_dense[doc_indices]
            num_docs = X_query.shape[0]

            X_padded = np.zeros((args.slate_length, args.n_features), dtype=np.float32)
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

    # --- 4. Evaluate and Log Results ---
    df_results = pd.DataFrame({
        'query_id': q_ids, 'manual_label': y_labels, 'predicted_score': predicted_scores
    })
    
    print("\n--- Calculating and Logging NDCG ---")
    if args.mlflow_run_id:
        with mlflow.start_run(run_id=args.mlflow_run_id):
            print(f"Successfully connected to MLflow Run ID: {args.mlflow_run_id}")
            ks = [1, 3, 5, 10]
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