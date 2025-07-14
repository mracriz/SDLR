import torch
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

# --- Configuration & Pre-requisites ---
svm_file_path = '/Users/david/Documents/phd/JusbrasilData/colecao_especialistas/manual_svm_.txt'

script_dir = Path(__file__).resolve().parent
model_file_path_obj = script_dir / "Student" / "allRank-master" / "out1" / "model.pkl"
model_file_path_str = str(model_file_path_obj)

project_root_path = model_file_path_obj.parent.parent.resolve()

print(f"Attempting to add project root to sys.path: {project_root_path}")

if project_root_path.is_dir():
    allrank_package_path = project_root_path / "allrank"
    if allrank_package_path.is_dir():
        project_root_str = str(project_root_path)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
            print(f"Successfully added to sys.path: {project_root_str}")
        else:
            print(f"Path already in sys.path: {project_root_str}")
    else:
        print(f"Warning: Expected 'allrank' package folder not found at {allrank_package_path}")
        print("Ensure your project structure is correct.")
else:
    print(f"Warning: Calculated project root path does not exist or is not a directory: {project_root_path}")


def get_inference_device():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

model_path = model_file_path_str
device = get_inference_device()
loaded_model = None

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
else:
    try:
        print(f"Loading model from: {model_path}")
        loaded_model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        print("Model loaded successfully.")

        loaded_model.to(device)
        print(f"Model moved to device: {device}")

        loaded_model.eval()
        print("Model set to evaluation mode.")

    except ImportError as e:
        print(f"Error loading model: An import error occurred. This often means the model's class definition is not found.")
        print(f"Ensure the 'allRank' library and its dependencies are correctly installed and importable.")
        print(f"Specific error: {e}")
    except Exception as e:
        print(f"Error loading model: {e}")

N_FEATURES_MODEL_EXPECTS = 11

if 'loaded_model' not in locals() or 'device' not in locals():
    print("Error: `loaded_model` or `device` is not defined. Please ensure they are set up.")
    if 'device' not in locals():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
        print(f"Placeholder: Using device: {device}")
    if 'loaded_model' not in locals():
        print("Placeholder: `loaded_model` is not loaded. You need to load your model.")
        class DummyLTRModel(torch.nn.Module):
            def __init__(self, n_features): super().__init__(); self.linear = torch.nn.Linear(n_features, 1)
            def forward(self, x, mask, indices): return self.linear(x)
        loaded_model = DummyLTRModel(N_FEATURES_MODEL_EXPECTS).to(device).eval()
        print(f"Placeholder: Using a DUMMY model on device {device} in eval mode.")

# 1. Load SVM Data (Features, Labels, Query IDs)
print(f"Attempting to load SVM data from: {svm_file_path}")
if not os.path.exists(svm_file_path):
    print(f"Error: SVM file not found at {svm_file_path}")
    exit()

try:
    X_sparse, y_svm_labels, query_ids_svm = load_svmlight_file(
        svm_file_path,
        n_features=N_FEATURES_MODEL_EXPECTS,
        query_id=True,
        zero_based="auto"
    )
    if X_sparse.shape[0] == 0:
        print("Error: No samples loaded. Check SVM file path, content, or N_FEATURES_MODEL_EXPECTS.")
        exit()
    X_dense_all = X_sparse.toarray()
    print(f"SVM Data loaded: {X_dense_all.shape[0]} samples, {X_dense_all.shape[1]} features.")
    if X_dense_all.shape[1] != N_FEATURES_MODEL_EXPECTS:
        print(f"Warning: Loaded data has {X_dense_all.shape[1]} features, but N_FEATURES_MODEL_EXPECTS is {N_FEATURES_MODEL_EXPECTS}.")
except ValueError as e:
    print(f"Error loading SVM file (ValueError): {e}")
    print(f"This can happen if n_features ({N_FEATURES_MODEL_EXPECTS}) is smaller than max feature index in file, or file is malformed.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading SVM file: {e}")
    exit()

print(f"Checking X_dense_all for NaN or Inf values...")
nan_check = np.isnan(X_dense_all).any()
inf_check = np.isinf(X_dense_all).any()

if nan_check or inf_check:
    print(f"!!! Critical Warning: X_dense_all CONTAINS NaN or Inf values! !!!")
    print(f"  NaNs present: {nan_check} (Total NaNs: {np.sum(np.isnan(X_dense_all))})")
    print(f"  Infs present: {inf_check} (Total Infs: {np.sum(np.isinf(X_dense_all))})")
else:
    print("  X_dense_all does not contain any NaN or Inf values. Input data looks clean in that regard.")

print(f"  Min value in X_dense_all: {np.min(X_dense_all)}")
print(f"  Max value in X_dense_all: {np.max(X_dense_all)}")


# --- Configuration ---
SLATE_LENGTH = 30
DEBUG_MODE = True
DEBUG_HOOKS_ACTIVE = True
QUERIES_TO_DEBUG_WITH_HOOKS = 1

# --- Global variables for hook debugging ---
stop_checking_hooks_for_this_pass = False
nan_inf_detection_report = None

# --- PyTorch Hook for NaN/Inf Detection ---
def nan_check_hook(module, input_args, output_val):
    global stop_checking_hooks_for_this_pass, nan_inf_detection_report
    if stop_checking_hooks_for_this_pass:
        return
    current_layer_name = module.__class__.__name__
    details_found_this_call = None
    if input_args:
        for i, inp_tensor in enumerate(input_args):
            if isinstance(inp_tensor, torch.Tensor):
                is_nan = torch.isnan(inp_tensor).any().item()
                is_inf = torch.isinf(inp_tensor).any().item()
                if is_nan or is_inf:
                    details_found_this_call = {'layer': current_layer_name, 'source': f'INPUT_{i}', 'shape': inp_tensor.shape, 'nan': is_nan, 'inf': is_inf}
                    break
    if not details_found_this_call:
        if isinstance(output_val, torch.Tensor):
            is_nan = torch.isnan(output_val).any().item()
            is_inf = torch.isinf(output_val).any().item()
            if is_nan or is_inf:
                details_found_this_call = {'layer': current_layer_name, 'source': 'OUTPUT', 'shape': output_val.shape, 'nan': is_nan, 'inf': is_inf}
        elif isinstance(output_val, (list, tuple)):
            for i, out_tensor in enumerate(output_val):
                if isinstance(out_tensor, torch.Tensor):
                    is_nan = torch.isnan(out_tensor).any().item()
                    is_inf = torch.isinf(out_tensor).any().item()
                    if is_nan or is_inf:
                        details_found_this_call = {'layer': current_layer_name, 'source': f'OUTPUT_{i}', 'shape': out_tensor.shape, 'nan': is_nan, 'inf': is_inf}
                        break
    if details_found_this_call:
        nan_inf_detection_report = details_found_this_call
        stop_checking_hooks_for_this_pass = True
        if DEBUG_MODE:
            print(f"  HOOK >>> NaN/Inf in {details_found_this_call['source']} of/to layer: {details_found_this_call['layer']} | Shape: {details_found_this_call['shape']} | NaN: {details_found_this_call['nan']}, Inf: {details_found_this_call['inf']}")

if DEBUG_MODE:
    print("DEBUG_MODE: Anomaly detection can be enabled if needed (currently commented out).")
    print("\nDEBUG_MODE: Checking model parameters for NaNs/Infs...")
    found_nan_in_params = False
    for name, param in loaded_model.named_parameters():
        if torch.isnan(param.data).any() or torch.isinf(param.data).any():
            print(f"  !!! NaN/Inf found in model parameter: {name} !!! Shape: {param.data.shape}")
            found_nan_in_params = True
    if not found_nan_in_params:
        print("  No NaNs/Infs found in model parameters.")

hooks = []

# 2. Perform Inference
all_predictions_list = []
unique_q_ids_svm = np.unique(query_ids_svm)
print(f"\nStarting inference for {len(unique_q_ids_svm)} unique queries (Target Slate Length: {SLATE_LENGTH})...")

loaded_model.eval()
with torch.no_grad():
    for i, q_id in enumerate(unique_q_ids_svm):
        stop_checking_hooks_for_this_pass = False
        nan_inf_detection_report = None

        if DEBUG_HOOKS_ACTIVE and i < QUERIES_TO_DEBUG_WITH_HOOKS and not hooks:
            print(f"  DEBUG_MODE: Registering hooks for query {q_id} (iteration {i})...")
            for name, module_item in loaded_model.named_modules():
                if name:
                    hooks.append(module_item.register_forward_hook(nan_check_hook))
            print(f"  Registered {len(hooks)} hooks.")

        query_items_indices_in_original = np.where(query_ids_svm == q_id)[0]
        X_query_dense_np_orig = X_dense_all[query_items_indices_in_original]
        num_actual_docs = X_query_dense_np_orig.shape[0]

        if num_actual_docs == 0:
            if DEBUG_MODE: print(f"  DEBUG_MODE: No documents for q_id {q_id}. Skipping.")
            continue

        X_query_padded_np = np.zeros((SLATE_LENGTH, N_FEATURES_MODEL_EXPECTS), dtype=np.float32)
        current_mask_for_slate_np = np.ones(SLATE_LENGTH, dtype=bool)

        if num_actual_docs >= SLATE_LENGTH:
            X_query_padded_np = X_query_dense_np_orig[:SLATE_LENGTH, :]
            current_mask_for_slate_np[:SLATE_LENGTH] = False
            num_docs_in_slate = SLATE_LENGTH
        else:
            X_query_padded_np[:num_actual_docs, :] = X_query_dense_np_orig
            current_mask_for_slate_np[:num_actual_docs] = False
            num_docs_in_slate = num_actual_docs

        X_query_tensor_orig = torch.tensor(X_query_padded_np, dtype=torch.float32).to(device)
        current_mask_orig = torch.tensor(current_mask_for_slate_np, dtype=torch.bool).to(device)
        current_indices_orig = torch.arange(SLATE_LENGTH).to(device)

        X_query_tensor_batched = X_query_tensor_orig.unsqueeze(0)
        current_mask_batched = current_mask_orig.unsqueeze(0)
        current_indices_input = current_indices_orig.unsqueeze(0)

        if DEBUG_MODE and i < QUERIES_TO_DEBUG_WITH_HOOKS:
            print(f"\n  DEBUG_MODE: Inputs for q_id {q_id} (iteration {i}, actual_docs: {num_actual_docs}, docs_in_slate: {num_docs_in_slate}):")
            print(f"    X_batched shape: {X_query_tensor_batched.shape}, device: {X_query_tensor_batched.device}")
            print(f"    mask_batched shape: {current_mask_batched.shape}, device: {current_mask_batched.device}")
            print(f"    indices_input shape: {current_indices_input.shape}, device: {current_indices_input.device}")

        try:
            predictions_q_tensor_batched = loaded_model(X_query_tensor_batched, mask=current_mask_batched, indices=current_indices_input)
            if nan_inf_detection_report:
                print(f"  DEBUG >>> NaN/Inf reported by hooks for q_id {q_id} (see HOOK log above).")

            predictions_for_slate_squeezed = predictions_q_tensor_batched.squeeze(0)
            predictions_for_actual_docs = predictions_for_slate_squeezed[:num_actual_docs]
            all_predictions_list.append(predictions_for_actual_docs.cpu().numpy())

        except RuntimeError as e_rt:
            print(f"RuntimeError during model inference for q_id {q_id}: {e_rt}")
            print(f"  Input tensor shapes to model (batched): X={X_query_tensor_batched.shape}, mask={current_mask_batched.shape}, indices={current_indices_input.shape}")
            exit()
        except Exception as e_other:
            print(f"An unexpected error for q_id {q_id}: {e_other}")
            exit()

        if DEBUG_HOOKS_ACTIVE and i == QUERIES_TO_DEBUG_WITH_HOOKS - 1 and hooks:
            print(f"  DEBUG_MODE: Removing {len(hooks)} hooks after processing query {q_id} (iteration {i}).")
            for h in hooks: h.remove()
            hooks = []

if hooks:
    print(f"  DEBUG_MODE: Cleaning up {len(hooks)} remaining hooks post-loop...")
    for h in hooks: h.remove()
    hooks = []

if not all_predictions_list:
    print("No predictions were generated. Exiting.")
    exit()

# 3. Process and Consolidate Predictions
print("\nProcessing and consolidating predictions...")
try:
    processed_predictions_for_concat = []
    for i, pred_array in enumerate(all_predictions_list):
        if not isinstance(pred_array, np.ndarray):
            print(f"  Warning: Element {i} in all_predictions_list is not a NumPy array (type: {type(pred_array)}). Skipping.")
            continue
        if pred_array.ndim > 1 and pred_array.shape[1] == 1:
            processed_predictions_for_concat.append(pred_array.squeeze(axis=1))
        elif pred_array.ndim == 1:
            processed_predictions_for_concat.append(pred_array)
        elif pred_array.ndim > 1 and pred_array.shape[1] > 1:
            print(f"  Warning: Prediction array at index {i} has shape {pred_array.shape}. Defaulting to scores from the first column.")
            processed_predictions_for_concat.append(pred_array[:, 0])
        elif pred_array.ndim == 0:
             print(f"  Warning: Prediction array at index {i} is a scalar ({pred_array}). Wrapping in array.")
             processed_predictions_for_concat.append(np.array([pred_array]))
        else:
            print(f"  Warning: Prediction array at index {i} has an unexpected shape {pred_array.shape}. Attempting to use as is or flatten.")
            processed_predictions_for_concat.append(pred_array.flatten())

    if not processed_predictions_for_concat:
        print("Error: No valid prediction arrays to concatenate after processing.")
        exit()

    predicted_scores_np = np.concatenate(processed_predictions_for_concat)
except ValueError as e:
    print(f"Error concatenating predictions: {e}")
    exit()

if len(predicted_scores_np) != X_dense_all.shape[0]:
    print(f"!!! Critical Error: Mismatch in the total number of collated prediction scores ({len(predicted_scores_np)}) and the total number of samples in the input data ({X_dense_all.shape[0]}).")
    exit()

print(f"Successfully consolidated {len(predicted_scores_np)} scores.")
if np.isnan(predicted_scores_np).any():
    nan_count = np.sum(np.isnan(predicted_scores_np))
    print(f"!!! WARNING: Final `predicted_scores_np` contains {nan_count} NaN(s) out of {len(predicted_scores_np)} scores!!!")
else:
    print("  Final `predicted_scores_np` is clean of NaNs.")

# 4. Create Pandas DataFrame for Evaluation
print("\nCreating Pandas DataFrame for evaluation...")
try:
    df_results = pd.DataFrame({
        'query_id': query_ids_svm,
        'manual_label': y_svm_labels,
        'predicted_score': predicted_scores_np
    })
    print("DataFrame created successfully (first 5 rows):")
    print(df_results.head())
except ValueError as e:
    print(f"Error creating DataFrame: {e}")
    print(f"  len(query_ids_svm): {len(query_ids_svm) if 'query_ids_svm' in locals() else 'Not defined'}")
    print(f"  len(y_svm_labels): {len(y_svm_labels) if 'y_svm_labels' in locals() else 'Not defined'}")
    print(f"  len(predicted_scores_np): {len(predicted_scores_np)}")
    exit()

# 5. Add Predicted Ranking
print("\nAdding predicted ranking to DataFrame...")
df_results['predicted_ranking'] = df_results.groupby('query_id')['predicted_score'].rank(
    ascending=False,
    method='first'
)

print("Predicted ranking added (showing head for first few query_ids as sample):")
unique_qids_in_results = df_results['query_id'].unique()
for q_idx, qid_val in enumerate(unique_qids_in_results):
    if q_idx < 3:
        print(f"\nQuery ID: {qid_val}")
        print(df_results[df_results['query_id'] == qid_val].head().sort_values(by='predicted_ranking'))
    else:
        break
if len(unique_qids_in_results) > 3:
    print("... and so on for other query_ids.")

#=============================================================================
# SEÇÃO DE CÁLCULO DO NDCG - INÍCIO
#=============================================================================

def ndcg_at_k(true_relevance, predicted_rank, k=5):
    """
    Calcula o NDCG@k para uma única consulta.
    """
    # Garante que os argumentos sejam Series do pandas para usar .iloc e .argsort()
    if not isinstance(true_relevance, pd.Series):
        true_relevance = pd.Series(true_relevance)
    if not isinstance(predicted_rank, pd.Series):
        predicted_rank = pd.Series(predicted_rank)
        
    # Ordena os rótulos de relevância verdadeiros de acordo com o ranking previsto
    # argsort() retorna os índices que ordenariam a série, e .iloc os usa para reordenar a outra série
    sorted_true_relevance = true_relevance.iloc[predicted_rank.argsort().values]
    
    # Pega os top-k para o cálculo do DCG
    relevance_at_k = sorted_true_relevance.head(k)
    
    # Calcula DCG@k
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum((np.power(2, relevance_at_k.values) - 1) / discounts)
    
    # Calcula IDCG@k
    ideal_relevance_at_k = true_relevance.sort_values(ascending=False).head(k)
    ideal_discounts = np.log2(np.arange(2, len(ideal_relevance_at_k) + 2))
    idcg = np.sum((np.power(2, ideal_relevance_at_k.values) - 1) / ideal_discounts)
    
    # Calcula NDCG@k
    if idcg == 0:
        return 0.0 # ou np.nan, dependendo de como você quer tratar consultas sem documentos relevantes
    else:
        return dcg / idcg

# Define os valores de k que você quer avaliar
ks = [1, 3, 5, 10]
df_test = df_results # Usa o DataFrame que você já criou

print("\n--- Calculando NDCG médio ---")

for k in ks:
    ndcg_scores = []
    for query_id, group in df_test.groupby('query_id'):
        # Pula consultas com menos de k documentos, pois NDCG@k não é bem definido
        if len(group) < k:
            continue
            
        true_relevance = group['manual_label']
        predicted_rank = group['predicted_ranking']
        
        ndcg_score = ndcg_at_k(true_relevance, predicted_rank, k=k)
        ndcg_scores.append(ndcg_score)

    # Calcula a média dos scores de NDCG@k para todas as consultas válidas
    average_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    print(f"Average NDCG@{k}: {average_ndcg:.4f}")

#=============================================================================
# SEÇÃO DE CÁLCULO DO NDCG - FIM
#=============================================================================