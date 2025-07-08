import torch
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

# --- Configuration & Pre-requisites ---
svm_file_path = '/Users/david/Documents/phd/JusbrasilData/colecao_especialistas/manual_svm_1.txt'

script_dir = Path(__file__).resolve().parent
model_file_path_obj = script_dir / "Student" / "allRank-master" / "out1" / "model.pkl"
model_file_path_str = str(model_file_path_obj)

# project_root should be the directory containing the 'allrank' package folder.
# If model.pkl is in .../allRank-master/out1/, then .../allRank-master/ is two levels up.
project_root_path = model_file_path_obj.parent.parent.resolve() # .resolve() makes it absolute and normalized

print(f"Attempting to add project root to sys.path: {project_root_path}")

if project_root_path.is_dir():
    # Check if the 'allrank' package folder exists directly within this project_root_path
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

# For debugging, print the current sys.path
# print("Current sys.path:")
# for p in sys.path:
# print(f"  {p}")
# print("-" * 20)

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

model_path = model_file_path_str # Use the already defined string
device = get_inference_device()
loaded_model = None

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
else:
    try:
        print(f"Loading model from: {model_path}")
        # Ensure allrank classes are importable before this line is called
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
        # print("Re-check sys.path and project structure if 'allrank' module is not found.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # For more detailed debugging, you can uncomment the next two lines:
        # import traceback
        # traceback.print_exc()



N_FEATURES_MODEL_EXPECTS = 11 # Example from your SVM line, adjust if needed for your model

if 'loaded_model' not in locals() or 'device' not in locals():
    print("Error: `loaded_model` or `device` is not defined. Please ensure they are set up.")
    # Example setup if missing:
    if 'device' not in locals():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
        print(f"Placeholder: Using device: {device}")
    if 'loaded_model' not in locals():
        print("Placeholder: `loaded_model` is not loaded. You need to load your model.")
        # As a dummy for the script to proceed without error (replace with your actual model)
        class DummyLTRModel(torch.nn.Module):
            def __init__(self, n_features): super().__init__(); self.linear = torch.nn.Linear(n_features, 1)
            def forward(self, x, mask, indices): return self.linear(x) # Accepts mask & indices
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
        zero_based="auto" # Handles 0-based or 1-based
    )
    if X_sparse.shape[0] == 0:
        print("Error: No samples loaded. Check SVM file path, content, or N_FEATURES_MODEL_EXPECTS.")
        exit()
    X_dense_all = X_sparse.toarray()
    print(f"SVM Data loaded: {X_dense_all.shape[0]} samples, {X_dense_all.shape[1]} features.")
    if X_dense_all.shape[1] != N_FEATURES_MODEL_EXPECTS:
        print(f"Warning: Loaded data has {X_dense_all.shape[1]} features, "
              f"but N_FEATURES_MODEL_EXPECTS is {N_FEATURES_MODEL_EXPECTS}.")
except ValueError as e:
    print(f"Error loading SVM file (ValueError): {e}")
    print(f"This can happen if n_features ({N_FEATURES_MODEL_EXPECTS}) is smaller than max feature index in file, "
          "or file is malformed.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading SVM file: {e}")
    exit()

# After X_dense_all = X_sparse.toarray()
print(f"Checking X_dense_all for NaN or Inf values...")
nan_check = np.isnan(X_dense_all).any()
inf_check = np.isinf(X_dense_all).any()

if nan_check or inf_check:
    print(f"!!! Critical Warning: X_dense_all CONTAINS NaN or Inf values! !!!")
    print(f"  NaNs present: {nan_check} (Total NaNs: {np.sum(np.isnan(X_dense_all))})")
    print(f"  Infs present: {inf_check} (Total Infs: {np.sum(np.isinf(X_dense_all))})")
    # Optional: Find where they are
    # nan_indices = np.where(np.isnan(X_dense_all))
    # print(f"  Indices of NaNs (first few): {list(zip(nan_indices[0][:5], nan_indices[1][:5]))}")
    # inf_indices = np.where(np.isinf(X_dense_all))
    # print(f"  Indices of Infs (first few): {list(zip(inf_indices[0][:5], inf_indices[1][:5]))}")
else:
    print("  X_dense_all does not contain any NaN or Inf values. Input data looks clean in that regard.")

# Also, check min/max values to understand the range
print(f"  Min value in X_dense_all: {np.min(X_dense_all)}")
print(f"  Max value in X_dense_all: {np.max(X_dense_all)}")

import torch
import numpy as np
# Assuming pandas, load_svmlight_file, os are imported, and
# X_dense_all, query_ids_svm, loaded_model, device, N_FEATURES_MODEL_EXPECTS
# are already defined and valid in your environment.

# --- Configuration from your JSON & Debugging ---
SLATE_LENGTH = 30  # From your model's training config "data/slate_length": 30
# N_FEATURES_MODEL_EXPECTS should be set correctly (e.g., 11)

DEBUG_MODE = True
DEBUG_HOOKS_ACTIVE = True
QUERIES_TO_DEBUG_WITH_HOOKS = 1 # Process only the first query with hooks for detailed inspection

# --- Global variables for hook debugging ---
stop_checking_hooks_for_this_pass = False
nan_inf_detection_report = None # Stores the first {layer, source, shape, nan, inf} dict

# --- PyTorch Hook for NaN/Inf Detection (checks inputs first) ---
def nan_check_hook(module, input_args, output_val): # Renamed 'output' to 'output_val'
    global stop_checking_hooks_for_this_pass, nan_inf_detection_report

    if stop_checking_hooks_for_this_pass: # If NaN already found in this pass, skip
        return

    current_layer_name = module.__class__.__name__
    details_found_this_call = None

    # 1. Check INPUTS to this module
    if input_args:
        for i, inp_tensor in enumerate(input_args):
            if isinstance(inp_tensor, torch.Tensor):
                is_nan = torch.isnan(inp_tensor).any().item()
                is_inf = torch.isinf(inp_tensor).any().item()
                if is_nan or is_inf:
                    details_found_this_call = {
                        'layer': current_layer_name, 'source': f'INPUT_{i}',
                        'shape': inp_tensor.shape, 'nan': is_nan, 'inf': is_inf
                    }
                    break

    # 2. If inputs are clean, check OUTPUT of this module
    if not details_found_this_call:
        processed_output = False
        if isinstance(output_val, torch.Tensor):
            is_nan = torch.isnan(output_val).any().item()
            is_inf = torch.isinf(output_val).any().item()
            if is_nan or is_inf:
                details_found_this_call = {
                    'layer': current_layer_name, 'source': 'OUTPUT',
                    'shape': output_val.shape, 'nan': is_nan, 'inf': is_inf
                }
            processed_output = True
        elif isinstance(output_val, (list, tuple)):
            for i, out_tensor in enumerate(output_val):
                if isinstance(out_tensor, torch.Tensor):
                    is_nan = torch.isnan(out_tensor).any().item()
                    is_inf = torch.isinf(out_tensor).any().item()
                    if is_nan or is_inf:
                        details_found_this_call = {
                            'layer': current_layer_name, 'source': f'OUTPUT_{i}',
                            'shape': out_tensor.shape, 'nan': is_nan, 'inf': is_inf
                        }
                        break
            processed_output = True

        # If output_val was not a tensor or list/tuple of tensors,
        # details_found_this_call would remain None from output check.
        # This is fine, as we only care about tensor NaNs.

    if details_found_this_call:
        nan_inf_detection_report = details_found_this_call
        stop_checking_hooks_for_this_pass = True
        if DEBUG_MODE:
            print(f"  HOOK >>> NaN/Inf in {details_found_this_call['source']} of/to layer: {details_found_this_call['layer']} "
                  f"| Shape: {details_found_this_call['shape']} | NaN: {details_found_this_call['nan']}, Inf: {details_found_this_call['inf']}")

# --- Optional: Enable Anomaly Detection (can slow things down) ---
if DEBUG_MODE:
    # torch.autograd.set_detect_anomaly(True)
    print("DEBUG_MODE: Anomaly detection can be enabled if needed (currently commented out).")

# --- Check Model Parameters for NaNs/Infs (once before the loop) ---
if DEBUG_MODE:
    print("\nDEBUG_MODE: Checking model parameters for NaNs/Infs...")
    found_nan_in_params = False
    for name, param in loaded_model.named_parameters(): # Ensure loaded_model is defined
        if torch.isnan(param.data).any() or torch.isinf(param.data).any():
            print(f"  !!! NaN/Inf found in model parameter: {name} !!! Shape: {param.data.shape}")
            found_nan_in_params = True
    if not found_nan_in_params:
        print("  No NaNs/Infs found in model parameters.")

hooks = [] # To store hook handles for removal

# 2. Perform Inference using the pre-loaded `loaded_model`
all_predictions_list = []
unique_q_ids_svm = np.unique(query_ids_svm) # Ensure query_ids_svm is defined
print(f"\nStarting inference for {len(unique_q_ids_svm)} unique queries (Target Slate Length: {SLATE_LENGTH})...")

loaded_model.eval() # Ensure model is in eval mode
with torch.no_grad(): # Context manager for inference
    for i, q_id in enumerate(unique_q_ids_svm):
        # Reset global flags for each new forward pass (each query in this loop)
        stop_checking_hooks_for_this_pass = False
        nan_inf_detection_report = None

        # Register hooks for the specified number of initial queries
        if DEBUG_HOOKS_ACTIVE and i < QUERIES_TO_DEBUG_WITH_HOOKS and not hooks: # Register only once for the debug batch
            print(f"  DEBUG_MODE: Registering hooks for query {q_id} (iteration {i})...")
            for name, module_item in loaded_model.named_modules(): # Corrected variable name to module_item
                if name: # Avoid hooking the top-level model itself unnecessarily
                    hooks.append(module_item.register_forward_hook(nan_check_hook))
            print(f"  Registered {len(hooks)} hooks.")

        query_items_indices_in_original = np.where(query_ids_svm == q_id)[0]
        X_query_dense_np_orig = X_dense_all[query_items_indices_in_original] # Ensure X_dense_all defined
        num_actual_docs = X_query_dense_np_orig.shape[0]

        if num_actual_docs == 0:
            if DEBUG_MODE: print(f"  DEBUG_MODE: No documents for q_id {q_id}. Skipping.")
            continue

        # --- Pad or Truncate features to SLATE_LENGTH ---
        X_query_padded_np = np.zeros((SLATE_LENGTH, N_FEATURES_MODEL_EXPECTS), dtype=np.float32)

        # --- Create the mask for valid items (before batching) ---
        # Convention: False = valid item, True = padding (to be masked out by attention)
        # This convention needs to match how your allRank model's attention layers use the mask.
        current_mask_for_slate_np = np.ones(SLATE_LENGTH, dtype=bool) # Initialize all to padding (True)

        if num_actual_docs >= SLATE_LENGTH:
            X_query_padded_np = X_query_dense_np_orig[:SLATE_LENGTH, :]
            current_mask_for_slate_np[:SLATE_LENGTH] = False # All SLATE_LENGTH items are valid
            num_docs_in_slate = SLATE_LENGTH
        else: # num_actual_docs < SLATE_LENGTH
            X_query_padded_np[:num_actual_docs, :] = X_query_dense_np_orig
            current_mask_for_slate_np[:num_actual_docs] = False # Mark actual docs as valid
            num_docs_in_slate = num_actual_docs # Used for slicing predictions later

        # Convert to tensors and move to device
        X_query_tensor_orig = torch.tensor(X_query_padded_np, dtype=torch.float32).to(device) # Ensure device is defined
        current_mask_orig = torch.tensor(current_mask_for_slate_np, dtype=torch.bool).to(device)
        current_indices_orig = torch.arange(SLATE_LENGTH).to(device) # Indices for the full slate

        # Add batch dimension
        X_query_tensor_batched = X_query_tensor_orig.unsqueeze(0)   # Shape: [1, SLATE_LENGTH, N_FEATURES]
        current_mask_batched = current_mask_orig.unsqueeze(0)    # Shape: [1, SLATE_LENGTH]
        current_indices_input = current_indices_orig.unsqueeze(0) # Shape: [1, SLATE_LENGTH]

        if DEBUG_MODE and i < QUERIES_TO_DEBUG_WITH_HOOKS:
            print(f"\n  DEBUG_MODE: Inputs for q_id {q_id} (iteration {i}, actual_docs: {num_actual_docs}, docs_in_slate: {num_docs_in_slate}):")
            print(f"    X_batched shape: {X_query_tensor_batched.shape}, device: {X_query_tensor_batched.device}")
            print(f"    mask_batched shape: {current_mask_batched.shape}, device: {current_mask_batched.device}")
            # To see mask values: print(f"    mask_batched values (first few): {current_mask_batched[0, :num_actual_docs + 2]}")
            print(f"    indices_input shape: {current_indices_input.shape}, device: {current_indices_input.device}")

        try:
            predictions_q_tensor_batched = loaded_model(
                X_query_tensor_batched,
                mask=current_mask_batched, # This mask (True for padding) is often used to generate attention_mask
                indices=current_indices_input
            )

            if nan_inf_detection_report: # Check if hook found anything during this pass
                print(f"  DEBUG >>> NaN/Inf reported by hooks for q_id {q_id} (see HOOK log above).")

            # Predictions will be for all SLATE_LENGTH items. Slice to get scores for actual documents.
            # The model outputs scores for each of the SLATE_LENGTH positions.
            predictions_for_slate_squeezed = predictions_q_tensor_batched.squeeze(0) # Shape: [SLATE_LENGTH, num_output_scores]

            # We only want predictions for the original, non-padded documents
            predictions_for_actual_docs = predictions_for_slate_squeezed[:num_actual_docs]

            all_predictions_list.append(predictions_for_actual_docs.cpu().numpy())

        except RuntimeError as e_rt:
            print(f"RuntimeError during model inference for q_id {q_id}: {e_rt}")
            print(f"  Input tensor shapes to model (batched): X={X_query_tensor_batched.shape}, mask={current_mask_batched.shape}, indices={current_indices_input.shape}")
            exit()
        except Exception as e_other:
            print(f"An unexpected error for q_id {q_id}: {e_other}")
            # import traceback; traceback.print_exc() # For full trace
            exit()

        # Remove hooks if we are done with the debugged queries
        if DEBUG_HOOKS_ACTIVE and i == QUERIES_TO_DEBUG_WITH_HOOKS - 1 and hooks:
            print(f"  DEBUG_MODE: Removing {len(hooks)} hooks after processing query {q_id} (iteration {i}).")
            for h in hooks: h.remove()
            hooks = []

# Ensure all hooks are removed if loop finishes or breaks early
if hooks:
    print(f"  DEBUG_MODE: Cleaning up {len(hooks)} remaining hooks post-loop...")
    for h in hooks: h.remove()
    hooks = []

if not all_predictions_list:
    print("No predictions were generated. Exiting.")
    exit()

# ... (The rest of your script: consolidate predictions, create DataFrame, etc., this part should largely remain the same
#       as `all_predictions_list` will now correctly contain arrays of scores for the *actual* number of documents per query)

# (The inference loop from your last message should be directly above this)
# It populates `all_predictions_list`

# 5. Process and Consolidate Predictions
print("\nProcessing and consolidating predictions...")
if not all_predictions_list: # Should have already been checked after the loop, but good for safety
    print("No predictions were in all_predictions_list. Exiting.")
    exit() # Or return if in a function

try:
    # Ensure all elements in all_predictions_list are 1D if they represent single scores per doc
    # or handle multi-score outputs appropriately before concatenation.
    processed_predictions_for_concat = []
    for i, pred_array in enumerate(all_predictions_list):
        if not isinstance(pred_array, np.ndarray):
            print(f"  Warning: Element {i} in all_predictions_list is not a NumPy array (type: {type(pred_array)}). Skipping this element for concatenation.")
            continue # Skip non-array elements to prevent error during concatenation

        if pred_array.ndim > 1 and pred_array.shape[1] == 1: # e.g. shape (num_docs, 1)
            processed_predictions_for_concat.append(pred_array.squeeze(axis=1))
        elif pred_array.ndim == 1: # e.g. shape (num_docs,)
            processed_predictions_for_concat.append(pred_array)
        elif pred_array.ndim > 1 and pred_array.shape[1] > 1: # e.g. shape (num_docs, num_classes)
            print(f"  Warning: Prediction array at index {i} has shape {pred_array.shape}. Defaulting to scores from the first column.")
            processed_predictions_for_concat.append(pred_array[:, 0])
        elif pred_array.ndim == 0: # Scalar prediction, unlikely for listwise but handle
             print(f"  Warning: Prediction array at index {i} is a scalar ({pred_array}). Wrapping in array.")
             processed_predictions_for_concat.append(np.array([pred_array]))
        else: # Other unexpected shapes
            print(f"  Warning: Prediction array at index {i} has an unexpected shape {pred_array.shape}. Attempting to use as is or flatten.")
            processed_predictions_for_concat.append(pred_array.flatten())


    if not processed_predictions_for_concat:
        print("Error: No valid prediction arrays to concatenate after processing.")
        exit()

    predicted_scores_np = np.concatenate(processed_predictions_for_concat) # <<<< DEFINED HERE
except ValueError as e:
    print(f"Error concatenating predictions: {e}")
    print("This might happen if processed prediction arrays for different queries have inconsistent structures that cannot be concatenated.")
    # For debugging, print shapes of arrays in processed_predictions_for_concat:
    # for idx, arr in enumerate(processed_predictions_for_concat):
    #     print(f"  Shape of processed prediction array at index {idx}: {arr.shape}")
    exit() # Or return if in a function

# Now, predicted_scores_np is defined.

# Debug print for all_predictions_list (already in your snippet)
if all_predictions_list: # Check if the list itself is not empty
    print("\nDEBUG: Snippet of all_predictions_list (first element, first few scores if available):")
    if all_predictions_list[0].size > 0: # Check if the first prediction array is not empty
        print(all_predictions_list[0][:min(5, len(all_predictions_list[0]))])
    else:
        print("First element of all_predictions_list is empty or has size 0.")

# Sanity check the total number of scores
if len(predicted_scores_np) != X_dense_all.shape[0]: # X_dense_all should be defined from SVM loading
    print(f"!!! Critical Error: Mismatch in the total number of collated prediction scores ({len(predicted_scores_np)}) "
          f"and the total number of samples in the input data ({X_dense_all.shape[0]}).")
    print("This indicates an issue with how predictions were collected or sliced after padding/truncation.")
    exit() # Or return

print(f"Successfully consolidated {len(predicted_scores_np)} scores.")

# Check for NaNs in the final consolidated scores (this was your initial debug print)
if np.isnan(predicted_scores_np).any():
    nan_count = np.sum(np.isnan(predicted_scores_np))
    print(f"!!! WARNING: Final `predicted_scores_np` contains {nan_count} NaN(s) out of {len(predicted_scores_np)} scores!!!")
else:
    print("  Final `predicted_scores_np` is clean of NaNs.")

# 6. Create Pandas DataFrame for Evaluation
# query_ids_svm and y_svm_labels should be defined from the load_svmlight_file step
# and correspond to the original, unpadded/untruncated documents.
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
    print("This can happen if the lengths of query_ids_svm, y_svm_labels, and predicted_scores_np do not match.")
    print(f"  len(query_ids_svm): {len(query_ids_svm) if 'query_ids_svm' in locals() else 'Not defined'}")
    print(f"  len(y_svm_labels): {len(y_svm_labels) if 'y_svm_labels' in locals() else 'Not defined'}")
    print(f"  len(predicted_scores_np): {len(predicted_scores_np)}")
    exit() # Or return

# 7. Add Predicted Ranking
print("\nAdding predicted ranking to DataFrame...")
df_results['predicted_ranking'] = df_results.groupby('query_id')['predicted_score'].rank(
    ascending=False,  # Higher scores get better rank (e.g., 1st)
    method='first'    # Tie-breaking: assign ranks based on order of appearance in group
)

print("Predicted ranking added (showing head for first few query_ids as sample):")
# Display a sample for verification
unique_qids_in_results = df_results['query_id'].unique()
for q_idx, qid_val in enumerate(unique_qids_in_results):
    if q_idx < 3: # Show for the first 3 unique query_ids to keep output concise
        print(f"\nQuery ID: {qid_val}")
        print(df_results[df_results['query_id'] == qid_val].head().sort_values(by='predicted_ranking'))
    else:
        break
if len(unique_qids_in_results) > 3:
    print("... and so on for other query_ids.")

print(f"\nScript section finished at {pd.Timestamp.now(tz='America/Manaus')}") # Using your tz from main()

# After the inference loop
if all_predictions_list:
    print("\nDEBUG: Snippet of all_predictions_list (first element, first few scores):")
    if all_predictions_list[0].size > 0:
        print(all_predictions_list[0][:min(5, len(all_predictions_list[0]))])
    else:
        print("First element of all_predictions_list is empty.")
# ... then proceed to concatenate ...
# After predicted_scores_np = np.concatenate(all_predictions_list)
if np.isnan(predicted_scores_np).any():
    print(f"!!! WARNING: NaNs found in final predicted_scores_np! Count: {np.sum(np.isnan(predicted_scores_np))} !!!")
    # Find which queries might have NaNs if needed
    # for k in range(len(all_predictions_list)):
    #     if np.isnan(all_predictions_list[k]).any():
    #         print(f"NaNs found in predictions for query index {k} (original q_id might differ due to unique_q_ids_svm sorting)")
else:
    print("No NaNs found in final predicted_scores_np.")

# 3. Process and Consolidate Predictions
try:
    predicted_scores_np = np.concatenate(all_predictions_list)
except ValueError as e:
    print(f"Error concatenating predictions: {e}. Check shapes of model outputs from different queries.")
    exit()

# 4. Create Pandas DataFrame and Add Rankings
df_results = pd.DataFrame({
    'query_id': query_ids_svm,         # Original query_ids from SVM file
    'manual_label': y_svm_labels,      # Original labels from SVM file
    'predicted_score': predicted_scores_np # Scores from your model
})

print("\nDataFrame created (first 5 rows):")
print(df_results.head())

# Add predicted_ranking (similar to your df_test logic)
df_results['predicted_ranking'] = df_results.groupby('query_id')['predicted_score'].rank(
    ascending=False, # Higher scores get better rank (1st, 2nd, etc.)
    method='first'   # Tie-breaking: assign ranks in order of appearance within group
)

print("\nDataFrame with predicted ranking (showing head for first few query_ids):")
# Display a sample for verification
unique_qids_in_results = df_results['query_id'].unique()
for q_idx, qid_val in enumerate(unique_qids_in_results):
    if q_idx < 3: # Show for the first 3 unique query_ids
        print(f"\nQuery ID: {qid_val}")
        print(df_results[df_results['query_id'] == qid_val].head().sort_values(by='predicted_ranking'))
    else:
        break
if len(unique_qids_in_results) > 3:
    print("... and so on for other query_ids.")

# Now 'df_results' contains the data in the desired format.
# You can save it or use it for further evaluation.
# Example: df_results.to_csv("evaluation_results.csv", index=False)
# print("\nResults saved to evaluation_results.csv")

# Ensure predicted_scores_np is 1D (or becomes 1D after squeeze/selection)
if predicted_scores_np.ndim > 1:
    if predicted_scores_np.shape[1] == 1: # Shape (N, 1)
        predicted_scores_np = predicted_scores_np.squeeze(axis=1)
    else: # Shape (N, C) where C > 1, e.g., multi-target or class scores
        print(f"Warning: Predictions have {predicted_scores_np.shape[1]} columns. Using scores from the first column for ranking.")
        predicted_scores_np = predicted_scores_np[:, 0]

if len(predicted_scores_np) != X_dense_all.shape[0]:
    print(f"Critical Error: Mismatch in number of collated predictions ({len(predicted_scores_np)}) "
          f"and total samples ({X_dense_all.shape[0]}).")
    exit()
print(f"\nInference complete. Generated {len(predicted_scores_np)} total scores.")

df_test = df_results

import numpy as np
import pandas as pd

def ndcg_at_k(true_relevance, predicted_rank, k=5):

    # Ensure the series are of the same length
    assert len(true_relevance) == len(predicted_rank), "Series must have the same length"

    # Skip queries with less than k documents
    if len(true_relevance) < k:
        return np.nan  # Return NaN for queries with fewer than k documents

    # Sort predicted_rank based on the predicted ranking order
    sorted_true_relevance = true_relevance.iloc[predicted_rank.argsort()]

    # Limit to the top k positions for DCG calculation
    sorted_true_relevance_at_k = sorted_true_relevance.head(k)

    # Calculate DCG@k (Discounted Cumulative Gain) for the top k ranking positions
    dcg_k = np.sum((2 ** sorted_true_relevance_at_k.values - 1) / np.log2(np.arange(2, len(sorted_true_relevance_at_k) + 2)))

    # Calculate IDCG@k (Ideal Discounted Cumulative Gain) for the ideal ranking (sorted by true relevance)
    ideal_rank_at_k = true_relevance.sort_values(ascending=False).head(k)
    idcg_k = np.sum((2 ** ideal_rank_at_k.values - 1) / np.log2(np.arange(2, len(ideal_rank_at_k) + 2)))

    # Calculate NDCG@k
    ndcg_k = dcg_k / idcg_k if idcg_k != 0 else 0  # Avoid division by zero

    return ndcg_k

# Example of processing the entire dataset


ks = [1,3,5]


for k in ks:
    ndcg_total = 0
    valid_queries = 0  # To count queries with at least k documents
    for query_id, subdf in df_test.groupby('query_id'):
        true_relevance = subdf['manual_label']  # True relevance scores from experts
        predicted_rank = subdf['predicted_ranking']  # Predicted ranking positions

        # Calculate NDCG@K for the current query (adjust k as needed)
        ndcg_score = ndcg_at_k(true_relevance, predicted_rank, k)  # Use k=5 as an example

        # Only include queries that have at least k documents
        if not np.isnan(ndcg_score):
            ndcg_total += ndcg_score
            valid_queries += 1

    # Average NDCG over all queries with at least k documents
    average_ndcg = ndcg_total / valid_queries if valid_queries > 0 else 0
    print(f"Average NDCG@{k}: {average_ndcg:.4f}")

