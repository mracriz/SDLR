import mlflow
import os
import subprocess
import uuid
from pathlib import Path
import argparse
from pprint import pformat  # <-- THIS IS THE FIX

def parse_args():
    """
    Defines and reads the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Orchestrator to train and evaluate a single model with MLflow.")
    
    parser.add_argument(
        "experiment_name",
        help="Name of the experiment to be used or created in MLflow."
    )
    parser.add_argument(
        "config_file",
        help="Relative path to the model's config file (e.g., allrank/in/config_neuralNDCG.json)."
    )
    parser.add_argument(
        "--model_dir", 
        default="Teacher/allRank-master",
        help="Base directory of the model to be trained (usually 'Teacher/allRank-master')."
    )
    parser.add_argument(
        "--inference_data",
        default="/home/david.acris/Documents/data/manual_collection/manual_svm_252.txt",
        help="Absolute path to the evaluation SVM file."
    )
    return parser.parse_args()

def run_command(command, working_directory, error_message):
    """Executes a command from a specific working directory."""
    print(f"\n--- EXECUTING FROM: '{working_directory}' ---")
    print(f"--- COMMAND: {' '.join(command)} ---\n")
    
    result = subprocess.run(command, cwd=working_directory, check=False)
    
    if result.returncode != 0:
        print(f"\nERROR: {error_message}")
        exit(1)
        
    return result

def main():
    """Orchestrates the complete training and evaluation workflow."""
    args = parse_args()
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    
    project_root = Path(__file__).resolve().parent
    model_base_dir = project_root / args.model_dir
    config_path = str(model_base_dir / args.config_file)
    inference_script_path = project_root / "inference.py"

    tracking_uri = "file:" + os.path.abspath(project_root / "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    print(f"Using MLflow experiment: '{args.experiment_name}'")

    parent_run_name = f"single_run_{Path(args.config_file).stem}_{uuid.uuid4().hex[:8]}"

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        parent_run_id = parent_run.info.run_id
        print(f"Parent Run Started: {parent_run_name} (ID: {parent_run_id})")
        
        # STEP 1: TRAINING THE MODEL (in a child run)
        print("\n" + "="*80)
        print("STEP 1: TRAINING MODEL")
        print("="*80)
        with mlflow.start_run(run_name="training", nested=True) as training_run:
            training_run_id = training_run.info.run_id
            
            train_command = [
                "python3", "allrank/main.py",
                "--job_dir", "allrank",
                "--run_id", parent_run_name, 
                "--config_file_name", config_path,
                "--mlflow_run_id", training_run_id
            ]
            run_command(train_command, model_base_dir, "Failed to train the model.")
        
        model_path = model_base_dir / "allrank" / "results" / parent_run_name / "model.pkl"

        # STEP 2: EVALUATING THE MODEL (in another child run)
        print("\n" + "="*80)
        print("STEP 2: EVALUATING MODEL")
        print("="*80)
        with mlflow.start_run(run_name="inference", nested=True) as inference_run:
            inference_run_id = inference_run.info.run_id
            
            inference_command = [
                "python3", str(inference_script_path),
                "--svm_file_path", args.inference_data,
                "--model_file_path", str(model_path),
                "--mlflow_run_id", inference_run_id
            ]
            run_command(inference_command, project_root, "Failed to run inference.")
            
            # Copy final NDCG@k metrics to the Parent Run
            client = mlflow.tracking.MlflowClient()
            inference_metrics = client.get_run(inference_run_id).data.metrics
            final_metrics_to_log = {f"final_{k}": v for k, v in inference_metrics.items()}
            mlflow.log_metrics(final_metrics_to_log, run_id=parent_run_id)
            print(f"\nFinal NDCG metrics saved to Parent Run: {pformat(final_metrics_to_log)}")

        print("\n" + "="*80)
        print(f"WORKFLOW COMPLETE! (Parent Run ID: {parent_run_id})")
        print("="*80)

if __name__ == "__main__":
    main()