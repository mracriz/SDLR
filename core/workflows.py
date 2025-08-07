import mlflow
import os
import subprocess
import uuid
from pathlib import Path
import json
from pprint import pformat

def run_command(command, working_directory, error_message):
    """
    Executes a command from a specific working directory and shows output in real-time.
    """
    print(f"\n--- EXECUTING FROM: '{working_directory}' ---")
    print(f"--- COMMAND: {' '.join(command)} ---\n")
    
    result = subprocess.run(command, cwd=working_directory, check=False)
    
    if result.returncode != 0:
        print(f"\nERROR: {error_message}")
        # The specific error will have already been printed to the console
        exit(1)
        
    return result

def prepare_config(template_path_str, data_path, project_root):
    """
    Reads a config template, injects the data path, and saves a temporary config file.
    Returns the absolute path to the temporary file.
    """
    template_path = project_root / template_path_str
    try:
        with open(template_path, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Config template not found at '{template_path}'")
        raise

    # Update the data path in the dictionary
    config_data["data"]["path"] = data_path
    
    # Create a temporary config file for this specific run
    temp_config_filename = f"temp_config_{uuid.uuid4().hex}.json"
    # Save it in the same directory as the template
    temp_config_path = template_path.parent / temp_config_filename
    with open(temp_config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
        
    return temp_config_path

def run_sdlr_workflow(experiment_name, config):
    """
    Orchestrates the full SDLR (Teacher -> Student -> Inference) workflow.
    """
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    project_root = Path(__file__).resolve().parent.parent

    teacher_base_dir = project_root / "Teacher" / "allRank-master"
    student_base_dir = project_root / "Student" / "allRank-master"
    inference_script_path = project_root / "inference.py"
    
    temp_teacher_config = None
    temp_student_config = None

    try:
        # Prepare temporary config files with the correct data paths
        temp_teacher_config = prepare_config(
            config["teacher"]["config_template"],
            config["teacher"]["data_path"],
            project_root
        )
        temp_student_config = prepare_config(
            config["student"]["config_template"],
            config["student"]["data_path"],
            project_root
        )

        tracking_uri = "file:" + os.path.abspath(project_root / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        parent_run_name = f"sdlr_run_{uuid.uuid4().hex[:8]}"

        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            parent_run_id = parent_run.info.run_id
            print(f"Parent Run Started: {parent_run_name} (ID: {parent_run_id})")
            mlflow.set_tag("experiment_type", "sdlr_knowledge_distillation")

            data_options = config.get("data_options", {})

            # STEP 1: TRAINING TEACHER MODEL
            print("\n" + "="*80)
            print("STEP 1: TRAINING TEACHER MODEL")
            print("="*80)
            with mlflow.start_run(run_name="teacher_training", nested=True) as teacher_run:
                teacher_run_id = teacher_run.info.run_id
                mlflow.set_tag("model_role", "teacher")
                
                teacher_command = [
                    "python3", "allrank/main.py",
                    "--job_dir", "allrank",
                    "--run_id", "teacher_run", # Static name for local folder
                    "--config_file_name", str(temp_teacher_config),
                    "--mlflow_run_id", teacher_run_id,
                    "--noise_percent", str(data_options.get("noise_percent", 0.0)),
                    "--max_noise", str(data_options.get("max_noise", 0.0)),
                    "--data_percent", str(data_options.get("data_percent", 1.0)),
                ]
                run_command(teacher_command, teacher_base_dir, "Failed to train Teacher model.")
            
            # STEP 2: TRAINING STUDENT MODEL
            print("\n" + "="*80)
            print("STEP 2: TRAINING STUDENT MODEL")
            print("="*80)
            with mlflow.start_run(run_name="student_training", nested=True) as student_run:
                student_run_id = student_run.info.run_id
                mlflow.set_tag("model_role", "student")
                mlflow.set_tag("teacher_run_id", teacher_run_id)

                student_command = [
                    "python3", "allrank/main.py",
                    "--job_dir", "allrank",
                    "--run_id", "student_run", # Static name for local folder
                    "--config_file_name", str(temp_student_config),
                    "--mlflow_run_id", student_run_id,
                    "--noise_percent", str(data_options.get("noise_percent", 0.0)),
                    "--max_noise", str(data_options.get("max_noise", 0.0)),
                    "--data_percent", str(data_options.get("data_percent", 1.0)),
                ]
                run_command(student_command, student_base_dir, "Failed to train Student model.")
                
                student_model_path = student_base_dir / "allrank" / "results" / "student_run" / "model.pkl"

            # STEP 3: EVALUATING STUDENT MODEL (Optional)
            inference_data_path = config.get("inference_data")
            if inference_data_path and inference_data_path.lower() != 'none':
                print("\n" + "="*80)
                print("STEP 3: EVALUATING STUDENT MODEL")
                print("="*80)
                with mlflow.start_run(run_name="student_inference", nested=True) as inference_run:
                    mlflow.set_tag("evaluation_target", "student")
                    mlflow.set_tag("student_run_id", student_run_id)
                    
                    inference_command = [
                        "python3", str(inference_script_path),
                        "--svm_file_path", inference_data_path,
                        "--model_file_path", str(student_model_path),
                        "--mlflow_run_id", inference_run.info.run_id
                    ]
                    run_command(inference_command, project_root, "Failed to run inference.")
                    
                    client = mlflow.tracking.MlflowClient()
                    inference_metrics = client.get_run(inference_run.info.run_id).data.metrics
                    mlflow.log_metrics({f"final_{k}": v for k, v in inference_metrics.items()}, run_id=parent_run_id)
            else:
                print("\n" + "="*80)
                print("STEP 3: SKIPPING EVALUATION (no inference data provided)")
                print(f"Trained model saved at: {student_model_path}")
                print("="*80)


            print("\n" + "="*80)
            print(f"SDLR WORKFLOW COMPLETE! (Parent Run ID: {parent_run_id})")
            print("="*80)

    finally:
        # Clean up temporary config files
        if temp_teacher_config and os.path.exists(temp_teacher_config):
            os.remove(temp_teacher_config)
        if temp_student_config and os.path.exists(temp_student_config):
            os.remove(temp_student_config)

def run_single_model_workflow(experiment_name, config):
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    project_root = Path(__file__).resolve().parent.parent

    model_base_dir = project_root / config["model"]["base_dir"]
    inference_script_path = project_root / "inference.py"
    
    temp_config_path = None

    try:
        temp_config_path = prepare_config(
            config["model"]["config_template"],
            config["model"]["data_path"],
            project_root
        )

        tracking_uri = "file:" + os.path.abspath(project_root / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        parent_run_name = f"single_run_{uuid.uuid4().hex[:8]}"

        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            parent_run_id = parent_run.info.run_id
            print(f"Parent Run Started: {parent_run_name} (ID: {parent_run_id})")

            data_options = config.get("data_options", {})

            # STEP 1: TRAINING MODEL
            print("\n" + "="*80)
            print("STEP 1: TRAINING MODEL")
            print("="*80)
            
            # This workflow uses unique folders to store models locally to avoid conflicts
            local_run_id = parent_run_name 
            
            with mlflow.start_run(run_name="training", nested=True) as training_run:
                training_run_id = training_run.info.run_id
                
                train_command = [
                    "python3", "allrank/main.py",
                    "--job_dir", "allrank",
                    "--run_id", local_run_id,
                    "--config_file_name", str(temp_config_path),
                    "--mlflow_run_id", training_run_id,
                    "--noise_percent", str(data_options.get("noise_percent", 0.0)),
                    "--max_noise", str(data_options.get("max_noise", 0.0)),
                    "--data_percent", str(data_options.get("data_percent", 1.0)),
                ]
                run_command(train_command, model_base_dir, "Failed to train the model.")
            
            model_path = model_base_dir / "allrank" / "results" / local_run_id / "model.pkl"
            print(f"\n✅ Training complete. Model saved locally at: {model_path}")

            # STEP 2: EVALUATING MODEL (Optional)
            inference_data_path = config.get("inference_data")
            if inference_data_path and inference_data_path.lower() != 'none':
                print("\n" + "="*80)
                print("STEP 2: EVALUATING MODEL")
                print("="*80)
                with mlflow.start_run(run_name="inference", nested=True) as inference_run:
                    inference_run_id = inference_run.info.run_id
                    
                    inference_command = [
                        "python3", str(inference_script_path),
                        "--svm_file_path", inference_data_path,
                        "--model_file_path", str(model_path),
                        "--mlflow_run_id", inference_run_id
                    ]
                    run_command(inference_command, project_root, "Failed to run inference.")
                    
                    client = mlflow.tracking.MlflowClient()
                    inference_metrics = client.get_run(inference_run_id).data.metrics
                    final_metrics_to_log = {f"final_{k}": v for k, v in inference_metrics.items()}
                    mlflow.log_metrics(final_metrics_to_log, run_id=parent_run_id)
                    print(f"\nFinal NDCG metrics saved to Parent Run: {pformat(final_metrics_to_log)}")
            else:
                print("\n" + "="*80)
                print("STEP 2: SKIPPING EVALUATION (no inference data provided)")
                print("="*80)

            print("\n" + "="*80)
            print(f"SINGLE MODEL WORKFLOW COMPLETE! (Parent Run ID: {parent_run_id})")
            print("="*80)

    finally:
        # Clean up temporary config file
        if temp_config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)