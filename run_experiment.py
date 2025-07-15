import mlflow
import os
import subprocess
import uuid
from pathlib import Path

def run_command(command, working_directory, error_message):
    """
    Executa um comando no shell a partir de um diretório de trabalho específico,
    mostrando a saída em tempo real para evitar deadlocks.
    """
    print(f"\n--- EXECUTANDO A PARTIR DE '{working_directory}' ---")
    print(f"--- COMANDO: {' '.join(command)} ---\n")
    
    result = subprocess.run(
        command,
        cwd=working_directory,
        check=False
    )
    
    if result.returncode != 0:
        print(f"\nERRO: {error_message}")
        exit(1)
        
    return result

def main():
    """Orchestrates the Teacher-Student training and evaluation workflow."""
    project_root = Path(__file__).resolve().parent

    teacher_base_dir = project_root / "Teacher" / "allRank-master"
    student_base_dir = project_root / "Student" / "allRank-master"
    
    teacher_config_path = str(teacher_base_dir / "allrank" / "in" / "config.json")
    student_config_path = str(student_base_dir / "allrank" / "in" / "config.json")
    
    inference_data = "/Users/david/Documents/phd/JusbrasilData/colecao_especialistas/manual_svm_252.txt"
    inference_script = project_root / "inference.py"
    
    tracking_uri = "file:" + os.path.abspath(project_root / "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow Tracking URI set to: {tracking_uri}")
    
    mlflow.set_experiment("allRank_LTR_Experiments")
    parent_run_name = f"distillation_exp_{uuid.uuid4().hex[:8]}"

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        parent_run_id = parent_run.info.run_id
        print(f"Iniciada a Execução Pai: {parent_run_name} (ID: {parent_run_id})")
        mlflow.set_tag("experiment_type", "knowledge_distillation")

        # --- 1. Child Run: Teacher Training ---
        print("\n" + "="*80)
        print("PASSO 1: TREINANDO O MODELO TEACHER")
        print("="*80)
        with mlflow.start_run(run_name="teacher_training", nested=True) as teacher_run:
            teacher_run_id = teacher_run.info.run_id
            mlflow.set_tag("model_role", "teacher")
            
            teacher_command = [
                "python3", "allrank/main.py",
                "--job_dir", "allrank",
                "--run_id", "teacher_run",
                "--config_file_name", teacher_config_path,
                "--mlflow_run_id", teacher_run_id
            ]
            run_command(teacher_command, teacher_base_dir, "Falha ao treinar o modelo Teacher.")
        
        # --- 2. Child Run: Student Training ---
        print("\n" + "="*80)
        print("PASSO 2: TREINANDO O MODELO STUDENT")
        print("="*80)
        with mlflow.start_run(run_name="student_training", nested=True) as student_run:
            student_run_id = student_run.info.run_id
            mlflow.set_tag("model_role", "student")
            mlflow.set_tag("teacher_run_id", teacher_run_id)

            student_command = [
                "python3", "allrank/main.py",
                "--job_dir", "allrank",
                "--run_id", "student_run",
                "--config_file_name", student_config_path,
                "--mlflow_run_id", student_run_id
            ]
            run_command(student_command, student_base_dir, "Falha ao treinar o modelo Student.")
            
            student_model_path = student_base_dir / "allrank" / "results" / "student_run" / "model.pkl"

        # --- 3. Child Run: Inference ---
        print("\n" + "="*80)
        print("PASSO 3: AVALIANDO O MODELO STUDENT")
        print("="*80)
        with mlflow.start_run(run_name="student_inference", nested=True) as inference_run:
            mlflow.set_tag("evaluation_target", "student")
            mlflow.set_tag("student_run_id", student_run_id)
            
            inference_command = [
                "python3", str(inference_script),
                "--svm_file_path", inference_data,
                "--model_file_path", str(student_model_path),
                "--mlflow_run_id", inference_run.info.run_id
            ]
            run_command(inference_command, project_root, "Falha ao executar a inferência.")
            
            # --- THIS IS THE FIX ---
            # Correctly log the metrics to the parent run.
            client = mlflow.tracking.MlflowClient()
            inference_metrics = client.get_run(inference_run.info.run_id).data.metrics
            mlflow.log_metrics({f"final_{k}": v for k, v in inference_metrics.items()}, run_id=parent_run.info.run_id)
            # --- END OF FIX ---

        print("\n" + "="*80)
        print(f"WORKFLOW COMPLETO CONCLUÍDO! (ID Pai: {parent_run_id})")
        print("="*80)

if __name__ == "__main__":
    main()