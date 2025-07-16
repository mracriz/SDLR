import mlflow
import os
import subprocess
import uuid
from pathlib import Path
import argparse

def parse_args():
    """
    Define e lê os argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(description="Orquestrador de Experimentos de Destilação de Conhecimento com MLflow.")
    
    parser.add_argument(
        "experiment_name",
        help="Nome do experimento a ser usado ou criado no MLflow."
    )
    parser.add_argument(
        "--teacher_dir", 
        default="Teacher/allRank-master",
        help="Caminho para o diretório base do Teacher."
    )
    parser.add_argument(
        "--student_dir", 
        default="Student/allRank-master",
        help="Caminho para o diretório base do Student."
    )
    parser.add_argument(
        "--teacher_config",
        default="allrank/in/config.json",
        help="Caminho relativo para o arquivo de configuração do Teacher."
    )
    parser.add_argument(
        "--student_config",
        default="allrank/in/config.json",
        help="Caminho relativo para o arquivo de configuração do Student."
    )
    parser.add_argument(
        "--inference_data",
        default="/Users/david/Documents/phd/JusbrasilData/colecao_especialistas/manual_svm_252.txt",
        help="Caminho absoluto para o arquivo SVM de avaliação."
    )
    parser.add_argument(
        "--inference_script",
        default="inference.py",
        help="Nome do script de inferência."
    )
    return parser.parse_args()


def run_command(command, working_directory, error_message):
    """Executa um comando no shell a partir de um diretório de trabalho específico."""
    print(f"\n--- EXECUTANDO A PARTIR DE '{working_directory}' ---")
    print(f"--- COMANDO: {' '.join(command)} ---\n")
    
    result = subprocess.run(command, cwd=working_directory, check=False)
    
    if result.returncode != 0:
        print(f"\nERRO: {error_message}")
        exit(1)
        
    return result

def main():
    """Orquestra o fluxo de trabalho completo."""
    args = parse_args()

    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    
    project_root = Path(__file__).resolve().parent

    teacher_base_dir = project_root / args.teacher_dir
    student_base_dir = project_root / args.student_dir
    teacher_config_path = str(teacher_base_dir / args.teacher_config)
    student_config_path = str(student_base_dir / args.student_config)
    inference_script_path = project_root / args.inference_script

    tracking_uri = "file:" + os.path.abspath(project_root / "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow Tracking URI set to: {tracking_uri}")
    
    mlflow.set_experiment(args.experiment_name)
    print(f"Usando o experimento do MLflow: '{args.experiment_name}'")

    parent_run_name = f"distillation_exp_{uuid.uuid4().hex[:8]}"

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        parent_run_id = parent_run.info.run_id
        print(f"Parent Run Started: {parent_run_name} (ID: {parent_run_id})")
        mlflow.set_tag("experiment_type", "knowledge_distillation")

        # PASSO 1: TREINANDO O MODELO TEACHER
        print("\n" + "="*80)
        print("STEP 1: TRAINING TEACHER MODEL")
        print("="*80)
        with mlflow.start_run(run_name="teacher_training", nested=True) as teacher_run:
            teacher_run_id = teacher_run.info.run_id
            mlflow.set_tag("model_role", "teacher")
            
            # --- MUDANÇA AQUI ---
            # Usando um ID estático para a pasta de resultados do Teacher
            teacher_command = [
                "python3", "allrank/main.py",
                "--job_dir", "allrank",
                "--run_id", "teacher_run", # <-- Nome da pasta será sempre "teacher_run"
                "--config_file_name", teacher_config_path,
                "--mlflow_run_id", teacher_run_id
            ]
            run_command(teacher_command, teacher_base_dir, "Failed to train Teacher model.")
        
        # PASSO 2: TREINANDO O MODELO STUDENT
        print("\n" + "="*80)
        print("STEP 2: TRAINING STUDENT MODEL")
        print("="*80)
        with mlflow.start_run(run_name="student_training", nested=True) as student_run:
            student_run_id = student_run.info.run_id
            mlflow.set_tag("model_role", "student")
            mlflow.set_tag("teacher_run_id", teacher_run_id)

            # --- MUDANÇA AQUI ---
            # Usando um ID estático para a pasta de resultados do Student
            student_command = [
                "python3", "allrank/main.py",
                "--job_dir", "allrank",
                "--run_id", "student_run", # <-- Nome da pasta será sempre "student_run"
                "--config_file_name", student_config_path,
                "--mlflow_run_id", student_run_id
            ]
            run_command(student_command, student_base_dir, "Failed to train Student model.")
            
            # --- MUDANÇA AQUI ---
            # O caminho para o modelo agora é estático novamente
            student_model_path = student_base_dir / "allrank" / "results" / "student_run" / "model.pkl"

        # PASSO 3: AVALIANDO O MODELO STUDENT
        print("\n" + "="*80)
        print("STEP 3: EVALUATING STUDENT MODEL")
        print("="*80)
        with mlflow.start_run(run_name="student_inference", nested=True) as inference_run:
            mlflow.set_tag("evaluation_target", "student")
            mlflow.set_tag("student_run_id", student_run_id)
            
            inference_command = [
                "python3", str(inference_script_path),
                "--svm_file_path", args.inference_data,
                "--model_file_path", str(student_model_path),
                "--mlflow_run_id", inference_run.info.run_id
            ]
            run_command(inference_command, project_root, "Failed to run inference.")
            
            client = mlflow.tracking.MlflowClient()
            inference_metrics = client.get_run(inference_run.info.run_id).data.metrics
            mlflow.log_metrics({f"final_{k}": v for k, v in inference_metrics.items()}, run_id=parent_run_id)

        print("\n" + "="*80)
        print(f"WORKFLOW COMPLETE! (Parent Run ID: {parent_run_id})")
        print("="*80)

if __name__ == "__main__":
    main()