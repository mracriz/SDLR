import argparse
import yaml
import sys
import os

# Ensure the 'core' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from core.workflows import run_sdlr_workflow, run_single_model_workflow

def main():
    parser = argparse.ArgumentParser(
        description="Unified Orchestrator for Learning to Rank Experiments."
    )
    parser.add_argument(
        "experiment_key",
        help="The key of the experiment to run, as defined as a top-level key in experiments.yaml."
    )
    args = parser.parse_args()

    # Load the experiments control panel
    try:
        with open("experiments.yaml", "r") as f:
            experiments = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ ERROR: experiments.yaml not found. Please create it in the root directory.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ ERROR: Invalid format in experiments.yaml file: {e}")
        sys.exit(1)

    # Check if the requested experiment exists
    if args.experiment_key not in experiments:
        print(f"❌ ERROR: Experiment key '{args.experiment_key}' not found in experiments.yaml.")
        print("Available experiments are:", list(experiments.keys()))
        sys.exit(1)

    config = experiments[args.experiment_key]
    
    # Get the MLflow experiment name from the config, with a fallback
    mlflow_experiment_name = config.get("experiment_name", args.experiment_key)
    
    print(f"🚀 Starting experiment run: {args.experiment_key}")
    print(f"   Logging to MLflow experiment: '{mlflow_experiment_name}'")

    # Decide which workflow to run based on the 'name' key
    workflow_name = config.get("name")
    if workflow_name == "sdlr":
        run_sdlr_workflow(experiment_name=mlflow_experiment_name, config=config)
    elif workflow_name == "single":
        run_single_model_workflow(experiment_name=mlflow_experiment_name, config=config)
    else:
        print(f"❌ ERROR: Unknown workflow name '{workflow_name}'. Must be 'sdlr' or 'single'.")

if __name__ == "__main__":
    main()