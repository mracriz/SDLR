import argparse
import yaml
import sys
import os
from core.workflows import run_sdlr_workflow, run_single_model_workflow

def main():
    parser = argparse.ArgumentParser(
        description="Main runner for SDLR and baseline Learning to Rank experiments."
    )
    parser.add_argument(
        "experiment_name",
        help="The name of the experiment to run, as defined in experiments.yaml."
    )
    args = parser.parse_args()

    # Load the experiments control panel
    try:
        with open("experiments.yaml", "r") as f:
            experiments = yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: experiments.yaml not found. Please create it in the root directory.")
        sys.exit(1)

    # Check if the requested experiment exists
    if args.experiment_name not in experiments:
        print(f"ERROR: Experiment '{args.experiment_name}' not found in experiments.yaml.")
        print("Available experiments are:", list(experiments.keys()))
        sys.exit(1)

    config = experiments[args.experiment_name]
    
    print(f"🚀 Starting experiment: {args.experiment_name}")

    # Decide which workflow to run based on the 'type' field
    if config.get("name") == "sdlr":
        run_sdlr_workflow(experiment_name=args.experiment_name, config=config)
    elif config.get("name") == "single":
        run_single_model_workflow(experiment_name=args.experiment_name, config=config)
    else:
        print(f"ERROR: Unknown experiment type '{config.get('type')}' in experiments.yaml. Must be 'sdlr' or 'single'.")

if __name__ == "__main__":
    # Ensure the core module can be found
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()