import os
import sys
import argparse
from functools import partial
from pprint import pformat
from urllib.parse import urlparse
from dataclasses import asdict, is_dataclass
from collections.abc import MutableMapping

import numpy as np
import torch
from torch import optim
import mlflow
import mlflow.pytorch

# --- Path Correction ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of Path Correction ---

from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank.training.train_utils import fit
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from allrank.models import losses

# Helper function
def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def parse_args():
    parser = argparse.ArgumentParser(description="allRank: an open-source learning-to-rank library")
    parser.add_argument("--job_dir", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--config_file_name", required=True)
    parser.add_argument("--mlflow_run_id", default=None, help="MLflow Run ID to resume logging to.")
    return parser.parse_args()


def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    args = parse_args()

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)
    create_output_dirs(paths.output_dir)

    logger = init_logger(paths.output_dir)

    with mlflow.start_run(run_id=args.mlflow_run_id):
        logger.info(f"Resumed MLflow run with ID: {mlflow.active_run().info.run_id}")
        logger.info(f"Created paths container {paths}")

        config = Config.from_json(paths.config_path)
        logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

        config_dict = {}
        for attr, value in vars(config).items():
            if is_dataclass(value):
                config_dict[attr] = asdict(value)
            else:
                config_dict[attr] = value
        
        flattened_config = flatten(config_dict)

        mlflow.log_params(flattened_config)
        from allrank import config as conf
        mlflow.log_param("noise_percent", conf.Noise_Percent)
        mlflow.log_param("max_noise", conf.Max_Noise)
        mlflow.log_artifact(paths.config_path, "config")

        # Load data
        train_ds, val_ds = load_libsvm_dataset(
            input_path=config.data.path,
            slate_length=config.data.slate_length,
            validation_ds_role=config.data.validation_ds_role,
        )
        n_features = train_ds.shape[-1]
        train_dl, val_dl = create_data_loaders(
            train_ds, val_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)
        
        dev = get_torch_device()
        logger.info(f"Model training will execute on {dev.type}")

        # Create model and optimizer
        model = make_model(
            n_features=n_features,
            fc_model=config.model.fc_model,
            transformer=config.model.transformer,
            post_model=config.model.post_model
        )
        if torch.cuda.device_count() > 1:
            model = CustomDataParallel(model)
        model.to(dev)

        optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
        loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args) if config.lr_scheduler.name else None
        
        # This parameter is used by both original scripts
        conf.BandWidth_LR = 1e-4

        # Run training
        with dummy_context_mgr():
            result = fit(
                model=model, loss_func=loss_func, optimizer=optimizer, scheduler=scheduler,
                train_dl=train_dl, valid_dl=val_dl, config=config, device=dev,
                output_dir=paths.output_dir, tensorboard_output_path=paths.tensorboard_output_path,
                epochs=config.training.epochs, gradient_clipping_norm=config.training.gradient_clipping_norm,
                early_stopping_patience=config.training.early_stopping_patience
            )

        # Log model and final metrics
        mlflow.pytorch.log_model(model, "model")
        logger.info("Modelo PyTorch salvo como artefato no MLflow.")
        if result:
            final_metrics = {f"final_{role}_{m}": v for role, metrics_dict in result.items() if isinstance(metrics_dict, dict) for m, v in metrics_dict.items()}
            mlflow.log_metrics(final_metrics)

        dump_experiment_result(args, config, paths.output_dir, result)
        if urlparse(args.job_dir).scheme == "gs":
            copy_local_to_gs(paths.local_base_output_path, args.job_dir)
        assert_expected_metrics(result, config.expected_metrics)

if __name__ == "__main__":
    from allrank import config as conf
    conf.Noise_Percent = 0.2
    conf.Max_Noise = 0.0 
    conf.Data_Percent = 0.8
    run()