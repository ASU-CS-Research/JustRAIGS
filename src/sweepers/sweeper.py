import os
from loguru import logger
import tensorflow as tf
from wandb.apis.public import Run
from wandb.integration.keras import WandbCallback
import wandb as wab
import wandb.util
from wandb.sdk.wandb_config import Config

from src.hypermodels.hypermodel import HyperModel
from src.utils.datasets import load_datasets

def main():
    # Note: Do not use the keras object for the optimizer (e.g. Adam(learning_rate=0.001) instead of 'adam')
    # or you get a RuntimeError('Should only create a single instance of _DefaultDistributionStrategy.')
    hyperparameters = {
        'num_epochs': {
            'value': 10,
            # 'values': [10, 20, 30]
        },
        'optimizer': {
            'parameters': {
                'type': {
                    'value': 'adam',
                    # 'values': ['adam', 'sgd', 'rmsprop']
                },
                'learning_rate': {
                    'value': 0.001,
                    # 'values': [0.001, 0.01, 0.1]
                }
            }
        }
    }
    sweep_configuration = {
        'method': 'grid',   # 'method': 'random'
        'project': 'JustRAIGS',
        'entity': 'appmais',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        # 'early_terminate': {
        #     'type': 'hyperband',
        #     'min_iter': 3
        # }
        'parameters': hyperparameters
    }
    sweep_id = wab.sweep(sweep=sweep_configuration, project='JustRAIGS', entity='appmais')
    '''
    Initialize TensorFlow datasets:
    '''
    train_ds, val_ds, test_ds = load_datasets(
        color_mode='rgb', target_size=(64, 64), interpolation='bilinear', keep_aspect_ratio=False,
        train_set_size=0.6, val_set_size=0.2, test_set_size=0.2, seed=42
    )
    # Construct KerasTuner HyperModel:
    hypermodel = HyperModel()
    # Initialize the agent in charge of running the sweep:
    wab.agent(
        count=NUM_TRIALS, sweep_id=sweep_id, project='JustRAIGS', entity='appmais', function=hypermodel.construct_model_run_trial
    )



if __name__ == '__main__':
    """
    Main entry point for the sweeper. This script is used to run the hyperparameter search. Define the hyperparameteres
    which will remain constant between runs below. The hyperparameters which will be varied should be defined in the
    sweep configuration.
    """
    NUM_TRIALS = 10
    BATCH_SIZE = 25
    NUM_CLASSES = 2
    SEED = 42
    REPO_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    LOG_DIR = os.path.join(REPO_ROOT_DIR, 'logs')
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    os.environ['WANDB_DIR'] = LOG_DIR
    LOCAL_DATA_DIR = os.path.join(REPO_ROOT_DIR, 'data')
    if not os.path.exists(LOCAL_DATA_DIR):
        os.makedirs(LOCAL_DATA_DIR)
    DATA_DIR = '/usr/local/data/JustRAIGS/raw/'
    logger.debug(f"WANDB_DIR: {LOG_DIR}")
    logger.debug(f"LOCAL_DATA_DIR: {LOCAL_DATA_DIR}")
    logger.debug(f"DATA_DIR: {DATA_DIR}")


    tf.random.set_seed(seed=SEED)