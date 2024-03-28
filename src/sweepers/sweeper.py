import os
from loguru import logger
import tensorflow as tf
from wandb.apis.public import Run
from wandb.integration.keras import WandbCallback
import wandb as wab
import wandb.util
from wandb.sdk.wandb_config import Config
from src.hypermodels.hypermodels import WaBHyperModel, InceptionV3WaBHyperModel
from src.utils.datasets import load_datasets


def main():
    """
    Main driver for the hyperparameter search. This function is responsible for initializing the sweep configuration,
    initializing the datasets, initializing the agent managing the sweep, and constructing the WaB HyperModel that the
    sweeps will leverage.

    .. todo:: Add an example for leveraging Custom Metrics with WaB.

    Notes:
        - The sweep configuration specifies the hyperparameters to be varied, the hyperparameter search method, the loss
          function to optimize, and the WaB project and entity to log the results to.
        - An individual trial is a specific combination of hyperparameters to be evaluated. The WaB agent is responsible
          for running however many unique trials are necessary as part of the specified sweep.
        - The WaB HyperModel is a single class responsible for constructing the model specified by the hyperparameters
          specified in the sweep configuration, training the model, and logging the results to WaB. The WaB HyperModel
          has a :meth:`~src.hypermodels.hypermodels.WaBHyperModel.construct_model_run_trial` method that is called by
          the WaB agent for each trial (unique set of hyperparameters). This method is in charge of constructing a new
          model that utilizes the provided set of hyperparameters.

    See Also:
        - To learn more about sweeps for hyperparameter searches in WaB checkout: https://docs.wandb.ai/guides/sweeps
        - To learn more about how to configure the sweep, checkout: https://docs.wandb.ai/guides/sweeps/configuration

    """
    # Note: Do not use the keras object for the optimizer (e.g. Adam(learning_rate=0.001) instead of 'adam')
    # or you get a RuntimeError('Should only create a single instance of _DefaultDistributionStrategy.') this may be a
    # WaB bug which will be addressed in future updates.
    hyperparameters = {
        'num_epochs': {
            'value': 5,
            # 'values': [10, 20, 30]
        },
        'loss': {
            'value': 'binary_crossentropy'
        },
        'conv_layer_activation_function': {
            'value': 'tanh'
        },
        'kernel_size': {
            'value': 11,
        },
        'num_nodes_conv_1': {
            'value': 2**3
        },
        'num_nodes_conv_2': {
            'value': 2**0
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
        },
        'inference_target_conv_layer_name': {
            'values': ['conv_2d_2']
        },
        # For transfer learning:
        'num_thawed_layers': {
            'value': 2
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
        color_mode='rgb', target_size=(75, 75), interpolation='bilinear', keep_aspect_ratio=False,
        train_set_size=0.6, val_set_size=0.2, test_set_size=0.2, seed=SEED, num_partitions=1, batch_size=BATCH_SIZE,
        num_images=800
    )
    '''
    Initialize the WaB HyperModel in charge of setting up and executing individual trials as part of the sweep: 
    '''
    # Construct WaB HyperModel:
    # hypermodel = WaBHyperModel(
    #     train_ds=train_ds,
    #     val_ds=val_ds,
    #     test_ds=test_ds,
    #     num_classes=NUM_CLASSES,
    #     training=True,
    #     batch_size=BATCH_SIZE,
    #     metrics=[
    #         'accuracy', 'binary_accuracy', tf.keras.metrics.BinaryCrossentropy(from_logits=False),
    #         tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(),
    #         tf.keras.metrics.FalseNegatives()
    #     ]
    # )
    # For Transfer Learning with InceptionV3:
    hypermodel = InceptionV3WaBHyperModel(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        num_classes=NUM_CLASSES,
        training=True,
        batch_size=BATCH_SIZE,
        metrics=[
            'accuracy', 'binary_accuracy', tf.keras.metrics.BinaryCrossentropy(from_logits=False),
            tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives()
        ]
    )
    # Initialize the agent in charge of running the sweep:
    wab.agent(
        count=NUM_TRIALS, sweep_id=sweep_id, project='JustRAIGS', entity='appmais', function=hypermodel.construct_model_run_trial
    )


if __name__ == '__main__':
    """
    Main entry point for the sweeper. This script is used to run the hyperparameter search. Define the hyperparameters
    which will remain constant between runs below. The hyperparameters which will be varied should be defined in the
    sweep configuration.
    """
    NUM_TRIALS = 10
    BATCH_SIZE = 16
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
    main()
