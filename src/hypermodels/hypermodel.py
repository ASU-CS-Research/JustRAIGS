from typing import Tuple, Union, List, Optional
import tensorflow as tf
import keras_tuner as kt
from loguru import logger
from tensorflow.keras.metrics import Metric
from tensorflow.keras import Model
from tensorflow.data import Dataset
from tensorflow.keras.losses import Loss


class KerasTunerWaBHyperModel(kt.HyperModel):
    """
    This code is designed to run on the lambda machine compute server with a NFS mount to the AppMAIS data storage
    server.
    """

    def __init__(
            self, input_image_shape: Tuple[int, int, int], num_classes: int, loss: Union[Loss, str],
            metrics: List[Union[Metric, str]], hyperparameters: Optional[kt.HyperParameters] = None,
            resampled_steps_per_epoch: Optional[int] = None):
        """
        This method should accept hyperparameters which remain constant during hyperparameter tuning.

        .. todo:: Docstrings.

        Args:
             resampled_steps_per_epoch (Optional[int]): In the event that the data has been resampled (i.e. for class
               balancing) then this value is expected to be provided. It indicates how many steps per epoch should be
               taken before every element of the positive/majority class will have been seen at least once. This
               information is required because during the oversampling process the validation and training datasets are
               made to have an infinite cardinality.

        """
        self._input_image_shape = input_image_shape
        self._vgg_19_input_image_shape = (224, 224, 3)
        self._num_classes = num_classes
        self._optimizer = None
        self._loss = loss
        self._metrics = metrics
        self.hyperparameters = hyperparameters
        self._resampled_steps_per_epoch = resampled_steps_per_epoch
        self._model = None
        super().__init__(name='KerasTunerWaBHyperModel', tunable=True)


    def build(self, hp: kt.HyperParameters, weights: Optional[str] = 'imagenet', *args, **kwargs) -> Model:
        """
        This method setups up the hyperparameter grid and search space.

        See Also:
            https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.from_pretrained_keras
        """
        logger.info(f"Building hypermodel with hyperparameters: {hp.values}")
        input_layer = tf.keras.layers.Input(shape=self._vgg_19_input_image_shape)
        preprocessing_layer = tf.keras.applications.vgg19.preprocess_input(input_layer)
        base_model = VGG19(include_top=False, weights=weights, input_shape=self._vgg_19_input_image_shape,
                           classes=self._num_classes)
        base_model.layers[0] = preprocessing_layer
        # Freeze the base model:
        for layer in base_model.layers:
            layer.trainable = False
        # Thaw the last layers if specified in the hyperparameters:
        if 'num_thawed_layers' in hp:
            for i in range(hp['num_thawed_layers']):
                base_model.layers[-1 * i].trainable = True
        # Add new head to the model (i.e. new Dense fully connected layer and softmax):
        model_head = Flatten()(base_model.outputs[0])
        model_head = tf.keras.layers.Dense(self._num_classes - 1, activation='softmax')(model_head)
        model = Model(inputs=base_model.inputs, outputs=model_head)
        model.build((None,) + self._vgg_19_input_image_shape)
        # compile the model:
        model.summary()
        # Reset the optimizer state (from any previous trials):
        self._optimizer = optimizers.Adam(learning_rate=hp['adam_learning_rate'])
        model.compile(loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)
        # self._optimizer.build(model.trainable_variables)
        return model
