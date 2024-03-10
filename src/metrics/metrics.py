from typing import Optional
import tensorflow as tf
import keras
from keras.metrics import BinaryAccuracy


class BalancedBinaryAccuracy(BinaryAccuracy):
    """
    Computes the balanced binary accuracy between ``y_true`` and ``y_pred`` weighted by the prevalence of each class.
    """

    def __init__(self, name: Optional[str] = 'balanced_binary_accuracy'):
        super().__init__(name=name, dtype=tf.float32, threshold=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the metric by computing the balanced binary accuracy between ``y_true`` and ``y_pred``
        weighted by the prevalence of each class.

        Args:
            y_true:
            y_pred:
            sample_weight:

        Returns:

        See Also:
            https://stackoverflow.com/a/59943572/3429090

        """
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_true = tf.squeeze(y_true, axis=-1)
        y_true_int = tf.cast(y_true, tf.int32)
        if sample_weight is None:
            class_counts = tf.math.bincount(y_true_int)
            class_counts = tf.math.reciprocal_no_nan(tf.cast(class_counts, tf.float32))
            sample_weight = tf.gather(class_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

    # def reset_state(self):
    #     super().reset_state()

    def get_config(self):
        return {'name': self.name}
        # return super().get_config()

    @classmethod
    def from_config(cls, config):
        super().from_config(config)
