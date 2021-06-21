import tensorflow as tf


class _ProjectorBlock(tf.keras.layers.Layer):

    def __init__(self, units: int, last_block: bool, output_dtype=None, **kwargs):
        super().__init__(**kwargs)
        self._units = units
        self._last_block = last_block

        self.dense = tf.keras.layers.Dense(self._units, use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU(max_value=None, dtype=output_dtype)

    def call(self, inputs, **kwargs):
        x = self.dense(inputs)

        if not self._last_block:
            x = self.batch_norm(x)
            x = self.relu(x)
        return x


class _ProjectionLayer(tf.keras.layers.Layer):
    """
    Last projection layer attached directly to the output of the backbone's last later (w/ GlobalAveragePooling)
    """

    def __init__(self, units: int = 8192, **kwargs):
        super().__init__(**kwargs)
        self._units = units

        self.p1 = _ProjectorBlock(units=self._units, last_block=False)
        self.p2 = _ProjectorBlock(units=self._units, last_block=False)
        # Dtype is float32 for the output (activation) which is needed because of the mixed precision
        self.p3 = _ProjectorBlock(units=self._units, last_block=True, output_dtype=tf.float32)

    def call(self, inputs, **kwargs):
        x = self.p1(inputs)
        x = self.p2(x)
        x = self.p3(x)
        return x


class PreprocessingLayer(tf.keras.layers.Layer):
    """
    This layer handles the preprocessing of the images (who would have guessed this based on the name? :D)
    Input images are in range [0, 255] without normalization
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, inputs, **kwargs):
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        return x


class BarlowTwinsModel(tf.keras.models.Model):

    def __init__(self,
                 input_height: int,
                 input_width: int,
                 projection_units: int,
                 load_imagenet: bool = False,
                 inference_mode: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._projection_units = projection_units
        self.input_height = input_height
        self.input_width = input_width
        self.inference_mode = inference_mode

        weights = None
        if load_imagenet:
            weights = "imagenet"

        # Expected model input values are in range [0, 255]
        self.preprocessing = PreprocessingLayer()

        # Backbone output shape is (None, 2048)
        self.backbone = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                                weights=weights,
                                                                input_shape=(self.input_height, self.input_width, 3),
                                                                pooling="avg")

        self.projector = _ProjectionLayer(units=self._projection_units)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.preprocessing(inputs)
        x = self.backbone(x)
        if not self.inference_mode:
            x = self.projector(x)
        return x

    def get_config(self):
        # TODO
        raise NotImplementedError()
