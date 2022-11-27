import tensorflow as tf
import IPython.display as display

import src.utils as utils


class NSTTrainer(tf.keras.models.Model):
    def __init__(
        self,
        style_image,
        content_image,
        style_layers: list[str],
        content_layers: list[str],
        content_weight=1e4,
        style_weight=1e-2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.extractor = StyleContentExtractor(style_layers, content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)
        self.style_targets = self.extractor(style_image)["style"]
        self.content_targets = self.extractor(content_image)["content"]
        self.__output_image = tf.Variable(content_image)
        self.style_weight = style_weight
        self.content_weight = content_weight

    def training_loop(self, steps_per_epoch, epochs):
        step = 0
        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                step += 1
                self.train_step()
                print(".", end="", flush=True)
        display.clear_output(wait=True)
        display.display(self.output_image)
        print("Train step: {}".format(step))

    @tf.function()
    def train_step(self):
        with tf.GradientTape() as tape:
            outputs = self.extractor(self.__output_image)
            loss = self.style_content_loss(outputs)

        grad = tape.gradient(loss, self.__output_image)
        self.optimizer.apply_gradients([(grad, self.__output_image)])
        self.__output_image.assign(utils.tf_utils.clip_0_1(self.__output_image))

    def style_content_loss(self, outputs):
        style_outputs = outputs["style"]
        content_outputs = outputs["content"]
        style_loss = tf.add_n(
            [
                tf.reduce_mean((style_outputs[name] - self.style_targets[name]) ** 2)
                for name in style_outputs.keys()
            ]
        )
        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n(
            [
                tf.reduce_mean(
                    (content_outputs[name] - self.content_targets[name]) ** 2
                )
                for name in content_outputs.keys()
            ]
        )
        content_loss *= self.content_weight / self.num_content_layers
        loss = style_loss + content_loss
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

    @property
    def output_image(self):
        return utils.tf_utils.tensor_to_image(self.__output_image)


class StyleContentExtractor(tf.keras.models.Model):
    base_model = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    base_model.trainable = False
    
    def __init__(
        self, style_layers: list[str], content_layers: list[str], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = self.__init_model(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

    def call(self, inputs, input_max_value=1):
        content_outputs, style_outputs = self.__get_model_outputs(
            inputs, input_max_value
        )
        content_dict = dict(zip(self.content_layers, content_outputs))
        style_dict = dict(zip(self.style_layers, style_outputs))
        return {"content": content_dict, "style": style_dict}

    def __init_model(self, layer_names: list[str]):
        outputs = [self.base_model.get_layer(name).output for name in layer_names]
        return tf.keras.Model([self.base_model.input], outputs)

    def __get_model_outputs(self, inputs, input_max_value=tf.constant(1, tf.float16)):
        processed_input = self.__preprocess_input(inputs, input_max_value)
        outputs = self.model(processed_input)
        content_outputs = outputs[self.num_style_layers :]
        style_outputs = map(utils.tf_utils.gram_matrix, outputs[: self.num_style_layers])
        return content_outputs, style_outputs

    def __preprocess_input(self, inputs, input_max_value):
        # self.__assert_max_value_not_exceeded(inputs, input_max_value)
        inputs = inputs * (255.0 / input_max_value)
        return tf.keras.applications.vgg19.preprocess_input(inputs)

    def __assert_max_value_not_exceeded(self, inputs, input_max_value):
        biggest_input = tf.reduce_max(inputs)
        if biggest_input > input_max_value:
            raise ValueError(
                f"Given tensor have values grater than {input_max_value = }, {biggest_input = }"
            )

