import random
import itertools

import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.stats
import sewar.full_ref

import src.utils as utils


class MultiNSTImageTrainer:
    def __init__(self, style_image, content_image, trainers_num) -> None:
        self.__trainers = self.__init_trainers(style_image, content_image, trainers_num)
        self.history = [self.__trainers]

    def __init_trainers(self, style_image, content_image, trainers_num):
        print("Trainers creation")
        trainers = []
        for _ in tqdm(range(trainers_num)):
            trainer = self.__create_random_compiled_trainer(style_image, content_image)
            trainers.append(trainer)
        return pd.Series(trainers)

    def __create_random_compiled_trainer(
        self, style_image, content_image, total_variation_weight=30
    ):
        style_layers, content_layers = self.__get_random_layers()
        trainer = NSTImageTrainer(
            style_image,
            content_image,
            style_layers,
            content_layers,
            total_variation_weight,
        )
        compiler = tf.keras.optimizers.Adam(
            learning_rate=0.02, beta_1=0.99, epsilon=1e-1
        )
        trainer.compile(compiler)  # LBFGS do obczajenia
        return trainer

    @staticmethod
    def __get_random_layers() -> tuple[list[str], list[str]]:
        conv_layers = [
            layer_name
            for layer_name in NSTImageTrainer.model_layers_names()
            if "conv" in layer_name
        ]
        conv_layers_num = len(conv_layers)

        content_weights_idx = range(conv_layers_num)
        content_weights = scipy.stats.norm.pdf(
            content_weights_idx, loc=conv_layers_num / 2, scale=conv_layers_num / 3.75
        )
        content_weights /= sum(content_weights)
        content_layers = random.choices(conv_layers, weights=content_weights, k=1)

        skip_layers = 4
        style_weights_idx = range(conv_layers_num - skip_layers)
        style_layers_num_weights = scipy.stats.norm.pdf(
            style_weights_idx,
            loc=(conv_layers_num - skip_layers) / 20,
            scale=conv_layers_num / 4,
        )
        style_layers_num_weights /= sum(style_layers_num_weights)
        style_layers_num_weights = np.cumsum(style_layers_num_weights)
        style_layers_num = (
            random.random() > style_layers_num_weights
        ).sum() + skip_layers
        style_layers = random.sample(conv_layers, k=style_layers_num)
        style_layers.sort()
        return style_layers, content_layers

    def train(self, epochs, steps) -> None:
        for i, trainer in enumerate(self.trainers):
            print(f"Traing for trainer {i} from {len(self.trainers)-1}")
            trainer.training_loop(epochs=epochs, steps_per_epoch=steps)

    def save_history(self):
        self.history.append(self.__trainers)

    def remove_second_half_trainers(self) -> None:
        self.__trainers = self.__trainers[: len(self.trainers) // 2]

    def sort_trainers_by_differences(self, method):
        t1, t2 = zip(*itertools.product(self.__trainers, self.__trainers))
        df = pd.DataFrame([t1, t2]).transpose()
        df.columns = ["t1", "t2"]
        df["dst"] = df.apply(
            lambda x: method(np.array(x[0].output_image), np.array(x[1].output_image)),
            axis=1,
        )
        df = df.set_index(["t1", "t2"]).unstack()
        df = df.mean(axis=1).sort_values(ascending=False)
        self.__trainers = list(df.index)

    @property
    def trainers(self):
        return self.__trainers


class NSTImageTrainer(tf.keras.models.Model):
    def __init__(
        self,
        style_image,
        content_image,
        style_layers: list[str],
        content_layers: list[str],
        content_weight=1e4,
        style_weight=1e-2,
        total_variation_weight=30,
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
        self.total_variation_weight = total_variation_weight
        self.__style_image = style_image
        self.__content_image = content_image

    def training_loop(self, steps_per_epoch, epochs):
        for epoch_num in range(1, epochs + 1):
            print(f"Epoch {epoch_num}/{epochs}:")
            for _ in tqdm(range(steps_per_epoch)):
                self.train_step()

    @tf.function()
    @tf.autograph.experimental.do_not_convert
    def train_step(self):
        with tf.GradientTape() as tape:
            outputs = self.extractor(self.__output_image)
            loss = self.style_content_loss(outputs)
            loss += self.__calculate_total_vairation()

        grad = tape.gradient(loss, self.__output_image)
        self.optimizer.apply_gradients([(grad, self.__output_image)])
        self.__output_image.assign(utils.tf_utils.clip_0_1(self.__output_image))

    def style_content_loss(self, outputs):
        # ToDo: dodać 2 hiperparetry do -> loss = alpha*style_loss + beta*content_loss
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

    def __calculate_total_vairation(self):
        return self.total_variation_weight * tf.image.total_variation(
            self.__output_image
        )

    @property
    def metrics(self):
        return [self.loss_tracker]

    @property
    def output_image(self):
        return utils.tf_utils.tensor_to_image(self.__output_image)

    @property
    def style_image(self):
        return utils.tf_utils.tensor_to_image(self.__style_image)

    @property
    def content_image(self):
        return utils.tf_utils.tensor_to_image(self.__content_image)

    @staticmethod
    def model_layers_names():
        return [layer.name for layer in StyleContentExtractor.base_model.layers]


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
        style_outputs = map(
            utils.tf_utils.gram_matrix, outputs[: self.num_style_layers]
        )
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

    @staticmethod
    def model_layers_names():
        return [layer.name for layer in StyleContentExtractor.base_model.layers]
