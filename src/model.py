from typing import Callable

import tensorflow as tf
from tqdm import tqdm
import PIL

import src.utils as utils


class NSTImageTrainer(tf.keras.models.Model):
    def __init__(
        self,
        style_image: tf.Tensor,
        content_image: tf.Tensor,
        style_layers: list[str],
        content_layers: list[str],
        content_weight: float = 1e4,
        style_weight: float = 1e-2,
        total_variation_weight: int = 30,
        noise: float = 0,
        *args,
        **kwargs,
    ) -> None:
        """NSTImageTrainer is able to generate image based on style and content images.
        It is based on pretrained vgg model, and Neural Style Transfer technique.

        Args:
            style_image (tf.Tensor): 3D Tensor represents style image
            content_image (tf.Tensor): 3D Tensor represents content image
            style_layers (list[str]): list of all vgg layers names used to calculate style.
                List of all avaliable layers can be returned by NSTImageTrainer.model_layers_names().
            content_layers (list[str]): list of all vgg layers names used to calculate content
                (Usualy one layer is recomended).
                List of all avaliable layers can be returned by NSTImageTrainer.model_layers_names()
            content_weight (float, optional): Weight of content loss. Defaults to 1e4.
            style_weight (float, optional): Weigth of style loss. Defaults to 1e-2.
            total_variation_weight (int, optional): Weight of total variation,
                increasing this should help to reduce artefacts in generated image. Defaults to 30.
            noise (float, optional): Amount of noise added to generated image on the beggining. Defaults to 0.25.

        Tips:
        In default paramteres role of style is much smaller than content. This aproach is suited for CPU training,
        because it gives good results faster. On the other hand if you use GPU,
        then increasing importance of style can give you more interesting images after longer traing.

        If you want to generate image more symilar to content image,
        you should use one of the ealrier layers as content layer, set small noise (or 0)
        and use content_weigth bigger than style weight.

        If you want to generate image more symilar to style image,
        you should use one of the ending layers as content layer, set big noise
        and use style_weigth bigger than content weight.

        """
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.extractor = StyleContentExtractor(style_layers, content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)
        self.style_targets = self.extractor(style_image)["style"]
        self.content_targets = self.extractor(content_image)["content"]
        noise = tf.random.uniform(tf.shape(content_image), -noise, noise)
        output_image = tf.add(content_image, noise)
        output_image = tf.clip_by_value(
            output_image, clip_value_min=0.0, clip_value_max=1.0
        )
        self.__output_image = tf.Variable(output_image)
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.total_variation_weight = total_variation_weight
        self.__style_image = style_image
        self.__content_image = content_image

    def training_loop(
        self,
        steps_per_epoch: int,
        epochs: int,
        callbacks: list[Callable[[], None]] = None,
    ) -> None:
        """Train output image. After every epoch callbacks are called.

        Args:
            epochs (int): Nuber of epoch in training.
            steps (int): Training steps in every epoch. After each step gradiends are applied to image.
            callbacks (list[Callable[[],]], optional): List of callable objects,
            that are called on the end of every trainer training.
            They cannot take any parameters. If you need to use function with parameters,
            wrap it like in exapmle below. Defaults to None.

        Callback wrap examples:

        def callback_without_arg():
            callback(global_arg1, global_arg2)

        or

        class CallbackObj:
            def __init__(arg1, arg2, callback):
                self.arg1 = arg1
                self.arg2 = arg2
                self.callback = callback

            def __call__(self):
                self.callback(self.arg1, self.arg2)
        """
        for epoch_num in range(1, epochs + 1):
            print(f"Epoch {epoch_num}/{epochs}:")
            for _ in tqdm(range(steps_per_epoch)):
                self.train_step()
            if callbacks:
                for callback in callbacks:
                    callback()

    @tf.function()
    @tf.autograph.experimental.do_not_convert
    def train_step(self) -> None:
        """Perform the actual training step."""
        with tf.GradientTape() as tape:
            outputs = self.extractor(self.__output_image)
            loss = self.__style_content_loss(outputs)
            loss += self.__calculate_total_variation()

        grad = tape.gradient(loss, self.__output_image)
        self.optimizer.apply_gradients([(grad, self.__output_image)])
        self.__output_image.assign(utils.tf_utils.clip_0_1(self.__output_image))

    def __style_content_loss(self, outputs: dict[str, dict]) -> float:
        """Compute the style and content loss based on output from StyleContentExtractor.

        Args:
            outputs (dict[str, dict]): output from StyleContentExtractor call.

        Returns:
            float: Total weighted style and content loss.
        """
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

    def __calculate_total_variation(self) -> float:
        """Calculates the total variation of the output image.

        Returns:
            [float]: The total, weighted, variation of output image.
        """
        return self.total_variation_weight * tf.image.total_variation(
            self.__output_image
        )

    @property
    def metrics(self) -> list:
        """Return a list of tracked metrics. In this case only loss is tracked,
            but list is returned because of compatibility with tensorflow.

        Returns:
            [list[tf.keras.metrics.Mean]]: tf.keras.metrics.Mean object wrapped in the list.
        """
        return [self.loss_tracker]

    @property
    def output_image(self) -> PIL.Image:
        """Outputs the image representation of generated image tensor.

        Returns:
            PIL.Image
        """
        return utils.tf_utils.tensor_to_image(self.__output_image)

    @property
    def style_image(self) -> PIL.Image:
        """Outputs the image representation of style image tensor.

        Returns:
            PIL.Image
        """
        return utils.tf_utils.tensor_to_image(self.__style_image)

    @property
    def content_image(self) -> PIL.Image:
        """Outputs the image representation of content image tensor.

        Returns:
            PIL.Image
        """
        return utils.tf_utils.tensor_to_image(self.__content_image)

    @staticmethod
    def model_layers_names() -> list[str]:
        """Returns a list of all model layers names.

        Returns:
            list[str]
        """
        return [layer.name for layer in StyleContentExtractor.base_model.layers]

    @classmethod
    def from_layers_selectors(
        self,
        style_image: tf.Tensor,
        content_image: tf.Tensor,
        style_layers_selector: Callable[[list[str]], list[str]],
        content_layers_selector: Callable[[list[str]], list[str]],
        trainer_kw=dict(total_variation_weight=30),
        style_layers_selector_kw: dict = dict(),
        content_layers_selector_kw: dict = dict(),
    ):
        """Create a NSTImageTrainer from a style_layers_selector and content_layler_selector functions.

        Args:
            style_image (tf.Tensor): 3D Tensor represents style image.
            content_image (tf.Tensor): 3D Tensor represents content image.
            style_layers_selector (Callable[[list[str]], list[str]]): Function, that takes as arugment
                list of all convolutional layers names and return subset of them. Returned list will be used
                as style layers in currently created NSTImageTrainer.
            content_layers_selector (Callable[[list[str]], list[str]]): Function, that takes as arugment
                list of all convolutional layers names and return subset of them. Returned list will be used
                as content layers in currently created NSTImageTrainer.
            trainer_kw (dict, optional): Keyword arguments to create NSTImageTrainer.
                Defaults to {"total_variation_weight": 30}.
            style_layers_selector_kw ([dict], optional): Keywords arguments of style_layers_selector.
                Defaults to dict().
            content_layers_selector_kw ([dict], optional): Keywords arguments of content_layers_selector.
                Defaults to dict().

        Returns:
            NSTImageTrainer
        """
        layers = NSTImageTrainer.model_layers_names()
        conv_layers = [name for name in layers if "conv" in name]

        style_layers = style_layers_selector(conv_layers, **style_layers_selector_kw)
        content_layers = content_layers_selector(
            conv_layers, **content_layers_selector_kw
        )

        trainer = NSTImageTrainer(
            style_image, content_image, style_layers, content_layers, **trainer_kw
        )
        return trainer


class StyleContentExtractor(tf.keras.models.Model):
    base_model = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    base_model.trainable = False

    def __init__(
        self, style_layers: list[str], content_layers: list[str], *args, **kwargs
    ) -> None:
        """StyleContentExtractor implements forward pass of NST image training. It is core of NSTImageTrainer.
            All created StyleContentExtractors share the same VGG19 model as base, to save memory.

        Args:
            style_layers (list[str]): list of all vgg layers names used to calculate style.
                List of all avaliable layers can be returnrd by NSTImageTrainer.model_layers_names()
            content_layers (list[str]): list of all vgg layers names used to calculate content
                (Usualy one layer is recomended).
                List of all avaliable layers can be returned by NSTImageTrainer.model_layers_names()

        """
        super().__init__(*args, **kwargs)
        self.model = self.__init_model(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

    def call(self, inputs: tf.Tensor, input_max_value: int = 1) -> dict[str, dict]:
        """Implement forward pass of the model.

        Args:
            inputs (tf.Tensor): Tensor with image/images batch to process
            input_max_value (int, optional): Maximal value in input image. Defaults to 1.

        Returns:
            [dict[str, dict]]: dict with two keys: "content" and "style",
            value of each key contains neasted dict with layer name as key
            and corespoding model output as value
        """
        content_outputs, style_outputs = self.__get_model_outputs(
            inputs, input_max_value
        )
        content_dict = dict(zip(self.content_layers, content_outputs))
        style_dict = dict(zip(self.style_layers, style_outputs))
        return {"content": content_dict, "style": style_dict}

    def __init_model(self, layer_names: list[str]):
        """Initialize vgg model with outputs specified in layers_names.

        Args:
            layer_names (list[str]): Both, style and content layers used to define output layers of model.

        Returns:
            tf.keras.Model: Vgg19 model with multiple outputs on diffrent layers.
        """
        outputs = [self.base_model.get_layer(name).output for name in layer_names]
        return tf.keras.Model([self.base_model.input], outputs)

    def __get_model_outputs(
        self, inputs: tf.Tensor, input_max_value: int
    ) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
        """Runs the model and returns the content and style matrices.

        Args:
            inputs (tf.Tensor): Tensor with image/images batch to process
            input_max_value (int, optional): Maximal value in input image. Defaults to 1.

        Returns:
            [tuple[list[tf.Tensor], list[tf.Tensor]]]: Two lists with content and style model outputs.
        """
        processed_input = self.__preprocess_input(inputs, input_max_value)
        outputs = self.model(processed_input)
        content_outputs = outputs[self.num_style_layers :]
        style_outputs = map(
            utils.tf_utils.gram_matrix, outputs[: self.num_style_layers]
        )
        return content_outputs, style_outputs

    def __preprocess_input(self, inputs: tf.Tensor, input_max_value) -> tf.Tensor:
        """Preprocess the input to meet vgg requirements.

        Args:
            inputs (tf.Tensor): Tensor with image/images batch to process.
            input_max_value (int, optional): Maximal value in input image. Defaults to 1.

        Returns:
            tf.Tensor: Preprocessed image
        """
        # self.__assert_max_value_not_exceeded(inputs, input_max_value)
        inputs = inputs * (255.0 / input_max_value)
        return tf.keras.applications.vgg19.preprocess_input(inputs)

    # def __assert_max_value_not_exceeded(self, inputs, input_max_value):
    #     biggest_input = tf.reduce_max(inputs)
    #     if biggest_input > input_max_value:
    #         raise ValueError(
    #             f"Given tensor have values grater than {input_max_value = }, {biggest_input = }"
    #         )

    @classmethod
    def model_layers_names(cls) -> list[str]:
        """Returns a list of vgg19 layer names.

        Returns:
            list[str]
        """
        return [layer.name for layer in cls.base_model.layers]
