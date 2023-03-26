import itertools
from typing import Callable

import tensorflow as tf
import pandas as pd
import numpy as np


class TraintersSelectioner:
    def __init__(self, trainers: list) -> None:
        """TraintersSelectioner provides methods, that can help in, or automate,
            process of choosing best hyperparameters for NSTImageTrainers.

        Args:
            trainers (list[NSTImageTrainer]): List of trainers to select from.
        """
        self.__trainers = pd.Series(trainers)
        self.history = []

    def train(
        self,
        epochs: int,
        steps: int,
        callbacks: list[Callable[[], None]] = None,
    ) -> None:
        """Train all the NSTImageTrainers. After every trainer training callbacks are called.

        Args:
            epochs (int): Nuber of epoch to train for single trainer.
            steps (int): Training steps in every epoch. After each step gradiends are applied.
            callbacks (list[Callable[[],]], optional): List of callable objects,
                that are called at the end of every trainer training.
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
        for i, trainer in enumerate(self.trainers):
            print(f"Traing for trainer {i+1} from {len(self.trainers)}")
            trainer.training_loop(epochs=epochs, steps_per_epoch=steps)
            if callbacks:
                for callback in callbacks:
                    callback()

    def save_history(self) -> None:
        """Save history of trainers. Be aware that trainers aren't copied,
        so in practice only current trainers order is saved, 
        in order to check what happend with particular trainer,
        or when you don't need to worry about removing trainers.
        """
        self.history.append(self.__trainers)

    def sort_trainers_by_differences(
        self, ordering_method: Callable[[tf.Tensor, tf.Tensor], float]
    ) -> None:
        """Apply ordering_method to all pairs of trained images, and average results per trainer,
            after that, sorts trainers in descending order, based on those averages.

        Args:
            ordering_method (Callable[[tf.Tensor, tf.Tensor], float]): Function that input two 3D tensors
                and return one number. The higher the number, 
                the earlier the trainers colerated with images will be in trainers list.
        """
        t1, t2 = zip(*itertools.product(self.__trainers, self.__trainers))
        df = pd.DataFrame([t1, t2]).transpose()
        df.columns = ["t1", "t2"]
        df["dst"] = df.apply(
            lambda x: ordering_method(
                np.array(x[0].output_image), np.array(x[1].output_image)
            ),
            axis=1,
        )
        df = df.set_index(["t1", "t2"]).unstack()
        df = df.mean(axis=1).sort_values(ascending=False)
        self.__trainers = list(df.index)

    def remove_second_half_trainers(self) -> None:
        """Removes the second half of trainers from the training list.
        Typiacally used after sorting trainers by differences.
        """
        self.__trainers = self.__trainers[: len(self.trainers) // 2]

    @property
    def trainers(self) -> list:
        """The list of all active trainers.

        Returns:
            list[NSTImageTrainers]
        """
        return self.__trainers
