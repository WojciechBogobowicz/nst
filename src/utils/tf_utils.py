import tensorflow as tf
import numpy as np
import PIL.Image



def gram_matrix(input_tensor: tf.Tensor) -> float:
    """Compute the gram matrix of a input_tensor.
       Apply to dimensions like: bijc,bijd->bcd

    Args:
        input_tensor (tf.Tensor): 4D tensor

    Returns:
        float: 3D tensor
    """
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def load_img(path_to_img: str) -> tf.Tensor:
    """Loads an image from disk.

    Args:
        path_to_img (str): Path where image is stored

    Returns:
        tf.Tensor: 3D tensor reprezentation of image.
    """
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor: tf.Tensor):
    """Converts a tensor to PIL Image.

    Args:
        tensor (tf.Tensor): 3D tensor reprezentation of image.

    Returns:
        PIL.Image: Image created from tensor.
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def clip_0_1(image):
    """Clips tensor values to values from range 0 to 1.
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
