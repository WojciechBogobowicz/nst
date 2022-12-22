import random
import scipy

import numpy as np

    
def normal_choice(data: list[any], rel_loc: float=0.5, rel_scale: float=0.5) -> list[any]:  # 0.5, 0.26
    """Weighted random choice of element from list. Weights are generated from normal distribution.
        Loc of the normal distribution is equal to rel_loc * number of elements in data.
        Analogous scale is equal to rel_scale * number of elements in data.

    Args:
        data (list[any]): Data collection to choose from.
        rel_loc (float, optional): Relative loc of normal distribution. 
            True loc is calculate based on that with the formula: 
            loc = rel_loc * number of elements in data. Defaults to 0.5.
        rel_scale (float, optional): Relative scale of normal distribution. 
            True scale is calculate based on that with the formula: 
            scale = rel_scale * number of elements in data. Defaults to 0.5.
    Returns:
        list[any]: List with single choosen element.
    """
    elements_count = len(data)

    elements_indexes = range(elements_count)
    elements_weights = scipy.stats.norm.pdf(
        elements_indexes, loc=elements_count * rel_loc, scale=elements_count * rel_scale
    )
    elements_weights /= sum(elements_weights)
    return random.choices(data, weights=elements_weights, k=1)


def random_length_choices(
    data: list[any],  
    min_output_elements_num: int=1, 
    rel_loc: float=0.5, 
    rel_scale: float=0.5
    ) -> list[any]:  # 4, 0.05, 0.25
    """Get subset of data, uniformly sampled, with sample size sampled from normal dystribution.

    Args:
        data (list[any]): Data collection to choose from.
        min_output_elements_num (int, optional): Minimal number of of output elments,
            that needs to be satisfied. Defaults to 1.
        rel_loc (float, optional): Relative loc of normal distribution, 
            used to sample sample size. 
            True loc is calculate based on that with the formula: 
            loc = rel_loc * number of elements in data. Defaults to 0.5.
        rel_scale (float, optional): Relative scale of normal distribution, 
            used to sample sample size. 
            True scale is calculate based on that with the formula: 
            scale = rel_scale * number of elements in data. Defaults to 0.5.

    Returns:
        list[any]: List with choosen elements.
    """
    elements_count = len(data)
    elements_indexes = range(elements_count - min_output_elements_num)
    elements_weights = scipy.stats.norm.pdf(
        elements_indexes,
        loc=(elements_count - min_output_elements_num) * rel_loc,
        scale=elements_count * rel_scale,
    )
    elements_weights /= sum(elements_weights)
    elements_weights = np.cumsum(elements_weights)
    output_elements_num = (
        random.random() > elements_weights
    ).sum() + min_output_elements_num
    output_elements = random.sample(data, k=output_elements_num)
    output_elements.sort()
    return output_elements
