import random
import scipy

import numpy as np

    
def normal_choice(data, rel_loc=1, rel_scale=1):  # 0.5, 0.26
    elements_count = len(data)

    elements_indexes = range(elements_count)
    elements_weights = scipy.stats.norm.pdf(
        elements_indexes, loc=elements_count * rel_loc, scale=elements_count * rel_scale
    )
    elements_weights /= sum(elements_weights)
    return random.choices(data, weights=elements_weights, k=1)


def random_length_normal_choices(data, min_output_elements_num=1, rel_loc=1, rel_scale=1):  # 4, 0.05, 0.25
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
