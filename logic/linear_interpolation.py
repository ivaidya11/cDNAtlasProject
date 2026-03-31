

def linear_interpolation(target_min: int, target_max: int, scale_min: float, scale_max: float, dict: dict):
    scaled_dict = {}
    for key, value in dict.items():
        scaled_value = (((value - scale_min) * (target_max - target_min)) / (scale_max - scale_min)) + target_min
        scaled_dict[key] = scaled_value
    return scaled_dict
