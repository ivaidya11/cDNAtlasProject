def linear_interpolation(target_min: int, target_max: int, scale_min: float, scale_max: float, dict: dict):
    scaled_dict = {}
    for key, value in dict.items():
        scaled_value = (((value - scale_min) * (target_max - target_min)) / (scale_max - scale_min)) + target_min
        scaled_dict[key] = scaled_value
    return scaled_dict


hydropath_dict= {'A': 1.800,
                'R': -4.500,
                'N': -3.500,
                'D': -3.500,
                'C':  2.500,
                'Q': -3.500,
                'E': -3.500,
                'G': -0.400,
                'H': -3.200,
                'I':  4.500,
                'L':  3.800,
                'K': -3.900,
                'M':  1.900,
                'F':  2.800,
                'P': -1.600,
                'S': -0.800,
                'T': -0.700,
                'W': -0.900,
                'Y': -1.300,
                'V':  4.200}


kyte_minimum = -4.5
kyte_maximum = 4.5

target_minimum = 0
target_maximum = 1
scaled_hydrophobicity = {}
for aa, hydrophobicity in hydropath_dict.items():
    scaled_value = (((hydrophobicity - kyte_minimum) * (1))/(kyte_maximum-kyte_minimum)) + 0
    scaled_hydrophobicity[aa] = scaled_value

