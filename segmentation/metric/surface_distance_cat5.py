
from metric.surface_distance_avg import surface_distance_calc


def surface_distance_cat5(output, target, misc):
    return surface_distance_calc(output, target, misc, 5)