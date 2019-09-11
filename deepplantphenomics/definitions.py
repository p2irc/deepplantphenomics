from enum import Enum


class ProblemType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2
    SEMANTIC_SEGMETNATION = 3
    OBJECT_DETECTION = 4
    OBJECT_COUNTING = 5
    HEATMAP_COUNTING = 6


class AugmentationType(Enum):
    FLIP_HOR = 1
    FLIP_VER = 2
    CROP = 3
    CONTRAST_BRIGHT = 4
    ROTATE = 5
