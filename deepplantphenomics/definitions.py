from enum import Enum


class ProblemType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2
    SEMANTIC_SEGMETNATION = 3
    OBJECT_DETECTION = 4