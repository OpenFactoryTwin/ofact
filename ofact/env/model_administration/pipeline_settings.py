from enum import Enum

class PipelineType(Enum):
    DATA_INTEGRATION = "Data Integration"
    MODEL_GENERATION = "Model Generation"


class PipelineArrangement(Enum):
    SEQUENTIAL = "Sequential"
    AGGREGATION = "Aggregation"