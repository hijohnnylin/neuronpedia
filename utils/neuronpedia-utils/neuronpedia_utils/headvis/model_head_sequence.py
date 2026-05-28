from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModelHeadSequence:
    id: str
    modelId: str
    layer: int
    headIndex: int
    modelName: str
    datasetName: str
    nSequences: int
    seqLen: int
    dtype: str
    attnImplementation: str
    interval: int
    tokens: list[str]
    attentionIndices: list[int]
    attentionValues: list[float]
    maxActivation: float
    createdAt: datetime = field(default_factory=datetime.now)
    updatedAt: datetime = field(default_factory=datetime.now)
