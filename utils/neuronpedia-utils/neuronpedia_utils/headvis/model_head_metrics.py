from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ModelHeadMetrics:
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
    selfAttentionScore: Optional[float] = None
    prevTokenScore: Optional[float] = None
    patternEntropy: Optional[float] = None
    qkDistance: Optional[float] = None
    qkDistanceVariance: Optional[float] = None
    inductionScore: Optional[float] = None
    qkDistanceHistogram: Optional[Any] = None
    topQueryTokens: Optional[Any] = None
    topKeyTokens: Optional[Any] = None
    activationHistogram: Optional[Any] = None
    headStatistics: Optional[Any] = None
    createdAt: datetime = field(default_factory=datetime.now)
    updatedAt: datetime = field(default_factory=datetime.now)
