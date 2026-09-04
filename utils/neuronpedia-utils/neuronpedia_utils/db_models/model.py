from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Model:
    id: str
    instruct: bool
    creatorId: str
    displayNameShort: str = ""
    displayName: str = ""
    tlensId: Optional[str] = None
    dimension: Optional[int] = None
    visibility: str = "PUBLIC"
    defaultSourceSetName: Optional[str] = None
    defaultGraphSourceSetName: Optional[str] = None
    defaultSourceId: Optional[str] = None
    inferenceEnabled: bool = True
    layers: int = 0
    neuronsPerLayer: int = 0
    createdAt: datetime = field(default_factory=datetime.now)
    owner: str = ""
    updatedAt: datetime = field(default_factory=datetime.now)
    website: Optional[str] = None
