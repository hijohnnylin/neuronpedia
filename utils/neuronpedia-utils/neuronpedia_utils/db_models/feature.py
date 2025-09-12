from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Feature:
    modelId: str
    layer: str
    index: str
    creatorId: Optional[str] = None
    createdAt: datetime = field(default_factory=datetime.now)
    maxActApprox: Optional[float] = 0
    hasVector: bool = False
    vector: List[float] = field(default_factory=list)
    vectorLabel: Optional[str] = None
    vectorDefaultSteerStrength: Optional[float] = 10
    hookName: Optional[str] = None
    topkCosSimIndices: List[int] = field(default_factory=list)
    topkCosSimValues: List[float] = field(default_factory=list)
    neuron_alignment_indices: List[int] = field(default_factory=list)
    neuron_alignment_values: List[float] = field(default_factory=list)
    neuron_alignment_l1: List[float] = field(default_factory=list)
    correlated_neurons_indices: List[int] = field(default_factory=list)
    correlated_neurons_pearson: List[float] = field(default_factory=list)
    correlated_neurons_l1: List[float] = field(default_factory=list)
    correlated_features_indices: List[int] = field(default_factory=list)
    correlated_features_pearson: List[float] = field(default_factory=list)
    correlated_features_l1: List[float] = field(default_factory=list)
    neg_str: List[str] = field(default_factory=list)
    neg_values: List[float] = field(default_factory=list)
    pos_str: List[str] = field(default_factory=list)
    pos_values: List[float] = field(default_factory=list)
    frac_nonzero: float = 0
    freq_hist_data_bar_heights: List[float] = field(default_factory=list)
    freq_hist_data_bar_values: List[float] = field(default_factory=list)
    logits_hist_data_bar_heights: List[float] = field(default_factory=list)
    logits_hist_data_bar_values: List[float] = field(default_factory=list)
    decoder_weights_dist: List[float] = field(default_factory=list)

    sourceSetName: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
