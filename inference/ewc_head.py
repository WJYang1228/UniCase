"""Paper-faithful EWC head, with an explicit K-class extension.

The shared input representation is frozen by the caller, not by this head.
No ranking/anchor auxiliary loss and no clamp on the anchor-dispersion scale.
For K=2, softmax(-alpha * distances)[:, 1] exactly recovers the paper sigmoid.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EWCConfig:
    input_dim: int
    num_classes: int = 2
    hidden: int = 128
    space_dim: int = 64
    anchors: int = 8
    dropout: float = .1
    anchor_temperature: float = .2
    epsilon: float = 1e-8

    def __post_init__(self):
        if min(self.input_dim, self.hidden, self.space_dim) < 1:
            raise ValueError("positive dimensions required")
        if self.num_classes < 2 or self.anchors < 2:
            raise ValueError("at least two classes and anchors required")
        if not 0 <= self.dropout < 1 or min(self.anchor_temperature, self.epsilon) <= 0:
            raise ValueError("invalid dropout, temperature, or epsilon")


def _mlp(input_dim, hidden, output_dim, dropout):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden, output_dim))


class EWCHead(nn.Module):
    """One shared P, K isomorphic but independent M_c and anchor sets."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projector = _mlp(config.input_dim, config.hidden, config.space_dim, config.dropout)
        self.maps = nn.ModuleList([_mlp(config.space_dim, config.hidden, config.space_dim,
                                        config.dropout) for _ in range(config.num_classes)])
        self.anchors = nn.Parameter(torch.randn(config.num_classes, config.anchors,
                                                config.space_dim) * .02)
        # Softplus is an unclipped, positive parameterization; initial alpha=1.
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(math.expm1(1.))))

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha) + torch.finfo(self.raw_alpha.dtype).tiny

    def details(self, features):
        if features.ndim != 2 or features.shape[1] != self.config.input_dim:
            raise ValueError("features must have shape [N,input_dim]")
        shared = self.projector(features.float())
        distances, weights, scales = [], [], []
        for label, mapping in enumerate(self.maps):
            vector = F.normalize(mapping(shared).float(), dim=-1)
            anchor = F.normalize(self.anchors[label].float(), dim=-1)
            delta = (vector[:, None] - anchor[None]).square().sum(-1)
            assignment = (-delta / self.config.anchor_temperature).softmax(-1)
            # Unordered distinct pairs have the same mean as all ordered
            # distinct pairs. The denominator alone has stopped gradients.
            scale = torch.pdist(anchor, p=2).square().mean().detach()
            distances.append((assignment * delta).sum(-1) / (scale + self.config.epsilon))
            weights.append(assignment)
            scales.append(scale)
        distance = torch.stack(distances, dim=-1)
        return {"logits": -self.alpha * distance, "distances": distance,
                "anchor_weights": torch.stack(weights, dim=1),
                "anchor_scales": torch.stack(scales), "alpha": self.alpha}

    def forward(self, features):
        return self.details(features)["logits"]

    def definition(self):
        return dict(config=asdict(self.config), multiclass_extension=self.config.num_classes > 2,
                    shared_projector=True, independent_class_maps=True,
                    distance="sum_j softmax(-delta/tau)_j * delta_j / (stopgrad(nu_c)+epsilon)",
                    scale="mean squared Euclidean distance of all distinct unit-anchor pairs; no clamp",
                    logits="-shared_positive_alpha * normalized_class_distance",
                    classification_loss="cross_entropy; exactly BCE-equivalent for K=2",
                    auxiliary_rank_or_anchor_losses=False)
