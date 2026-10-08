"""The engine point a steer at a TransformerLens hook name writes.

The hook vocabulary is this server's; ``SteeringSpec.at`` takes the points this returns.
"""

from __future__ import annotations

import pytest
from interp_engine import Address

from neuronpedia_inference.endpoints.steer.completion import steer_target


@pytest.mark.parametrize(
    ("hook", "want"),
    [
        ("blocks.0.hook_resid_pre", Address("embeddings")),
        ("blocks.4.hook_resid_pre", Address("resid_post", 3)),
        ("blocks.4.hook_resid_post", Address("resid_post", 4)),
        ("blocks.4.attn.hook_z", Address("z", 4)),
    ],
)
def test_a_hook_maps_to_the_point_its_steer_writes(hook: str, want: Address) -> None:
    assert steer_target(hook) == want


@pytest.mark.parametrize("hook", ["blocks.4.hook_mlp_out", "ln_final.hook_normalized"])
def test_a_hook_with_no_write_target_is_refused(hook: str) -> None:
    with pytest.raises(ValueError):
        steer_target(hook)
