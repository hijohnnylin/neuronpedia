"""TransformerLens has two names for the MLP output, and the SAEs read the post-norm one.

`blocks.{i}.hook_mlp_out` is the sublayer's residual *contribution*, so on gemma-2/3/4 and OLMo-2/3
it fires after the post-sublayer norm; `blocks.{i}.mlp.hook_out` is the raw module output. Both
collapse onto `mlp_out` when the engine's mapper is asked without a model, and this app cannot pass
one -- `has_sandwich_norms` reads a detected module tree and the vLLM client holds none. So the
adapter asks for the `*_post` point, which means the contribution on every architecture.

Nothing about getting this wrong raises. Reading `gemmascope-mlp-16k` off raw `mlp_out` reconstructs
at FVU 9.8 against 0.26 (worse than predicting the mean) and leaves the source silently dead: the
gemma-2-2b layer-4 feature whose dashboard tops out at 23.5 on "mass-production" fired at no
position in that text at all. That is what these tests exist to keep from coming back.
"""

from __future__ import annotations

import pytest
from interp_engine import Address

from neuronpedia_inference.engine_adapter import _capture_points, tlens_hook_to_point


class TestBlockLevelHooksAreTheContribution:
    def test_hook_mlp_out_is_the_post_norm_contribution(self):
        assert tlens_hook_to_point("blocks.4.hook_mlp_out") == Address("mlp_out_post", 4)

    def test_hook_attn_out_is_the_post_norm_contribution(self):
        assert tlens_hook_to_point("blocks.7.hook_attn_out") == Address("attn_out_post", 7)

    def test_the_layer_survives_the_upgrade(self):
        # The bug this guards is an off-by-a-sublayer, not an off-by-a-layer, but the point name and
        # the layer are rewritten in one place, so pin both.
        assert [tlens_hook_to_point(f"blocks.{i}.hook_mlp_out") for i in (0, 13, 25)] == [
            Address("mlp_out_post", 0),
            Address("mlp_out_post", 13),
            Address("mlp_out_post", 25),
        ]

    def test_capture_points_route_the_hook_without_a_normalize_step(self):
        # `_capture_points` is the path the endpoints actually take. The contribution is a real point,
        # so unlike `ln2.hook_normalized` it needs no post-capture arithmetic.
        (point,) = _capture_points(["blocks.4.hook_mlp_out"])
        assert point.hook == "blocks.4.hook_mlp_out"
        assert point.address == Address("mlp_out_post", 4)
        assert point.normalize is False


class TestTheRawOutputNamesAreUntouched:
    """TransformerLens' submodule spellings mean the raw module output on every architecture."""

    @pytest.mark.parametrize(
        ("hook", "expected"),
        [
            ("blocks.4.mlp.hook_out", Address("mlp_out", 4)),
            ("blocks.7.attn.hook_out", Address("attn_out", 7)),
        ],
    )
    def test_submodule_hooks_stay_raw(self, hook: str, expected: Address):
        # The whole reason the upgrade is keyed on the hook *name* rather than on the point the engine
        # returned: these two land on the same points as the block-level hooks above and must not be
        # rewritten with them.
        assert tlens_hook_to_point(hook) == expected

    @pytest.mark.parametrize(
        ("hook", "expected"),
        [
            ("blocks.0.hook_resid_pre", Address("resid_pre", 0)),
            ("blocks.5.hook_resid_post", Address("resid_post", 5)),
            ("blocks.5.hook_resid_mid", Address("resid_mid", 5)),
            ("blocks.5.attn.hook_z", Address("z", 5)),
            ("blocks.5.mlp.hook_post", Address("mlp_act", 5)),
            # An alias whose round trip also disagrees with the name asked for (`hook_mlp_in` comes
            # back as `mlp.hook_in`), so it exercises the guard that only sublayer *outputs* are
            # eligible for the upgrade.
            ("blocks.5.hook_mlp_in", Address("mlp_in", 5)),
        ],
    )
    def test_every_other_hook_is_plain_translation(self, hook: str, expected: Address):
        assert tlens_hook_to_point(hook) == expected


class TestTheTwinTableIsCheckedAgainstTheEngine:
    def test_each_pair_is_verified_at_import(self):
        # The table is a local copy of a relationship the engine owns, and a stale copy here would
        # silently re-serve the raw output. Importing the module runs this; calling it again is how a
        # reader sees that it is a real check rather than a comment.
        from neuronpedia_inference.engine_adapter import _assert_contribution_twins

        _assert_contribution_twins()

    def test_the_table_covers_both_sublayers(self):
        from neuronpedia_inference.engine_adapter import _BLOCK_LEVEL_CONTRIBUTION

        assert _BLOCK_LEVEL_CONTRIBUTION == {"mlp_out": "mlp_out_post", "attn_out": "attn_out_post"}
