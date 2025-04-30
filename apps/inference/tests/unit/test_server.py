from unittest.mock import patch

from neuronpedia_inference.server import parse_env_and_args


@patch.dict("os.environ", {"SAE_SETS": '["res-jb", "att-kk"]'})
def test_multiple_sae_sets():
    parsed_args = parse_env_and_args()
    assert parsed_args.sae_sets == ["res-jb", "att-kk"]
