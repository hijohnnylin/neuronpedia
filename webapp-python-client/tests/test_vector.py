import random
import pytest
from neuronpedia.np_vector import NPVector
from neuronpedia.requests.vector_request import VectorRequest
from neuronpedia.sample_data import GPT2_SMALL_RES_JB_JEDI_VECTOR


class TestNewVectorRequest:
    @pytest.fixture
    def vector_request(self):
        return VectorRequest()

    def test_new_and_deletevector(self, vector_request):
        label = "Jedi " + str(random.randint(0, 1000000))
        model_id = "gpt2-small"
        layer_num = 0
        hook_type = "hook_resid_pre"
        vector = GPT2_SMALL_RES_JB_JEDI_VECTOR
        steer_strength = 15.0

        # create it
        np_vector = NPVector.new(
            label=label,
            model_id=model_id,
            layer_num=layer_num,
            hook_type=hook_type,
            vector=vector,
            default_steer_strength=steer_strength,
        )

        assert np_vector.url is not None
        assert model_id in np_vector.url
        assert np_vector.model_id == model_id
        assert np_vector.source.startswith(f"{layer_num}")
        assert np_vector.label == label
        assert np_vector.hook_name.endswith(hook_type)
        assert all(abs(a - b) <= 0.0001 for a, b in zip(np_vector.values, vector))
        assert np_vector.default_steer_strength == steer_strength

        # delete it
        np_vector.delete()

    def test_get_owned(self, vector_request):

        # make a test vector
        label = "Jedi " + str(random.randint(0, 1000000))
        model_id = "gpt2-small"
        layer_num = 0
        hook_type = "hook_resid_pre"
        vector = GPT2_SMALL_RES_JB_JEDI_VECTOR
        steer_strength = 15.0

        # create it
        new_np_vector = vector_request.new(
            label=label,
            model_id=model_id,
            layer_num=layer_num,
            hook_type=hook_type,
            vector=vector,
            default_steer_strength=steer_strength,
        )

        # list all vectors
        np_vectors = vector_request.get_owned()
        assert len(np_vectors) > 0

        # check for new_np_vector in the list
        matching_vector = None
        for vector in np_vectors:
            print(vector)
            if vector == new_np_vector:
                matching_vector = vector
                break
        assert matching_vector is not None

        # delete the vector
        new_np_vector.delete()
