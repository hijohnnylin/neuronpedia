# Copyright 2026 Anthropic PBC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Jacobian lens — fitting and inference for decoder transformers.

The Jacobian lens reads out an early-layer residual ``h_ℓ`` by linearly
transporting it into the final-layer basis with the average input–output
Jacobian, then decoding with the model's own unembedding::

    lens_ℓ(h)  =  unembed( J̄_ℓ @ h )

See the *Verbalizable Workspace* paper for background and the ``demo/``
directory for an end-to-end walkthrough.

The library is typed against :class:`LensModel` — a small protocol any model
can implement. :func:`from_hf` wraps an already-loaded HuggingFace model::

    import torch, transformers, jlens

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = jlens.from_hf(hf_model, tokenizer, compile=True)

    lens = jlens.fit(model, prompts=my_prompts)
    lens.save("lens.pt")
"""

from jlens._logging import configure_logging
from jlens.fitting import FitProgress, fit, jacobian_for_prompt
from jlens.hf import HFLensModel, Layout, from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

__all__ = [
    "ActivationRecorder",
    "FitProgress",
    "HFLensModel",
    "JacobianLens",
    "Layout",
    "LensModel",
    "configure_logging",
    "fit",
    "from_hf",
    "jacobian_for_prompt",
]
