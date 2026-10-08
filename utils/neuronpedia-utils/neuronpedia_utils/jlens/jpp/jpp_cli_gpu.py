"""scripts/jpp_cli.py, with the router's PCA and k-means on the GPU, one layer at a time."""

import dataclasses
import runpy
import sys

import torch as t

if sys.argv[1:2] == ["fit-shard"]:
    sys.modules["fla"] = None  # RelP surgery needs the torch GatedDeltaNet path
from workspace_lens.routing import router as routing

_fit = routing.ActivationRouterCollection.fit


def _fit_on_gpu(self, activations_L_dict_PN):
    for layer, activations_PN in activations_L_dict_PN.items():
        _fit(self, {layer: activations_PN.cuda()})
        fitted = self.layer_routers_L_dict[layer]
        moved = {
            f.name: getattr(fitted, f.name).cpu() for f in dataclasses.fields(fitted)
        }
        self.layer_routers_L_dict[layer] = dataclasses.replace(fitted, **moved)
        t.cuda.empty_cache()


routing.ActivationRouterCollection.fit = _fit_on_gpu
sys.argv = ["scripts/jpp_cli.py", *sys.argv[1:]]
runpy.run_path("scripts/jpp_cli.py", run_name="__main__")
